from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import chromadb

from src.config import load_corpus_config
from src.index.build_vector_index import runtime_persist_dir
from src.retrieval.bm25_retriever import text_preview


_ACTIVE_CORPUS = load_corpus_config()
DEFAULT_PERSIST_DIR = _ACTIVE_CORPUS.vector_persist_dir
DEFAULT_COLLECTION = _ACTIVE_CORPUS.collection_name


class QueryEncoder(Protocol):
    def encode_query(self, query: str) -> list[float]:
        ...


class SentenceTransformerQueryEncoder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for dense retrieval.") from exc

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dense retrieval model: {model_name}") from exc

    def encode_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return embedding.tolist()


class SubprocessSentenceTransformerQueryEncoder:
    """Encode one query in a short-lived process so model RAM is released afterward."""

    def __init__(
        self,
        model_name: str,
        timeout: float = 240.0,
        max_attempts: int = 2,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)

    def encode_query(self, query: str) -> list[float]:
        code = """
import json
import sys
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(sys.argv[1])
embedding = model.encode([sys.argv[2]], normalize_embeddings=True, show_progress_bar=False)[0]
print(json.dumps(embedding.tolist()))
"""
        environment = dict(os.environ)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        result = None
        for attempt in range(1, self.max_attempts + 1):
            result = subprocess.run(
                [sys.executable, "-c", code, self.model_name, query],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                env=environment,
                check=False,
            )
            if result.returncode == 0:
                break
            if attempt < self.max_attempts:
                time.sleep(1.0)
        if result is None or result.returncode != 0:
            stderr_lines = result.stderr.strip().splitlines() if result is not None else []
            detail = " | ".join(stderr_lines[-4:]) or "unknown error"
            return_code = result.returncode if result is not None else "not_started"
            raise RuntimeError(
                f"Dense query encoder subprocess failed after {self.max_attempts} attempts "
                f"(returncode={return_code}): {detail}"
            )
        try:
            embedding = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError("Dense query encoder returned invalid output.") from exc
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Dense query encoder returned an empty embedding.")
        return [float(value) for value in embedding]


class DenseRetriever:
    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        model_name: str | None = None,
        encoder: QueryEncoder | None = None,
        collection: Any | None = None,
        subprocess_encoder: bool = False,
    ) -> None:
        corpus = load_corpus_config()
        self.persist_dir = Path(persist_dir or corpus.vector_persist_dir)
        self.collection_name = collection_name or corpus.collection_name
        if collection is None:
            if not self.persist_dir.exists():
                raise FileNotFoundError(f"Chroma persist_dir does not exist: {self.persist_dir}")
            runtime_dir = runtime_persist_dir(self.persist_dir)
            client = chromadb.PersistentClient(path=str(runtime_dir))
            collection = client.get_collection(self.collection_name)
        self.collection = collection

        collection_metadata = self.collection.metadata or {}
        self.model_name = str(
            model_name
            or collection_metadata.get("embedding_local_path")
            or collection_metadata.get("embedding_model")
            or ""
        )
        if not self.model_name and encoder is None:
            raise ValueError("The Chroma collection does not record an embedding model.")
        if encoder is not None:
            self.encoder = encoder
        elif subprocess_encoder:
            self.encoder = SubprocessSentenceTransformerQueryEncoder(self.model_name)
        else:
            self.encoder = SentenceTransformerQueryEncoder(self.model_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or self.collection.count() == 0:
            return []

        query_embedding = self.encoder.encode_query(query)
        query_args: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        response = self.collection.query(**query_args)

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        results: list[dict[str, Any]] = []
        for chroma_id, document, metadata_row, distance in zip(ids, documents, metadatas, distances):
            metadata = dict(metadata_row or {})
            numeric_distance = max(float(distance), 0.0)
            metadata["dense_distance"] = numeric_distance
            results.append(
                {
                    "chunk_id": str(metadata.get("chunk_id") or chroma_id),
                    "paper_id": str(metadata.get("paper_id", "")),
                    "title": str(metadata.get("title", "")),
                    "section_name": str(metadata.get("section_name", "")),
                    "chunk_type": str(metadata.get("chunk_type", "")),
                    "score": 1.0 / (1.0 + numeric_distance),
                    "source": "dense",
                    "text_preview": text_preview(str(document or "")),
                    "metadata": metadata,
                }
            )
        return results
