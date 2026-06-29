from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import chromadb
import yaml
from tqdm import tqdm

from src.config import load_corpus_config
from src.index.index_lock import DEFAULT_LOCK_DIR, index_write_lock
from src.index.index_registry import (
    DEFAULT_REGISTRY_PATH,
    infer_corpus_mode,
    register_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BUILD_CORPUS = load_corpus_config(prefer_configured=True)
DEFAULT_CHUNKS = _BUILD_CORPUS.chunks_path
DEFAULT_PERSIST_DIR = _BUILD_CORPUS.vector_persist_dir
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "retrieval_config.yaml"
DEFAULT_COLLECTION = _BUILD_CORPUS.collection_name
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_HASHING_DIM = 384
REQUIRED_METADATA_FIELDS = [
    "chunk_id",
    "paper_id",
    "title",
    "year",
    "section_name",
    "chunk_type",
    "page_start",
    "page_end",
    "file_name",
]


class EmbeddingModel(Protocol):
    model_name: str

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is not available. Run `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
            ) from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load local sentence-transformers model "
                f"`{model_name}`. If this is a HuggingFace download problem, try one of: "
                "`BAAI/bge-small-zh-v1.5`, `BAAI/bge-small-en-v1.5`, or pre-download "
                "`BAAI/bge-m3` with a working network/proxy. No paid API was called."
            ) from exc

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class HashingEmbeddingModel:
    def __init__(self, n_features: int = DEFAULT_HASHING_DIM) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.model_name = f"hashing-vectorizer-{n_features}"
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()


def load_embedding_model_name(config_path: Path = DEFAULT_CONFIG) -> str:
    if not config_path.exists():
        return DEFAULT_MODEL
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return str(config.get("embedding", {}).get("model") or DEFAULT_MODEL)


def read_chunks(chunks_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]
    if limit is not None:
        return chunks[:limit]
    return chunks


def metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False)


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata = {
        "chunk_id": chunk.get("chunk_id", ""),
        "paper_id": chunk.get("paper_id", ""),
        "title": chunk.get("title", ""),
        "year": chunk.get("year", ""),
        "section_name": chunk.get("section_name", ""),
        "chunk_type": chunk.get("chunk_type", ""),
        "page_start": chunk.get("page_start", ""),
        "page_end": chunk.get("page_end", ""),
        "file_name": chunk.get("file_name", ""),
    }
    return {key: metadata_value(value) for key, value in metadata.items()}


def assign_unique_index_ids(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    indexed_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        indexed_chunk = dict(chunk)
        base_id = str(chunk["chunk_id"])
        seen[base_id] = seen.get(base_id, 0) + 1
        indexed_chunk["_chroma_id"] = base_id if seen[base_id] == 1 else f"{base_id}__dup{seen[base_id]}"
        indexed_chunks.append(indexed_chunk)
    return indexed_chunks


def batched(items: list[dict[str, Any]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def reset_collection(client: chromadb.PersistentClient, collection_name: str) -> None:
    existing_names = collection_names(client)
    if collection_name in existing_names:
        client.delete_collection(collection_name)


def collection_names(client: chromadb.PersistentClient) -> list[str]:
    collections = client.list_collections()
    names: list[str] = []
    for collection in collections:
        if isinstance(collection, str):
            names.append(collection)
        else:
            names.append(collection.name)
    return names


def runtime_persist_dir(persist_dir: Path) -> Path:
    try:
        return persist_dir.resolve()
    except OSError:
        return persist_dir


def ensure_persist_dir(persist_dir: Path) -> Path:
    """Create a Chroma directory without recreating an existing Windows junction."""
    persist_dir = Path(persist_dir)
    if persist_dir.exists():
        return persist_dir
    if os.path.lexists(persist_dir):
        runtime_persist_dir(persist_dir).mkdir(parents=True, exist_ok=True)
        return persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


def _build_index_unlocked(
    chunks_path: Path = DEFAULT_CHUNKS,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str | None = None,
    reset: bool = False,
    limit: int | None = None,
    batch_size: int = 32,
    backend: str = "sentence-transformers",
    hashing_dim: int = DEFAULT_HASHING_DIM,
    embedding_model: EmbeddingModel | None = None,
    report_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if reset and resume:
        raise ValueError("reset and resume cannot be used together")
    started_at = datetime.now().astimezone()
    started_timer = perf_counter()
    chunks = read_chunks(chunks_path, limit=limit)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    chunks = assign_unique_index_ids(chunks)
    total_input_chunk_count = len(chunks)
    input_paper_count = len(
        {str(chunk.get("paper_id", "")) for chunk in chunks if chunk.get("paper_id")}
    )

    ensure_persist_dir(persist_dir)
    runtime_dir = runtime_persist_dir(persist_dir)
    client = chromadb.PersistentClient(path=str(runtime_dir))
    if reset:
        reset_collection(client, collection_name)

    if embedding_model is not None:
        model = embedding_model
    elif backend == "hashing":
        model = HashingEmbeddingModel(n_features=hashing_dim)
    elif backend == "sentence-transformers":
        model = SentenceTransformerEmbeddingModel(model_name or load_embedding_model_name())
    else:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    collection_metadata = {
        "embedding_model": model.model_name,
        "embedding_backend": backend,
        "source": str(chunks_path),
    }
    model_path = Path(model.model_name).expanduser()
    if model_path.is_dir():
        collection_metadata["embedding_local_path"] = str(model_path.resolve())
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=collection_metadata,
    )
    if resume:
        existing_ids = set(collection.get(include=[]).get("ids", []))
        chunks = [chunk for chunk in chunks if str(chunk["_chroma_id"]) not in existing_ids]

    first_embedding_dim: int | None = None
    # Preserve the requested model batch size while amortizing encode/upsert overhead.
    window_size = max(64, batch_size * 32)
    with tqdm(total=len(chunks), desc="Embedding and indexing", unit="chunk") as progress:
        for chunk_batch in batched(chunks, batch_size=window_size):
            texts = [str(chunk.get("text", "")) for chunk in chunk_batch]
            embeddings = model.encode_texts(texts, batch_size=batch_size)
            if embeddings and first_embedding_dim is None:
                first_embedding_dim = len(embeddings[0])
            collection.upsert(
                ids=[str(chunk["_chroma_id"]) for chunk in chunk_batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[chroma_metadata(chunk) for chunk in chunk_batch],
            )
            progress.update(len(chunk_batch))

    if first_embedding_dim is None and collection.count() > 0:
        stored = collection.get(limit=1, include=["embeddings"]).get("embeddings")
        if stored is not None and len(stored) > 0:
            first_embedding_dim = len(stored[0])

    first_metadatas = collection.get(limit=3, include=["metadatas"]).get("metadatas", [])
    stats = {
        "collection_name": collection_name,
        "collections": len(collection_names(client)),
        "collection_count": collection.count(),
        "embedding_dim": first_embedding_dim,
        "first_metadatas": first_metadatas,
        "persist_dir": str(persist_dir),
        "runtime_persist_dir": str(runtime_dir),
        "model_name": model.model_name,
        "embedding_backend": backend,
        "input_chunk_count": total_input_chunk_count,
        "paper_count": input_paper_count,
        "processed_this_run": len(chunks),
        "batch_size": batch_size,
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_seconds": round(perf_counter() - started_timer, 2),
    }
    if report_path is not None:
        write_index_report(report_path, stats)
    print_index_summary(stats)
    return stats


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str | None = None,
    reset: bool = False,
    limit: int | None = None,
    batch_size: int = 32,
    backend: str = "sentence-transformers",
    hashing_dim: int = DEFAULT_HASHING_DIM,
    embedding_model: EmbeddingModel | None = None,
    report_path: Path | None = None,
    resume: bool = False,
    corpus_mode: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    lock_dir: Path = DEFAULT_LOCK_DIR,
    lock_timeout: float = 0.0,
) -> dict[str, Any]:
    with index_write_lock(
        Path(persist_dir),
        collection_name,
        lock_dir=Path(lock_dir),
        timeout=lock_timeout,
    ):
        stats = _build_index_unlocked(
            chunks_path=Path(chunks_path),
            persist_dir=Path(persist_dir),
            collection_name=collection_name,
            model_name=model_name,
            reset=reset,
            limit=limit,
            batch_size=batch_size,
            backend=backend,
            hashing_dim=hashing_dim,
            embedding_model=embedding_model,
            report_path=report_path,
            resume=resume,
        )
        record = register_index(
            corpus_mode=corpus_mode or infer_corpus_mode(collection_name),
            collection_name=collection_name,
            chunks_path=Path(chunks_path),
            persist_dir=Path(persist_dir),
            paper_count=int(stats.get("paper_count", 0)),
            chunk_count=int(stats.get("collection_count", 0)),
            embedding_model=str(stats.get("model_name", "")),
            built_at=str(stats.get("completed_at", "")) or None,
            registry_path=Path(registry_path),
        )
        stats["index_version"] = record["version"]
        stats["corpus_mode"] = record["corpus_mode"]
        if report_path is not None:
            write_index_report(Path(report_path), stats)
        return stats


def write_index_report(report_path: Path, stats: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Corpus Vector Index Report",
        "",
        "- status: completed",
        f"- index_version: {stats.get('index_version', 'pending')}",
        f"- corpus_mode: {stats.get('corpus_mode', 'unknown')}",
        f"- chunks_indexed: {stats['input_chunk_count']}",
        f"- chunks_processed_this_run: {stats['processed_this_run']}",
        f"- vector_count: {stats['collection_count']}",
        f"- embedding_backend: {stats['embedding_backend']}",
        f"- embedding_model: {stats['model_name']}",
        f"- embedding_dimension: {stats['embedding_dim']}",
        f"- collection_name: {stats['collection_name']}",
        f"- persist_dir: {stats['persist_dir']}",
        f"- runtime_persist_dir: {stats['runtime_persist_dir']}",
        f"- batch_size: {stats['batch_size']}",
        f"- started_at: {stats['started_at']}",
        f"- completed_at: {stats['completed_at']}",
        f"- elapsed_seconds: {stats['elapsed_seconds']}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_index_summary(stats: dict[str, Any]) -> None:
    print(f"collection_name: {stats['collection_name']}")
    print(f"collection_count: {stats['collections']}")
    print(f"vector_count: {stats['collection_count']}")
    print(f"processed_this_run: {stats['processed_this_run']}")
    print(f"embedding_dimension: {stats['embedding_dim']}")
    print(f"elapsed_seconds: {stats['elapsed_seconds']}")
    print("first_3_metadata:")
    for metadata in stats["first_metadatas"]:
        print(f"  {metadata}")


def print_build_stats(stats: dict[str, Any]) -> None:
    print(f"collection_name: {stats['collection_name']}")
    print(f"collection 数量: {stats['collections']}")
    print(f"当前 collection chunk 数: {stats['collection_count']}")
    print(f"embedding 维度: {stats['embedding_dim']}")
    print("前 3 条 metadata:")
    for metadata in stats["first_metadatas"]:
        print(f"  {metadata}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Chroma vector index from RAG chunks.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--persist_dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--corpus_mode", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--backend",
        choices=["sentence-transformers", "hashing"],
        default="sentence-transformers",
        help="Default is local sentence-transformers. Use hashing only as a no-download local fallback.",
    )
    parser.add_argument("--hashing_dim", type=int, default=DEFAULT_HASHING_DIM)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional Markdown report path for index build statistics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_index(
        chunks_path=args.chunks,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        reset=args.reset,
        limit=args.limit,
        batch_size=args.batch_size,
        backend=args.backend,
        hashing_dim=args.hashing_dim,
        report_path=args.report,
        resume=args.resume,
        corpus_mode=args.corpus_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
