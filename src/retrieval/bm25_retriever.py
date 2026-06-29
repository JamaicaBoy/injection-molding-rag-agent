from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import load_corpus_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = load_corpus_config().chunks_path
PREVIEW_LENGTH = 200

QUERY_EXPANSIONS = {
    "注塑": ["injection molding", "injection moulding"],
    "翘曲": ["warpage", "warping"],
    "工艺参数": ["process parameter", "melt temperature", "mold temperature", "injection speed"],
    "保压压力": ["packing pressure", "holding pressure"],
    "缩水": ["shrinkage", "sink mark"],
    "机器学习": ["machine learning", "deep learning", "neural network"],
    "质量预测": ["quality prediction", "defect prediction"],
    "透过率": ["transmittance", "transparency", "optical"],
    "论文": ["paper", "study", "research"],
}


def text_preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    visible_text = "".join(character for character in text if not unicodedata.category(character).startswith("C"))
    return re.sub(r"\s+", " ", visible_text).strip()[:limit]


def tokenize(text: str, expand_query: bool = False) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    if expand_query:
        expansions = [term for key, terms in QUERY_EXPANSIONS.items() if key in normalized for term in terms]
        normalized = f"{normalized} {' '.join(expansions)}"

    tokens = re.findall(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*", normalized)
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file does not exist: {chunks_path}")
    with chunks_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def result_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata") or {})
    for field in ("file_name", "year", "page_start", "page_end"):
        metadata.setdefault(field, chunk.get(field, ""))
    return metadata


class BM25Retriever:
    def __init__(self, chunks_path: Path | None = None) -> None:
        self.chunks_path = Path(chunks_path or load_corpus_config().chunks_path)
        self.chunks = load_chunks(self.chunks_path)
        if not self.chunks:
            raise ValueError(f"No chunks found in {self.chunks_path}")
        corpus = [tokenize(str(chunk.get("text", ""))) for chunk in self.chunks]
        self.index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_tokens = tokenize(query, expand_query=True)
        if not query_tokens:
            return []

        raw_scores = np.asarray(self.index.get_scores(query_tokens), dtype=float)
        result_count = min(top_k, len(self.chunks))
        ranked_indices = np.argsort(-raw_scores, kind="stable")[:result_count]
        max_score = max(float(raw_scores[ranked_indices[0]]), 0.0) if len(ranked_indices) else 0.0

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            chunk = self.chunks[int(index)]
            raw_score = float(raw_scores[index])
            metadata = result_metadata(chunk)
            metadata["bm25_raw_score"] = raw_score
            results.append(
                {
                    "chunk_id": str(chunk.get("chunk_id", "")),
                    "paper_id": str(chunk.get("paper_id", "")),
                    "title": str(chunk.get("title", "")),
                    "section_name": str(chunk.get("section_name", "")),
                    "chunk_type": str(chunk.get("chunk_type", "")),
                    "score": max(raw_score, 0.0) / max_score if max_score else 0.0,
                    "source": "bm25",
                    "text_preview": text_preview(str(chunk.get("text", ""))),
                    "metadata": metadata,
                }
            )
        return results
