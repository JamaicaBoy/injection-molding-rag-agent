from __future__ import annotations

from typing import Any

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def validate_weights(dense_weight: float, bm25_weight: float) -> tuple[float, float]:
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("Retriever weights must be non-negative.")
    total = dense_weight + bm25_weight
    if total <= 0:
        raise ValueError("At least one retriever weight must be greater than zero.")
    return dense_weight / total, bm25_weight / total


def merge_results(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict[str, Any]]:
    dense_weight, bm25_weight = validate_weights(dense_weight, bm25_weight)
    merged: dict[str, dict[str, Any]] = {}

    for source, weight, results in (
        ("bm25", bm25_weight, bm25_results),
        ("dense", dense_weight, dense_results),
    ):
        for result in results:
            chunk_id = str(result["chunk_id"])
            entry = merged.setdefault(
                chunk_id,
                {
                    **result,
                    "score": 0.0,
                    "source": "",
                    "metadata": dict(result.get("metadata") or {}),
                    "_scores": {},
                },
            )
            source_score = float(result.get("score", 0.0))
            previous_score = float(entry["_scores"].get(source, 0.0))
            if source_score > previous_score:
                entry["score"] += weight * (source_score - previous_score)
                entry["_scores"][source] = source_score
            entry["metadata"].update(result.get("metadata") or {})

    for entry in merged.values():
        sources = [source for source in ("bm25", "dense") if source in entry["_scores"]]
        entry["source"] = "+".join(sources)
        entry["metadata"]["bm25_score"] = float(entry["_scores"].get("bm25", 0.0))
        entry["metadata"]["dense_score"] = float(entry["_scores"].get("dense", 0.0))
        entry.pop("_scores")

    return sorted(merged.values(), key=lambda result: (-float(result["score"]), result["chunk_id"]))[:top_k]


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        dense_retriever: DenseRetriever | None = None,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.dense_weight, self.bm25_weight = validate_weights(dense_weight, bm25_weight)

    def search(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> list[dict[str, Any]]:
        candidate_k = candidate_k or max(top_k * 2, top_k)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        return merge_results(
            bm25_results,
            dense_results,
            top_k=top_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )

