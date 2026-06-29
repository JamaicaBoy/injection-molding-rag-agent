from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import Reranker


class MultiCollectionRetriever:
    """Search a base and session upload collection, then merge by chunk_id."""

    def __init__(
        self,
        base_retriever: Any,
        upload_retriever: Any | None = None,
        *,
        base_collection_name: str = "base",
        upload_collection_name: str = "uploads",
        reranker: Any | None = None,
        rerank_results: bool = True,
    ) -> None:
        self.base_retriever = base_retriever
        self.upload_retriever = upload_retriever
        self.base_collection_name = base_collection_name
        self.upload_collection_name = upload_collection_name
        self.reranker = reranker or Reranker(mode="rule")
        self.rerank_results = rerank_results

    @classmethod
    def from_chroma(
        cls,
        *,
        persist_dir: Path,
        base_collection_name: str,
        upload_collection_name: str,
        base_retriever: DenseRetriever | None = None,
        reranker: Any | None = None,
        rerank_results: bool = True,
    ) -> "MultiCollectionRetriever":
        base = base_retriever or DenseRetriever(
            persist_dir=persist_dir,
            collection_name=base_collection_name,
        )
        try:
            upload = DenseRetriever(
                persist_dir=persist_dir,
                collection_name=upload_collection_name,
                encoder=base.encoder,
            )
        except Exception:
            upload = None
        return cls(
            base,
            upload,
            base_collection_name=base_collection_name,
            upload_collection_name=upload_collection_name,
            reranker=reranker,
            rerank_results=rerank_results,
        )

    @staticmethod
    def _tag(result: dict[str, Any], collection_name: str) -> dict[str, Any]:
        output = dict(result)
        output["metadata"] = dict(result.get("metadata") or {})
        output["metadata"]["collection_name"] = collection_name
        output["source"] = f"dense:{collection_name}"
        return output

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        candidate_k = max(top_k * 2, top_k)
        base_results = [
            self._tag(item, self.base_collection_name)
            for item in self.base_retriever.search(query, top_k=candidate_k, where=where)
        ]
        upload_results: list[dict[str, Any]] = []
        if self.upload_retriever is not None:
            upload_results = [
                self._tag(item, self.upload_collection_name)
                for item in self.upload_retriever.search(query, top_k=candidate_k, where=where)
            ]

        merged: dict[str, dict[str, Any]] = {}
        for result in [*base_results, *upload_results]:
            chunk_id = str(result.get("chunk_id", ""))
            if not chunk_id:
                continue
            existing = merged.get(chunk_id)
            if existing is None or float(result.get("score", 0.0)) > float(
                existing.get("score", 0.0)
            ):
                merged[chunk_id] = result
            elif existing is not None:
                collections = {
                    str(existing.get("metadata", {}).get("collection_name", "")),
                    str(result.get("metadata", {}).get("collection_name", "")),
                }
                existing["metadata"]["matched_collections"] = ";".join(
                    sorted(item for item in collections if item)
                )
        candidates = sorted(
            merged.values(),
            key=lambda item: (-float(item.get("score", 0.0)), str(item.get("chunk_id", ""))),
        )
        if self.rerank_results and candidates:
            return self.reranker.rerank(query, candidates, top_n=top_k)
        return candidates[:top_k]
