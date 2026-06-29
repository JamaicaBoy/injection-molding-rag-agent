from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


def result_score(result: dict[str, Any]) -> float:
    return float(result.get("rerank_score", result.get("score", 0.0)) or 0.0)


def retrieval_result_to_document(result: dict[str, Any]) -> Document:
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "paper_id": str(result.get("paper_id", "")),
            "title": str(result.get("title", "")),
            "section": str(result.get("section_name", metadata.get("section", ""))),
            "score": result_score(result),
            "chunk_id": str(result.get("chunk_id", "")),
        }
    )
    content = str(result.get("text_preview") or result.get("text") or "")
    return Document(page_content=content, metadata=metadata)


def retrieval_results_to_documents(results: list[dict[str, Any]]) -> list[Document]:
    return [retrieval_result_to_document(result) for result in results]


def documents_to_evidence(documents: list[Document]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        metadata = dict(document.metadata)
        evidence.append(
            {
                "evidence_id": f"E{index}",
                "chunk_id": str(metadata.get("chunk_id", "")),
                "paper_id": str(metadata.get("paper_id", "")),
                "title": str(metadata.get("title", "")),
                "section_name": str(metadata.get("section", "")),
                "chunk_type": str(metadata.get("chunk_type", "")),
                "score": float(metadata.get("score", 0.0) or 0.0),
                "text_preview": document.page_content[:600],
                "metadata": metadata,
            }
        )
    return evidence
