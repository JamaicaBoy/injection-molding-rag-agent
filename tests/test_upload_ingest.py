from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import fitz

from src.index.build_vector_index import runtime_persist_dir
from src.index.incremental_index import (
    add_uploaded_chunks,
    clear_upload_collection,
    upload_collection_name,
)
from src.ingest.ingest_uploaded_paper import (
    ingest_uploaded_papers,
    safe_pdf_filename,
    save_uploaded_pdf,
)
from src.retrieval.multi_collection_retriever import MultiCollectionRetriever


class TinyEmbeddingModel:
    model_name = "local-test-embedding"

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        del batch_size
        return [
            [float(len(text) % 7), float(len(text) % 11), 0.5, 1.0]
            for text in texts
        ]


class FakeDenseRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def search(
        self, query: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        del query, where
        return self.results[:top_k]


class FakeReranker:
    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        del query
        return sorted(candidates, key=lambda item: -float(item["score"]))[:top_n]


def make_pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    text = (
        "Abstract\n"
        "Injection molding packing pressure affects flash, shrinkage, and final part quality. "
        "This uploaded study evaluates pressure effects using controlled experiments.\n\n"
        "Results\n"
        "The experiment observed that excessive packing pressure increased flash risk while "
        "insufficient packing pressure increased shrinkage. Process settings require validation.\n\n"
        "Conclusion\n"
        "Packing pressure should be optimized with mold temperature and material behavior."
    )
    page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


def test_upload_parse_clean_chunk_and_incremental_index(
    tmp_path: Path, capsys: Any
) -> None:
    session_id = "session-smoke"
    pdf_path = save_uploaded_pdf(
        "../../unsafe paper name.pdf",
        make_pdf_bytes(),
        session_id=session_id,
        upload_root=tmp_path / "uploads",
    )
    assert pdf_path.name == "unsafe_paper_name.pdf"
    assert pdf_path.parent.name == "raw"

    ingest_stats = ingest_uploaded_papers(
        session_id,
        pdf_paths=[pdf_path],
        upload_root=tmp_path / "uploads",
    )
    assert ingest_stats.parsed_count == 1
    assert ingest_stats.failed_count == 0
    assert ingest_stats.section_count > 0
    assert ingest_stats.chunk_count > 0
    chunks = [
        json.loads(line)
        for line in Path(ingest_stats.chunks_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(chunk["paper_id"].startswith("paper_") for chunk in chunks)

    persist_dir = tmp_path / "chroma"
    index_stats = add_uploaded_chunks(
        chunks_path=Path(ingest_stats.chunks_path),
        persist_dir=persist_dir,
        session_id=session_id,
        base_collection_name="base_collection",
        embedding_model=TinyEmbeddingModel(),
        batch_size=2,
        registry_path=tmp_path / "index_registry.json",
        lock_dir=tmp_path / "locks",
    )
    collection_name = upload_collection_name(session_id)
    client = chromadb.PersistentClient(path=str(runtime_persist_dir(persist_dir)))
    assert index_stats["upload_collection_name"] == collection_name
    assert client.get_collection(collection_name).count() == ingest_stats.chunk_count
    assert clear_upload_collection(
        persist_dir=persist_dir,
        session_id=session_id,
        registry_path=tmp_path / "index_registry.json",
        lock_dir=tmp_path / "locks",
    ) is True

    terminal_output = capsys.readouterr().out
    assert "excessive packing pressure increased flash risk" not in terminal_output


def test_safe_pdf_filename_rejects_non_pdf() -> None:
    assert safe_pdf_filename("folder/论文 01.PDF") == "论文_01.pdf"
    try:
        safe_pdf_filename("paper.txt")
    except ValueError as exc:
        assert "PDF" in str(exc)
    else:
        raise AssertionError("Expected non-PDF filename to be rejected")


def test_multi_collection_retriever_merges_deduplicates_and_reranks() -> None:
    base = FakeDenseRetriever(
        [
            {"chunk_id": "shared", "paper_id": "base", "score": 0.6, "metadata": {}},
            {"chunk_id": "base-only", "paper_id": "base", "score": 0.5, "metadata": {}},
        ]
    )
    uploads = FakeDenseRetriever(
        [
            {"chunk_id": "shared", "paper_id": "upload", "score": 0.8, "metadata": {}},
            {"chunk_id": "upload-only", "paper_id": "upload", "score": 0.7, "metadata": {}},
        ]
    )
    retriever = MultiCollectionRetriever(
        base,
        uploads,
        base_collection_name="base_collection",
        upload_collection_name="upload_collection",
        reranker=FakeReranker(),
    )

    results = retriever.search("packing pressure", top_k=3)

    assert [item["chunk_id"] for item in results] == ["shared", "upload-only", "base-only"]
    assert len({item["chunk_id"] for item in results}) == 3
    assert results[0]["metadata"]["collection_name"] == "upload_collection"
