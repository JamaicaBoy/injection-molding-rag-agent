import json
from pathlib import Path
from typing import Any

from src.retrieval.bm25_retriever import BM25Retriever, text_preview
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import merge_results
from src.retrieval.retrieval_debug import inspect_retrieval_state


def write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "warpage",
            "paper_id": "paper_1",
            "title": "Warpage control",
            "section_name": "Results",
            "chunk_type": "text",
            "text": "Warpage is affected by melt temperature, mold temperature and injection speed.",
            "file_name": "warpage.pdf",
            "year": "2024",
            "page_start": 1,
            "page_end": 1,
            "metadata": {},
        },
        {
            "chunk_id": "shrinkage",
            "paper_id": "paper_2",
            "title": "Shrinkage study",
            "section_name": "Conclusion",
            "chunk_type": "text",
            "text": "Packing pressure reduces shrinkage and sink marks in injection molding.",
            "file_name": "shrinkage.pdf",
            "year": "2023",
            "page_start": 2,
            "page_end": 2,
            "metadata": {},
        },
    ]
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


class FakeEncoder:
    def encode_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeCollection:
    metadata = {"embedding_model": "fake-model"}

    def count(self) -> int:
        return 1

    def query(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "ids": [["chroma_id"]],
            "documents": [["Packing pressure reduces shrinkage." * 20]],
            "metadatas": [[{
                "chunk_id": "shrinkage",
                "paper_id": "paper_2",
                "title": "Shrinkage study",
                "section_name": "Conclusion",
                "chunk_type": "text",
            }]],
            "distances": [[0.25]],
        }


def test_bm25_retrieval_expands_chinese_domain_terms(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path)
    results = BM25Retriever(chunks_path).search("翘曲和哪些工艺参数有关？", top_k=1)

    assert results[0]["chunk_id"] == "warpage"
    assert results[0]["source"] == "bm25"
    assert len(results[0]["text_preview"]) <= 200


def test_dense_retrieval_returns_unified_structure() -> None:
    retriever = DenseRetriever(collection=FakeCollection(), encoder=FakeEncoder())
    result = retriever.search("保压压力对缩水有什么影响？", top_k=1)[0]

    assert result["chunk_id"] == "shrinkage"
    assert result["source"] == "dense"
    assert result["score"] == 0.8
    assert len(result["text_preview"]) <= 200


def test_hybrid_merge_deduplicates_by_chunk_id() -> None:
    base = {
        "chunk_id": "shared",
        "paper_id": "paper_1",
        "title": "Shared result",
        "section_name": "Results",
        "chunk_type": "text",
        "text_preview": "Evidence",
        "metadata": {},
    }
    results = merge_results(
        [{**base, "score": 1.0, "source": "bm25"}],
        [{**base, "score": 0.5, "source": "dense"}],
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    assert len(results) == 1
    assert results[0]["source"] == "bm25+dense"
    assert results[0]["score"] == 0.7


def test_text_preview_removes_invisible_pdf_characters() -> None:
    assert text_preview("injection\u00admolding") == "injectionmolding"


def test_retrieval_state_requires_nonempty_named_collection(tmp_path: Path) -> None:
    import chromadb

    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path)
    persist_dir = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.create_collection("injection_molding_chunks")
    collection.add(ids=["one"], embeddings=[[0.1, 0.2]], documents=["evidence"])

    state = inspect_retrieval_state(chunks_path, persist_dir, "injection_molding_chunks")

    assert state["chunks_count"] == 2
    assert state["collection_count"] == 1
    assert state["collection_name"] == "injection_molding_chunks"
    assert state["vector_store_ready"] is True
