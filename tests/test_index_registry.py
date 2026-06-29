from __future__ import annotations

import json
from pathlib import Path

from src.index.index_registry import (
    get_index_record,
    infer_corpus_mode,
    load_index_registry,
    register_index,
    remove_index_record,
)


def test_register_and_read_index_metadata(tmp_path: Path) -> None:
    registry_path = tmp_path / "index_registry.json"
    record = register_index(
        corpus_mode="full",
        collection_name="injection_papers_full",
        chunks_path=Path("data/chunks/full_chunks.jsonl"),
        persist_dir=Path("vector_store/chroma_full"),
        paper_count=819,
        chunk_count=43343,
        embedding_model="E:/AI_Models/BAAI/bge-m3",
        built_at="2026-06-29T10:00:00+00:00",
        version="idx-full-v1",
        registry_path=registry_path,
    )

    loaded = get_index_record(
        corpus_mode="full",
        collection_name="injection_papers_full",
        registry_path=registry_path,
    )
    assert loaded == record
    assert loaded["paper_count"] == 819
    assert loaded["chunk_count"] == 43343
    assert loaded["embedding_model"].endswith("bge-m3")
    assert json.loads(registry_path.read_text(encoding="utf-8"))["indexes"]


def test_registry_keeps_modes_separate_and_updates_version(tmp_path: Path) -> None:
    registry_path = tmp_path / "index_registry.json"
    common = {
        "chunks_path": Path("chunks.jsonl"),
        "persist_dir": Path("chroma"),
        "paper_count": 3,
        "chunk_count": 10,
        "embedding_model": "bge-m3",
        "registry_path": registry_path,
    }
    register_index(
        corpus_mode="dev",
        collection_name="shared_collection",
        version="dev-v1",
        **common,
    )
    register_index(
        corpus_mode="full",
        collection_name="shared_collection",
        version="full-v1",
        **common,
    )
    register_index(
        corpus_mode="dev",
        collection_name="shared_collection",
        version="dev-v2",
        **common,
    )

    payload = load_index_registry(registry_path)
    assert len(payload["indexes"]) == 2
    assert get_index_record(
        corpus_mode="dev",
        collection_name="shared_collection",
        registry_path=registry_path,
    )["version"] == "dev-v2"
    assert get_index_record(
        corpus_mode="full",
        collection_name="shared_collection",
        registry_path=registry_path,
    )["version"] == "full-v1"


def test_remove_and_infer_modes(tmp_path: Path) -> None:
    registry_path = tmp_path / "index_registry.json"
    register_index(
        corpus_mode="upload:abc",
        collection_name="injection_user_uploads_abc",
        chunks_path=Path("upload_chunks.jsonl"),
        persist_dir=Path("chroma"),
        paper_count=1,
        chunk_count=5,
        embedding_model="bge-m3",
        registry_path=registry_path,
    )
    assert infer_corpus_mode("injection_papers_full") == "full"
    assert infer_corpus_mode("injection_user_uploads_abc") == "upload"
    assert remove_index_record(
        corpus_mode="upload:abc",
        collection_name="injection_user_uploads_abc",
        registry_path=registry_path,
    ) is True
    assert get_index_record(
        corpus_mode="upload:abc",
        collection_name="injection_user_uploads_abc",
        registry_path=registry_path,
    ) is None
