from __future__ import annotations

from pathlib import Path

import pytest

from src.index.build_vector_index import build_index
from src.index.index_lock import IndexLockError, index_write_lock


def test_same_collection_cannot_be_locked_twice(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    lock_dir = tmp_path / "locks"

    with index_write_lock(persist_dir, "collection_a", lock_dir=lock_dir):
        with pytest.raises(IndexLockError, match="索引正在更新，请稍后"):
            with index_write_lock(persist_dir, "collection_a", lock_dir=lock_dir):
                pass


def test_different_collections_can_be_locked_independently(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    lock_dir = tmp_path / "locks"

    with index_write_lock(persist_dir, "collection_a", lock_dir=lock_dir):
        with index_write_lock(persist_dir, "collection_b", lock_dir=lock_dir) as second:
            assert second.acquired is True


def test_lock_is_released_after_exception(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    lock_dir = tmp_path / "locks"

    with pytest.raises(RuntimeError):
        with index_write_lock(persist_dir, "collection_a", lock_dir=lock_dir):
            raise RuntimeError("build failed")

    with index_write_lock(persist_dir, "collection_a", lock_dir=lock_dir) as acquired:
        assert acquired.acquired is True


def test_build_index_checks_collection_lock_before_writing(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    lock_dir = tmp_path / "locks"

    with index_write_lock(persist_dir, "collection_a", lock_dir=lock_dir):
        with pytest.raises(IndexLockError):
            build_index(
                chunks_path=tmp_path / "missing_chunks.jsonl",
                persist_dir=persist_dir,
                collection_name="collection_a",
                lock_dir=lock_dir,
                registry_path=tmp_path / "index_registry.json",
            )

