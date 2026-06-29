from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb

from src.index.build_vector_index import (
    build_index,
    collection_names,
    ensure_persist_dir,
    runtime_persist_dir,
)
from src.index.index_lock import DEFAULT_LOCK_DIR, index_write_lock
from src.index.index_registry import DEFAULT_REGISTRY_PATH, remove_index_record


def upload_collection_name(session_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(session_id)).strip("_-")
    if not safe_id:
        raise ValueError("session_id must contain letters or numbers")
    return f"injection_user_uploads_{safe_id[:50]}"


def _client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir = ensure_persist_dir(Path(persist_dir))
    return chromadb.PersistentClient(path=str(runtime_persist_dir(persist_dir)))


def resolve_base_embedding_model(
    persist_dir: Path,
    base_collection_name: str,
) -> str | None:
    client = _client(persist_dir)
    if base_collection_name not in collection_names(client):
        return None
    metadata = client.get_collection(base_collection_name).metadata or {}
    return str(
        metadata.get("embedding_local_path")
        or metadata.get("embedding_model")
        or ""
    ) or None


def add_uploaded_chunks(
    *,
    chunks_path: Path,
    persist_dir: Path,
    session_id: str,
    base_collection_name: str,
    batch_size: int = 4,
    model_name: str | None = None,
    embedding_model: Any | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    lock_dir: Path = DEFAULT_LOCK_DIR,
) -> dict[str, Any]:
    collection_name = upload_collection_name(session_id)
    effective_model = model_name or resolve_base_embedding_model(
        persist_dir, base_collection_name
    )
    if embedding_model is None and not effective_model:
        raise RuntimeError(
            "The base collection does not record an embedding model; "
            "provide a local sentence-transformers model path."
        )
    stats = build_index(
        chunks_path=Path(chunks_path),
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        model_name=effective_model,
        reset=False,
        batch_size=batch_size,
        backend="sentence-transformers",
        embedding_model=embedding_model,
        resume=True,
        corpus_mode=f"upload:{session_id}",
        registry_path=registry_path,
        lock_dir=lock_dir,
    )
    stats["base_collection_name"] = base_collection_name
    stats["upload_collection_name"] = collection_name
    return stats


def upload_collection_stats(
    *,
    persist_dir: Path,
    session_id: str,
) -> dict[str, Any]:
    collection_name = upload_collection_name(session_id)
    runtime_dir = runtime_persist_dir(Path(persist_dir))
    if not runtime_dir.exists():
        return {
            "collection_name": collection_name,
            "exists": False,
            "vector_count": 0,
        }
    client = _client(persist_dir)
    exists = collection_name in collection_names(client)
    return {
        "collection_name": collection_name,
        "exists": exists,
        "vector_count": client.get_collection(collection_name).count() if exists else 0,
    }


def clear_upload_collection(
    *,
    persist_dir: Path,
    session_id: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    lock_dir: Path = DEFAULT_LOCK_DIR,
) -> bool:
    collection_name = upload_collection_name(session_id)
    with index_write_lock(
        Path(persist_dir), collection_name, lock_dir=Path(lock_dir), timeout=0.0
    ):
        client = _client(persist_dir)
        if collection_name not in collection_names(client):
            return False
        client.delete_collection(collection_name)
        remove_index_record(
            corpus_mode=f"upload:{session_id}",
            collection_name=collection_name,
            registry_path=registry_path,
        )
    return True
