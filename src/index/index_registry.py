from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.index.index_lock import index_write_lock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "runtime" / "index_registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(corpus_mode: str, collection_name: str) -> str:
    return f"{corpus_mode}::{collection_name}"


def load_index_registry(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"updated_at": "", "indexes": {}}
    if not isinstance(payload.get("indexes"), dict):
        payload["indexes"] = {}
    return payload


def infer_corpus_mode(collection_name: str) -> str:
    lowered = collection_name.lower()
    if lowered.startswith("injection_user_uploads_"):
        return "upload"
    for mode in (
        "public_full_artifact",
        "public_sample",
        "selected",
        "full",
        "dev",
        "upload_only",
    ):
        if mode in lowered:
            return mode
    if collection_name == "injection_molding_chunks":
        return "dev"
    return "custom"


def register_index(
    *,
    corpus_mode: str,
    collection_name: str,
    chunks_path: Path,
    persist_dir: Path,
    paper_count: int,
    chunk_count: int,
    embedding_model: str,
    built_at: str | None = None,
    version: str | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    registry_path = Path(registry_path)
    built_at = built_at or _now()
    version = version or (
        f"idx-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    record = {
        "version": version,
        "corpus_mode": corpus_mode,
        "collection_name": collection_name,
        "chunks_path": str(Path(chunks_path)),
        "persist_dir": str(Path(persist_dir)),
        "built_at": built_at,
        "paper_count": int(paper_count),
        "chunk_count": int(chunk_count),
        "embedding_model": str(embedding_model),
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_lock_dir = registry_path.parent / "index_locks"
    with index_write_lock(
        registry_path.parent,
        "__index_registry__",
        lock_dir=registry_lock_dir,
        timeout=10.0,
        stale_after=60.0,
    ):
        payload = load_index_registry(registry_path)
        payload["indexes"][_key(corpus_mode, collection_name)] = record
        payload["updated_at"] = built_at
        temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(registry_path)
    return record


def get_index_record(
    *,
    corpus_mode: str,
    collection_name: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any] | None:
    payload = load_index_registry(registry_path)
    record = payload["indexes"].get(_key(corpus_mode, collection_name))
    if record is None:
        matches = [
            item
            for item in payload["indexes"].values()
            if item.get("collection_name") == collection_name
        ]
        if matches:
            record = max(matches, key=lambda item: str(item.get("built_at", "")))
    return dict(record) if isinstance(record, dict) else None


def remove_index_record(
    *,
    corpus_mode: str,
    collection_name: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> bool:
    registry_path = Path(registry_path)
    registry_lock_dir = registry_path.parent / "index_locks"
    with index_write_lock(
        registry_path.parent,
        "__index_registry__",
        lock_dir=registry_lock_dir,
        timeout=10.0,
        stale_after=60.0,
    ):
        payload = load_index_registry(registry_path)
        removed = payload["indexes"].pop(_key(corpus_mode, collection_name), None) is not None
        if removed:
            payload["updated_at"] = _now()
            temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(registry_path)
    return removed
