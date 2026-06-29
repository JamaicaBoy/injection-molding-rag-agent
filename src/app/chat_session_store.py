from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAT_DIR = PROJECT_ROOT / "data" / "runtime" / "chat_sessions"
INDEX_FILE_NAME = "chat_index.json"
DEFAULT_TITLE = "新聊天"
MAX_MESSAGE_CHARS = 12_000
MAX_PREVIEW_CHARS = 200
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chat_path(conversation_id: str, storage_dir: Path) -> Path:
    if not _ID_PATTERN.fullmatch(conversation_id):
        raise ValueError("Invalid conversation_id")
    return storage_dir / f"{conversation_id}.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _compact_evidence(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in (items or [])[:10]:
        source = item.get("source_location") or {}
        score = item.get("rerank_score")
        if score is None:
            score = item.get("relevance_score", item.get("score", 0.0))
        compact.append(
            {
                "evidence_id": str(
                    item.get("evidence_id") or (item.get("metadata") or {}).get("evidence_id") or ""
                ),
                "paper_id": str(item.get("paper_id", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "title": str(item.get("title", ""))[:300],
                "section": str(item.get("section_name") or source.get("section") or "")[:100],
                "score": round(float(score or 0.0), 6),
                "text_preview": str(
                    item.get("text_preview") or item.get("matched_text") or ""
                )[:MAX_PREVIEW_CHARS],
            }
        )
    return compact


def _compact_trace(value: dict[str, Any] | None) -> dict[str, Any]:
    value = value or {}
    return {
        "workflow_backend": str(value.get("workflow_backend", "")),
        "retrieved_count": int(value.get("retrieved_count", 0) or 0),
        "reranked_count": int(value.get("reranked_count", 0) or 0),
        "top_score": round(float(value.get("top_score", 0.0) or 0.0), 6),
        "confidence": value.get("confidence", ""),
        "confidence_reason": str(value.get("confidence_reason", ""))[:500],
        "human_review_reason": str(value.get("human_review_reason", ""))[:200],
        "executed_nodes": [str(item) for item in value.get("executed_nodes", [])[:20]],
    }


def _index_entry(chat: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_id": chat["conversation_id"],
        "title": chat.get("title", DEFAULT_TITLE),
        "created_at": chat.get("created_at", ""),
        "updated_at": chat.get("updated_at", ""),
        "mode": chat.get("mode", "普通 RAG"),
        "corpus_mode": chat.get("corpus_mode", "full"),
        "message_count": len(chat.get("messages", [])),
    }


def _update_index(chat: dict[str, Any], storage_dir: Path) -> None:
    index_path = storage_dir / INDEX_FILE_NAME
    payload = _read_json(index_path, {"chats": []})
    entries = [
        item
        for item in payload.get("chats", [])
        if item.get("conversation_id") != chat["conversation_id"]
    ]
    entries.append(_index_entry(chat))
    entries.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    _write_json(index_path, {"chats": entries})


def create_chat(
    mode: str = "普通 RAG",
    corpus_mode: str = "full",
    *,
    conversation_id: str | None = None,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    storage_dir = Path(storage_dir)
    now = _now()
    chat = {
        "conversation_id": conversation_id or uuid.uuid4().hex,
        "title": DEFAULT_TITLE,
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "summary": None,
        "mode": mode,
        "corpus_mode": corpus_mode,
    }
    with _LOCK:
        save_chat(chat, storage_dir=storage_dir)
    return chat


def list_chats(*, storage_dir: Path = DEFAULT_CHAT_DIR) -> list[dict[str, Any]]:
    storage_dir = Path(storage_dir)
    with _LOCK:
        payload = _read_json(storage_dir / INDEX_FILE_NAME, {"chats": []})
        entries = [dict(item) for item in payload.get("chats", [])]
    return sorted(entries, key=lambda item: str(item.get("updated_at", "")), reverse=True)


def load_chat(
    conversation_id: str,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any] | None:
    with _LOCK:
        payload = _read_json(_chat_path(conversation_id, Path(storage_dir)), None)
    return dict(payload) if isinstance(payload, dict) else None


def save_chat(
    chat: dict[str, Any],
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    storage_dir = Path(storage_dir)
    payload = {
        "conversation_id": str(chat["conversation_id"]),
        "title": str(chat.get("title") or DEFAULT_TITLE)[:100],
        "created_at": str(chat.get("created_at") or _now()),
        "updated_at": str(chat.get("updated_at") or _now()),
        "messages": list(chat.get("messages", [])),
        "summary": chat.get("summary"),
        "mode": str(chat.get("mode") or "普通 RAG"),
        "corpus_mode": str(chat.get("corpus_mode") or "full"),
    }
    with _LOCK:
        _write_json(_chat_path(payload["conversation_id"], storage_dir), payload)
        _update_index(payload, storage_dir)
    return payload


def delete_chat(
    conversation_id: str,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> bool:
    storage_dir = Path(storage_dir)
    path = _chat_path(conversation_id, storage_dir)
    with _LOCK:
        existed = path.exists()
        if existed:
            path.unlink()
        payload = _read_json(storage_dir / INDEX_FILE_NAME, {"chats": []})
        payload["chats"] = [
            item for item in payload.get("chats", []) if item.get("conversation_id") != conversation_id
        ]
        _write_json(storage_dir / INDEX_FILE_NAME, payload)
    return existed


def append_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    evidence: list[dict[str, Any]] | None = None,
    confidence: str | float | None = None,
    need_human_review: bool = False,
    query_rewrite: dict[str, Any] | None = None,
    agent_trace_summary: dict[str, Any] | None = None,
    context_debug: dict[str, Any] | None = None,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    chat = load_chat(conversation_id, storage_dir=storage_dir)
    if chat is None:
        raise FileNotFoundError(f"Chat not found: {conversation_id}")
    message: dict[str, Any] = {
        "message_id": uuid.uuid4().hex,
        "role": role,
        "content": str(content)[:MAX_MESSAGE_CHARS],
        "created_at": _now(),
    }
    if role == "assistant":
        message.update(
            {
                "evidence": _compact_evidence(evidence),
                "confidence": confidence if confidence is not None else "low",
                "need_human_review": bool(need_human_review),
                "query_rewrite": dict(query_rewrite or {}),
                "agent_trace_summary": _compact_trace(agent_trace_summary),
                "context_debug": dict(context_debug or {}),
            }
        )
    chat["messages"].append(message)
    if role == "user" and not any(
        item.get("role") == "user" for item in chat["messages"][:-1]
    ):
        chat["title"] = " ".join(str(content).split())[:20] or DEFAULT_TITLE
    chat["updated_at"] = message["created_at"]
    save_chat(chat, storage_dir=storage_dir)
    return message


def update_title(
    conversation_id: str,
    title: str,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    chat = load_chat(conversation_id, storage_dir=storage_dir)
    if chat is None:
        raise FileNotFoundError(f"Chat not found: {conversation_id}")
    chat["title"] = " ".join(str(title).split())[:100] or DEFAULT_TITLE
    chat["updated_at"] = _now()
    return save_chat(chat, storage_dir=storage_dir)


def update_summary(
    conversation_id: str,
    summary: dict[str, Any] | str | None,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    chat = load_chat(conversation_id, storage_dir=storage_dir)
    if chat is None:
        raise FileNotFoundError(f"Chat not found: {conversation_id}")
    chat["summary"] = summary
    chat["updated_at"] = _now()
    return save_chat(chat, storage_dir=storage_dir)
