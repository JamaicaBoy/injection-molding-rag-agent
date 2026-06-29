from __future__ import annotations

import json
from pathlib import Path

from src.app.chat_session_store import (
    append_message,
    create_chat,
    delete_chat,
    list_chats,
    load_chat,
    save_chat,
    update_summary,
    update_title,
)


def test_create_append_save_and_load_chat(tmp_path: Path) -> None:
    chat = create_chat(
        mode="普通 RAG",
        corpus_mode="full",
        conversation_id="chat-one",
        storage_dir=tmp_path,
    )
    question = "保压压力对飞边有什么影响以及应该如何理解"
    append_message("chat-one", "user", question, storage_dir=tmp_path)
    append_message(
        "chat-one",
        "assistant",
        "保压压力会影响飞边风险。[E1]",
        evidence=[
            {
                "evidence_id": "E1",
                "paper_id": "paper_1",
                "chunk_id": "chunk_1",
                "title": "Packing pressure study",
                "section_name": "Results",
                "score": 0.8,
                "text_preview": "Evidence preview",
                "raw_text": "MUST_NOT_BE_SAVED",
                "full_text": "MUST_NOT_BE_SAVED",
            }
        ],
        confidence="high",
        query_rewrite={"intent": "parameter_effect"},
        storage_dir=tmp_path,
    )

    loaded = load_chat(chat["conversation_id"], storage_dir=tmp_path)

    assert loaded is not None
    assert loaded["title"] == question[:20]
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant"]
    stored_text = (tmp_path / "chat-one.json").read_text(encoding="utf-8")
    assert "MUST_NOT_BE_SAVED" not in stored_text
    assert loaded["messages"][1]["evidence"][0]["paper_id"] == "paper_1"


def test_history_is_sorted_by_updated_at(tmp_path: Path) -> None:
    older = create_chat(conversation_id="older", storage_dir=tmp_path)
    newer = create_chat(conversation_id="newer", storage_dir=tmp_path)
    older["updated_at"] = "2026-06-29T08:00:00+00:00"
    newer["updated_at"] = "2026-06-29T10:00:00+00:00"
    save_chat(older, storage_dir=tmp_path)
    save_chat(newer, storage_dir=tmp_path)

    assert [item["conversation_id"] for item in list_chats(storage_dir=tmp_path)] == [
        "newer",
        "older",
    ]
    index = json.loads((tmp_path / "chat_index.json").read_text(encoding="utf-8"))
    assert len(index["chats"]) == 2


def test_update_fields_and_delete_chat(tmp_path: Path) -> None:
    create_chat(conversation_id="editable", storage_dir=tmp_path)
    update_title("editable", "新的标题", storage_dir=tmp_path)
    update_summary(
        "editable",
        {"user_goal": "了解翘曲原因", "cited_papers": ["paper_2"]},
        storage_dir=tmp_path,
    )

    loaded = load_chat("editable", storage_dir=tmp_path)
    assert loaded is not None
    assert loaded["title"] == "新的标题"
    assert loaded["summary"]["cited_papers"] == ["paper_2"]
    assert delete_chat("editable", storage_dir=tmp_path) is True
    assert load_chat("editable", storage_dir=tmp_path) is None

