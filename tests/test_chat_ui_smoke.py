from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.context_manager import build_llm_context
from src.app.chat_session_store import create_chat, load_chat
from src.app.chat_ui import ensure_active_chat, process_chat_turn


def fake_execute(*args: Any, **kwargs: Any) -> dict[str, Any]:
    mode, question, top_k, corpus_mode, workflow_backend = args
    fake_execute.calls.append({"args": args, "kwargs": kwargs})
    return {
        "answer": f"回答：{question} [E1]",
        "evidence_list": [
            {
                "evidence_id": "E1",
                "paper_id": "paper_1",
                "chunk_id": "chunk_1",
                "title": "Packing pressure study",
                "section_name": "Results",
                "score": 0.81,
                "text_preview": "Packing pressure affects flash.",
                "full_text": "DO_NOT_STORE_FULL_TEXT",
            }
        ],
        "confidence": "medium",
        "need_human_review": False,
        "query_rewrite": {
            "intent": "parameter_effect",
            "parameters": ["packing_pressure"],
        },
        "mode": mode,
        "corpus_mode": corpus_mode,
        "workflow_backend": workflow_backend,
        "debug": {
            "trace_summary": {
                "workflow_backend": workflow_backend,
                "retrieved_count": top_k,
                "reranked_count": top_k,
                "top_score": 0.81,
                "executed_nodes": ["retrieve_node", "answer_node"],
            }
        },
        "context_debug": {
            "conversation_id": kwargs["conversation_id"],
            "history_turns_received": len(kwargs["recent_turns"]),
        },
    }


fake_execute.calls = []


def test_chat_turn_adapter_passes_bounded_conversation_context(tmp_path: Path) -> None:
    fake_execute.calls.clear()
    chat = create_chat(conversation_id="smoke-chat", storage_dir=tmp_path)
    _, chat = process_chat_turn(
        fake_execute,
        chat=chat,
        question="保压压力对飞边有什么影响？",
        top_k=5,
        corpus_mode="full",
        mode="普通 RAG",
        workflow_backend="langgraph",
        storage_dir=tmp_path,
    )
    _, chat = process_chat_turn(
        fake_execute,
        chat=chat,
        question="那对缩水呢？",
        top_k=5,
        corpus_mode="full",
        mode="普通 RAG",
        workflow_backend="langgraph",
        storage_dir=tmp_path,
    )

    second_call = fake_execute.calls[1]["kwargs"]
    assert second_call["conversation_id"] == "smoke-chat"
    assert len(second_call["recent_turns"]) == 1
    assert second_call["recent_turns"][0]["user_question"] == "保压压力对飞边有什么影响？"
    assert "conversation_summary" in second_call
    loaded = load_chat("smoke-chat", storage_dir=tmp_path)
    assert loaded is not None
    assert len(loaded["messages"]) == 4
    assert "DO_NOT_STORE_FULL_TEXT" not in str(loaded)
    assert loaded["messages"][-1]["agent_trace_summary"]["retrieved_count"] == 5


def test_ensure_active_chat_and_context_debug(tmp_path: Path) -> None:
    session_state: dict[str, Any] = {}
    chat = ensure_active_chat(
        session_state,
        default_mode="普通 RAG",
        default_corpus_mode="dev",
        storage_dir=tmp_path,
    )
    managed = build_llm_context(
        current_query="那对缩水呢？",
        query_info={"risk_level": "low"},
        conversation_history=None,
        conversation_summary={"user_goal": "了解保压压力"},
        reranked_evidence=[],
        conversation_id=chat["conversation_id"],
        recent_turns=[
            {
                "turn_index": 1,
                "user_question": "保压压力对飞边有什么影响？",
                "system_answer_brief": "存在影响。",
                "cited_paper_ids": ["paper_1"],
            }
        ],
    )

    assert session_state["active_chat_id"] == chat["conversation_id"]
    assert managed.context_debug["conversation_id"] == chat["conversation_id"]
    assert managed.context_debug["history_source"] == "recent_turns"
    assert managed.context_debug["history_turns_received"] == 1
    assert managed.context_debug["conversation_summary_received"] is True
