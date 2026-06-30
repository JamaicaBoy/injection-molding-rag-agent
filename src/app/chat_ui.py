from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, MutableMapping

import streamlit as st

from src.agent.conversation_state import ConversationState
from src.app.chat_session_store import (
    DEFAULT_CHAT_DIR,
    append_message,
    create_chat,
    delete_chat,
    list_chats,
    load_chat,
    save_chat,
    update_summary,
)


ExecuteFunction = Callable[..., dict[str, Any]]


def ensure_active_chat(
    session_state: MutableMapping[str, Any],
    *,
    default_mode: str,
    default_corpus_mode: str,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    conversation_id = session_state.get("active_chat_id")
    chat = load_chat(str(conversation_id), storage_dir=storage_dir) if conversation_id else None
    if chat is None:
        chat = create_chat(
            mode=default_mode,
            corpus_mode=default_corpus_mode,
            storage_dir=storage_dir,
        )
        session_state["active_chat_id"] = chat["conversation_id"]
    return chat


def start_new_chat(
    session_state: MutableMapping[str, Any],
    *,
    mode: str,
    corpus_mode: str,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    chat = create_chat(mode=mode, corpus_mode=corpus_mode, storage_dir=storage_dir)
    session_state["active_chat_id"] = chat["conversation_id"]
    session_state.pop("last_result", None)
    return chat


def select_chat(
    session_state: MutableMapping[str, Any],
    conversation_id: str,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    chat = load_chat(conversation_id, storage_dir=storage_dir)
    if chat is None:
        raise FileNotFoundError(f"Chat not found: {conversation_id}")
    session_state["active_chat_id"] = conversation_id
    session_state.pop("last_result", None)
    return chat


def delete_chat_from_history(
    session_state: MutableMapping[str, Any],
    conversation_id: str,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> bool:
    """Delete one chat and keep the active-chat pointer valid."""
    deleted = delete_chat(conversation_id, storage_dir=storage_dir)
    if not deleted:
        return False

    if str(session_state.get("active_chat_id", "")) == conversation_id:
        remaining = list_chats(storage_dir=storage_dir)
        if remaining:
            session_state["active_chat_id"] = str(remaining[0]["conversation_id"])
        else:
            session_state.pop("active_chat_id", None)
    session_state.pop("last_result", None)
    session_state.pop("pending_delete_chat_id", None)
    return True


def save_chat_settings(
    chat: dict[str, Any],
    *,
    mode: str,
    corpus_mode: str,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> dict[str, Any]:
    if chat.get("mode") == mode and chat.get("corpus_mode") == corpus_mode:
        return chat
    chat = dict(chat)
    chat["mode"] = mode
    chat["corpus_mode"] = corpus_mode
    return save_chat(chat, storage_dir=storage_dir)


def _trace_summary(result: dict[str, Any]) -> dict[str, Any]:
    debug = result.get("debug") or {}
    existing = debug.get("trace_summary")
    if isinstance(existing, dict):
        return existing
    return {
        "workflow_backend": result.get("workflow_backend", "classic"),
        "retrieved_count": len(result.get("evidence_list", [])),
        "reranked_count": len(result.get("evidence_list", [])),
        "top_score": max(
            (
                float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
                for item in result.get("evidence_list", [])
            ),
            default=0.0,
        ),
        "confidence": result.get("confidence", "low"),
        "confidence_reason": result.get("confidence_reason", ""),
        "llm_mode": result.get("llm_mode", ""),
        "llm_model": result.get("llm_model", ""),
        "llm_fallback_reason": result.get("llm_fallback_reason", ""),
        "human_review_reason": result.get("human_review_reason", ""),
        "executed_nodes": result.get("node_history", []),
    }


def process_chat_turn(
    execute_fn: ExecuteFunction,
    *,
    chat: dict[str, Any],
    question: str,
    top_k: int,
    corpus_mode: str,
    mode: str,
    workflow_backend: str,
    upload_session_id: str | None = None,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist one user/assistant turn around the existing workflow call."""
    conversation = ConversationState.from_chat_session(chat)
    recent_turns = [turn.to_dict() for turn in conversation.recent_turns()]
    conversation_summary = conversation.summary_text()
    conversation_id = str(chat["conversation_id"])

    append_message(
        conversation_id,
        "user",
        question,
        storage_dir=storage_dir,
    )
    result = execute_fn(
        mode,
        question,
        top_k,
        corpus_mode,
        workflow_backend,
        conversation=conversation,
        conversation_id=conversation_id,
        recent_turns=recent_turns,
        conversation_summary=conversation_summary,
        upload_session_id=upload_session_id,
    )
    append_message(
        conversation_id,
        "assistant",
        result.get("answer") or "当前论文库证据不足。",
        evidence=result.get("evidence_list"),
        confidence=result.get("confidence", "low"),
        need_human_review=bool(result.get("need_human_review", False)),
        query_rewrite=result.get("query_rewrite"),
        agent_trace_summary=_trace_summary(result),
        context_debug=result.get("context_debug"),
        storage_dir=storage_dir,
    )
    updated = load_chat(conversation_id, storage_dir=storage_dir)
    if updated is None:
        raise RuntimeError("Chat disappeared after saving the assistant response.")
    refreshed_state = ConversationState.from_chat_session(updated)
    if refreshed_state.summary is not None:
        updated = update_summary(
            conversation_id,
            refreshed_state.summary_dict(),
            storage_dir=storage_dir,
        )
    return result, updated


def render_sidebar_history(
    active_chat_id: str,
    *,
    storage_dir: Path = DEFAULT_CHAT_DIR,
) -> str | None:
    st.subheader("历史记录")
    selected: str | None = None
    chats = list_chats(storage_dir=storage_dir)
    if not chats:
        st.caption("暂无历史聊天")
        return None
    for item in chats[:30]:
        conversation_id = str(item["conversation_id"])
        label = str(item.get("title") or "新聊天")
        title_col, delete_col = st.columns([5, 1], gap="small")
        with title_col:
            if st.button(
                label,
                key=f"chat_history_{conversation_id}",
                width="stretch",
                type="primary" if conversation_id == active_chat_id else "secondary",
            ):
                selected = conversation_id
        with delete_col:
            if st.button(
                "",
                key=f"delete_chat_{conversation_id}",
                icon=":material/delete:",
                help=f"删除聊天：{label}",
                width="stretch",
            ):
                st.session_state["pending_delete_chat_id"] = conversation_id
                st.rerun()
        st.caption(
            f"{item.get('updated_at', '')[:16].replace('T', ' ')} · {item.get('message_count', 0)} 条消息"
        )
        if st.session_state.get("pending_delete_chat_id") == conversation_id:
            st.warning(f"确认删除“{label}”？此操作无法撤销。")
            confirm_col, cancel_col = st.columns(2, gap="small")
            with confirm_col:
                if st.button(
                    "确认删除",
                    key=f"confirm_delete_chat_{conversation_id}",
                    type="primary",
                    width="stretch",
                ):
                    delete_chat_from_history(st.session_state, conversation_id)
                    st.rerun()
            with cancel_col:
                if st.button(
                    "取消",
                    key=f"cancel_delete_chat_{conversation_id}",
                    width="stretch",
                ):
                    st.session_state.pop("pending_delete_chat_id", None)
                    st.rerun()
    return selected


def render_chat_messages(
    chat: dict[str, Any],
    *,
    show_rewrite: bool,
    show_debug: bool,
) -> None:
    for message in chat.get("messages", []):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        with st.chat_message(role):
            st.markdown(str(message.get("content", "")))
            if role != "assistant":
                continue
            with st.expander("回答依据与运行信息", expanded=False):
                trace_summary = message.get("agent_trace_summary") or {}
                if trace_summary.get("llm_mode") == "mock":
                    st.warning(
                        "本地 LLM 生成失败，本条回答使用了 Mock 证据整理。"
                    )
                    if trace_summary.get("llm_fallback_reason"):
                        st.caption(str(trace_summary["llm_fallback_reason"]))
                elif trace_summary.get("llm_model"):
                    st.caption(f"本地生成模型：{trace_summary['llm_model']}")
                confidence_col, review_col = st.columns(2)
                confidence_col.metric("Confidence", str(message.get("confidence", "low")))
                review_col.metric(
                    "Human Review",
                    "需要" if message.get("need_human_review") else "不需要",
                )
                evidence = list(message.get("evidence") or [])
                if evidence:
                    st.caption("引用证据")
                    st.dataframe(evidence, width="stretch", hide_index=True)
                else:
                    st.caption("本条消息没有保存引用证据。")
                if show_rewrite:
                    st.caption("Query Rewrite")
                    st.json(message.get("query_rewrite") or {})
                if show_debug:
                    st.caption("Agent Trace Summary")
                    st.json(message.get("agent_trace_summary") or {})
                    context_debug = message.get("context_debug") or {}
                    if context_debug:
                        st.caption("Context Debug")
                        st.json(context_debug)
