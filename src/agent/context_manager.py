from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable

from src.rag.prompts import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 3200

MIN_TOP_N_EVIDENCE = 5
MAX_TOP_N_EVIDENCE = 8
DEFAULT_TOP_N_EVIDENCE = 6

# Evidence text is compressed harder than the raw 600-char text_preview used
# elsewhere in the pipeline, since the LLM context only needs enough text to
# ground a citation, not the full passage.
EVIDENCE_TEXT_CHAR_CAP = 320

MIN_RECENT_TURNS = 2
MAX_RECENT_TURNS = 4
DEFAULT_RECENT_TURNS = 3

MIN_SUMMARY_CHARS = 40

ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+")

RISK_RULES: tuple[str, ...] = (
    "只能基于给定 evidence 回答，不得使用未提供的外部事实。",
    "不得编造论文名、参数范围、具体数值或实验结论。",
    "如果证据不足，必须明确写出“当前论文库证据不足”。",
    "工艺参数建议只能作为候选方向，不能作为直接生产指令；实际生产设置需要工程师结合材料、设备、模具和验证试验确认。",
    "每条关键结论后必须附 evidence 编号，格式为 [E1]、[E2]。",
    "不要引用不存在的 evidence 编号，不要输出参考文献中未出现的论文名。",
)

HIGH_RISK_RULE = (
    "当前问题风险等级为 high：必须在答案中明确提示该内容仅为候选方向、需要人工/工程师复核，"
    "不得给出可直接执行的生产参数。"
)


def estimate_tokens(text: str) -> int:
    """Rough, dependency-free token estimate for mixed CJK/English text.

    Mirrors the heuristic already used in `src.index.build_chunks.estimate_tokens`
    so token-budget accounting stays consistent across the codebase: mostly-English
    text is estimated by word count, mostly-CJK text by character count.
    """
    if not text:
        return 0
    english_words = ENGLISH_WORD_PATTERN.findall(text)
    if english_words and len(" ".join(english_words)) / max(len(text), 1) > 0.45:
        return max(1, int(len(english_words) * 1.15))
    return max(1, math.ceil(len(text) / 1.7))


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected a dictionary-like value, got {type(value).__name__}")


def _turn_to_dict(turn: Any) -> dict[str, Any]:
    if isinstance(turn, dict):
        return dict(turn)
    if hasattr(turn, "to_dict"):
        return turn.to_dict()
    raise TypeError(f"Expected a conversation turn dict or object with to_dict(), got {type(turn).__name__}")


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _summary_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if hasattr(value, "to_dict"):
        return json.dumps(value.to_dict(), ensure_ascii=False)
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _compress_evidence_text(item: dict[str, Any], max_chars: int) -> str:
    text = str(
        item.get("evidence_text")
        or item.get("text_preview")
        or item.get("matched_text")
        or ""
    )
    return _truncate(_collapse_whitespace(text), max_chars)


def _evidence_entry(item: dict[str, Any], max_chars: int) -> dict[str, Any]:
    return {
        "evidence_id": str(item.get("evidence_id") or ""),
        "paper_id": str(item.get("paper_id") or ""),
        "title": str(item.get("title") or ""),
        "section_name": str(item.get("section_name") or ""),
        "chunk_type": str(item.get("chunk_type") or ""),
        "text": _compress_evidence_text(item, max_chars),
    }


def _turn_summary_line(turn: dict[str, Any]) -> str:
    question = str(turn.get("user_question") or "").strip()
    answer = str(turn.get("system_answer_brief") or "").strip()
    papers = [str(p) for p in (turn.get("cited_paper_ids") or []) if p]
    papers_part = f"（引用：{', '.join(papers)}）" if papers else ""
    index = turn.get("turn_index", "")
    line = f"Q{index}: {question}"
    if answer:
        line += f"\nA{index}: {answer}{papers_part}"
    return line


@dataclass(frozen=True)
class ManagedContext:
    """Result of `build_llm_context`: the structured context plus an audit trail
    of what was kept/dropped/compressed to fit the token budget."""

    llm_context: dict[str, Any]
    context_debug: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"llm_context": self.llm_context, "context_debug": self.context_debug}


def build_llm_context(
    current_query: str,
    query_info: Any,
    conversation_history: Iterable[Any] | None,
    conversation_summary: Any,
    reranked_evidence: list[dict[str, Any]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    top_n_evidence: int = DEFAULT_TOP_N_EVIDENCE,
    recent_turns_window: int = DEFAULT_RECENT_TURNS,
    *,
    conversation_id: str | None = None,
    recent_turns: Iterable[Any] | None = None,
) -> ManagedContext:
    """Assemble the bounded context handed to the LLM, in priority order:

    current_query > top evidence > necessary conversation_summary
    > most recent 2-4 turns > older history (dropped, never sent).

    Evidence is always capped at `top_n_evidence` (5-8) up front and is the
    last thing trimmed if the assembled context still exceeds `token_budget`;
    conversation history (recent turns, then the summary) is compressed or
    dropped first.
    """
    if not current_query or not current_query.strip():
        raise ValueError("current_query must not be empty.")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive.")

    top_n_evidence = max(MIN_TOP_N_EVIDENCE, min(int(top_n_evidence), MAX_TOP_N_EVIDENCE))
    recent_turns_window = max(MIN_RECENT_TURNS, min(int(recent_turns_window), MAX_RECENT_TURNS))

    query_info_dict = _as_dict(query_info)
    current_query_text = current_query.strip()

    history_source = recent_turns if recent_turns is not None else conversation_history
    all_turns = [_turn_to_dict(turn) for turn in (history_source or [])]
    # Conversation history is assumed chronological (oldest first), matching
    # ConversationState.turns / recent_turns().
    recent_candidates = all_turns[-recent_turns_window:]
    older_turns = all_turns[: len(all_turns) - len(recent_candidates)]
    older_turn_indices = [int(t.get("turn_index", -1)) for t in older_turns]

    summary_text = _collapse_whitespace(_summary_to_text(conversation_summary))
    summary_received = bool(summary_text)

    # --- Evidence: select top_n only, compress text, never touched after this. ---
    evidence_table = [_evidence_entry(item, EVIDENCE_TEXT_CHAR_CAP) for item in reranked_evidence[:top_n_evidence]]
    dropped_evidence_ids = [
        str(item.get("evidence_id") or "") for item in reranked_evidence[top_n_evidence:]
    ]

    # --- Risk rules: static rule set, plus a high-risk note when applicable. ---
    risk_level = str(query_info_dict.get("risk_level", "")).lower()
    risk_rules = list(RISK_RULES)
    if risk_level in {"high", "critical"}:
        risk_rules.append(HIGH_RISK_RULE)

    # --- Token accounting ---
    fixed_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(current_query_text)
    fixed_tokens += sum(estimate_tokens(rule) for rule in risk_rules)
    evidence_tokens = sum(
        estimate_tokens(entry["text"]) + estimate_tokens(entry["title"]) + 6 for entry in evidence_table
    )
    summary_tokens = estimate_tokens(summary_text)
    recent_token_list = [estimate_tokens(_turn_summary_line(turn)) for turn in recent_candidates]

    used_tokens = fixed_tokens + evidence_tokens + summary_tokens + sum(recent_token_list)

    dropped_recent_turn_indices: list[int] = []
    summary_compressed = False
    summary_dropped = False

    # Priority: never touch evidence or current_query/system_instruction/risk_rules.
    # Step 1 — drop recent turns oldest-of-the-kept-window first (summary outranks
    # recent turns per the required priority order, so turns go before summary).
    while used_tokens > token_budget and recent_candidates:
        dropped_turn = recent_candidates.pop(0)
        dropped_tokens = recent_token_list.pop(0)
        dropped_recent_turn_indices.append(int(dropped_turn.get("turn_index", -1)))
        used_tokens -= dropped_tokens

    # Step 2 — compress, then if necessary drop, the conversation summary.
    if used_tokens > token_budget and summary_text:
        while used_tokens > token_budget and len(summary_text) > MIN_SUMMARY_CHARS:
            shrunk_len = max(MIN_SUMMARY_CHARS, int(len(summary_text) * 0.7))
            if shrunk_len >= len(summary_text):
                break
            summary_text = _truncate(summary_text, shrunk_len)
            summary_compressed = True
            new_summary_tokens = estimate_tokens(summary_text)
            used_tokens += new_summary_tokens - summary_tokens
            summary_tokens = new_summary_tokens
        if used_tokens > token_budget:
            used_tokens -= summary_tokens
            summary_tokens = 0
            summary_text = ""
            summary_dropped = True

    llm_context = {
        "system_instruction": SYSTEM_PROMPT,
        "conversation_summary": summary_text,
        "recent_turns": list(recent_candidates),
        "current_query": current_query_text,
        "evidence_table": evidence_table,
        "risk_rules": risk_rules,
    }

    context_debug = {
        "conversation_id": str(conversation_id or ""),
        "history_source": "recent_turns" if recent_turns is not None else "conversation_history",
        "history_turns_received": len(all_turns),
        "conversation_summary_received": summary_received,
        "conversation_summary_used": bool(summary_text),
        "token_budget": token_budget,
        "estimated_tokens_used": used_tokens,
        "over_budget": used_tokens > token_budget,
        "evidence_requested_top_n": top_n_evidence,
        "evidence_kept_ids": [entry["evidence_id"] for entry in evidence_table],
        "evidence_dropped_ids": dropped_evidence_ids,
        "recent_turns_kept_indices": [int(t.get("turn_index", -1)) for t in recent_candidates],
        "recent_turns_dropped_indices": dropped_recent_turn_indices,
        "older_turns_excluded_indices": older_turn_indices,
        "conversation_summary_compressed": summary_compressed,
        "conversation_summary_dropped": summary_dropped,
    }

    return ManagedContext(llm_context=llm_context, context_debug=context_debug)


def _format_evidence_table(evidence_table: list[dict[str, Any]]) -> str:
    if not evidence_table:
        return "(no evidence)"
    blocks: list[str] = []
    for entry in evidence_table:
        blocks.append(
            "\n".join(
                (
                    f"[{entry.get('evidence_id', '')}]",
                    f"title: {entry.get('title', '')}",
                    f"paper_id: {entry.get('paper_id', '')}",
                    f"section: {entry.get('section_name', '')}",
                    f"chunk_type: {entry.get('chunk_type', '')}",
                    f"content: {entry.get('text', '')}",
                )
            )
        )
    return "\n\n".join(blocks)


def render_prompt(llm_context: dict[str, Any], query_info: Any | None = None) -> str:
    """Render the structured `llm_context` into the final user-prompt text sent
    to the LLM (the system instruction is sent separately as `system_instruction`)."""
    sections: list[str] = []

    summary = llm_context.get("conversation_summary")
    if summary:
        sections.append(f"历史对话摘要：\n{summary}")

    recent_turns = llm_context.get("recent_turns") or []
    if recent_turns:
        lines = [_turn_summary_line(turn) for turn in recent_turns]
        sections.append("最近对话：\n" + "\n".join(lines))

    sections.append(f"用户当前问题：\n{llm_context.get('current_query', '')}")

    if query_info is not None:
        sections.append(f"结构化查询：\n{json.dumps(_as_dict(query_info), ensure_ascii=False, indent=2)}")

    sections.append(f"Evidence：\n{_format_evidence_table(llm_context.get('evidence_table', []))}")

    risk_rules = llm_context.get("risk_rules") or []
    if risk_rules:
        numbered = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(risk_rules))
        sections.append(f"必须遵守以下规则：\n{numbered}")

    sections.append(
        "请直接给出中文答案。先回答核心问题，再说明证据限制。每条关键结论后使用 [E编号] 引用。"
    )

    return "\n\n".join(sections)
