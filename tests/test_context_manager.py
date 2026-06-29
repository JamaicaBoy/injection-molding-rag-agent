from __future__ import annotations

import pytest

from src.agent.context_manager import (
    DEFAULT_RECENT_TURNS,
    DEFAULT_TOP_N_EVIDENCE,
    EVIDENCE_TEXT_CHAR_CAP,
    HIGH_RISK_RULE,
    MAX_TOP_N_EVIDENCE,
    MIN_TOP_N_EVIDENCE,
    ManagedContext,
    build_llm_context,
    estimate_tokens,
    render_prompt,
)
from src.rag.prompts import SYSTEM_PROMPT
from src.retrieval.query_rewrite import rewrite_query


def make_evidence(count: int, text_len: int = 50) -> list[dict[str, object]]:
    return [
        {
            "evidence_id": f"E{i}",
            "paper_id": f"paper_{i}",
            "title": f"Paper {i}",
            "section_name": "Results",
            "chunk_type": "knowledge_card",
            "score": 1.0 - i * 0.01,
            "text_preview": ("保压压力对缩水影响显著。" * 10)[:text_len] if text_len else "",
        }
        for i in range(1, count + 1)
    ]


def make_turn(index: int, question: str = "保压压力对翘曲有什么影响？", answer: str = "保压压力升高通常能降低翘曲。") -> dict[str, object]:
    return {
        "turn_index": index,
        "timestamp": f"2026-06-29T10:{index:02d}:00+00:00",
        "user_question": f"{question}（第{index}轮）",
        "system_answer_brief": answer,
        "key_entities": {"defect_type": ["warpage"], "material": [], "parameters": ["packing_pressure"], "quality_metric": []},
        "cited_paper_ids": [f"paper_{index}"],
    }


def test_llm_context_has_required_keys() -> None:
    managed = build_llm_context(
        current_query="保压压力对缩水有什么影响？",
        query_info=rewrite_query("保压压力对缩水有什么影响？"),
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=make_evidence(3),
    )
    context = managed.llm_context
    assert set(context.keys()) == {
        "system_instruction",
        "conversation_summary",
        "recent_turns",
        "current_query",
        "evidence_table",
        "risk_rules",
    }
    assert context["system_instruction"] == SYSTEM_PROMPT
    assert context["current_query"] == "保压压力对缩水有什么影响？"


def test_current_query_required() -> None:
    with pytest.raises(ValueError):
        build_llm_context(
            current_query="   ",
            query_info={},
            conversation_history=None,
            conversation_summary=None,
            reranked_evidence=[],
        )


def test_token_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        build_llm_context(
            current_query="q",
            query_info={},
            conversation_history=None,
            conversation_summary=None,
            reranked_evidence=[],
            token_budget=0,
        )


def test_evidence_capped_at_top_n_default() -> None:
    managed = build_llm_context(
        current_query="保压压力对缩水有什么影响？",
        query_info={},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=make_evidence(12),
    )
    assert len(managed.llm_context["evidence_table"]) == DEFAULT_TOP_N_EVIDENCE
    assert MIN_TOP_N_EVIDENCE <= DEFAULT_TOP_N_EVIDENCE <= MAX_TOP_N_EVIDENCE
    kept_ids = [e["evidence_id"] for e in managed.llm_context["evidence_table"]]
    assert kept_ids == [f"E{i}" for i in range(1, DEFAULT_TOP_N_EVIDENCE + 1)]
    assert managed.context_debug["evidence_kept_ids"] == kept_ids
    assert managed.context_debug["evidence_dropped_ids"] == [
        f"E{i}" for i in range(DEFAULT_TOP_N_EVIDENCE + 1, 13)
    ]


def test_top_n_evidence_is_clamped_to_allowed_range() -> None:
    managed_low = build_llm_context(
        current_query="q",
        query_info={},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=make_evidence(12),
        top_n_evidence=2,
    )
    assert len(managed_low.llm_context["evidence_table"]) == MIN_TOP_N_EVIDENCE

    managed_high = build_llm_context(
        current_query="q",
        query_info={},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=make_evidence(12),
        top_n_evidence=99,
    )
    assert len(managed_high.llm_context["evidence_table"]) == MAX_TOP_N_EVIDENCE


def test_evidence_text_is_compressed_not_full_length() -> None:
    long_text = "保压压力对缩水率有显著影响。" * 50
    evidence = [{"evidence_id": "E1", "paper_id": "p1", "title": "T", "text_preview": long_text}]
    managed = build_llm_context(
        current_query="q",
        query_info={},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=evidence,
    )
    compressed = managed.llm_context["evidence_table"][0]["text"]
    assert len(compressed) <= EVIDENCE_TEXT_CHAR_CAP
    assert compressed.endswith("…")
    assert compressed != long_text


def test_recent_turns_window_excludes_older_history() -> None:
    history = [make_turn(i) for i in range(1, 8)]  # 7 turns
    managed = build_llm_context(
        current_query="那对缩水呢？",
        query_info={},
        conversation_history=history,
        conversation_summary=None,
        reranked_evidence=make_evidence(2),
    )
    kept_indices = [t["turn_index"] for t in managed.llm_context["recent_turns"]]
    assert len(kept_indices) == DEFAULT_RECENT_TURNS
    assert kept_indices == [5, 6, 7]
    assert managed.context_debug["older_turns_excluded_indices"] == [1, 2, 3, 4]


def test_risk_rules_include_high_risk_note_when_applicable() -> None:
    managed_low = build_llm_context(
        current_query="q",
        query_info={"risk_level": "low"},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=make_evidence(1),
    )
    assert HIGH_RISK_RULE not in managed_low.llm_context["risk_rules"]

    managed_high = build_llm_context(
        current_query="请直接给出生产参数",
        query_info={"risk_level": "high"},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=make_evidence(1),
    )
    assert HIGH_RISK_RULE in managed_high.llm_context["risk_rules"]


def test_tight_budget_drops_recent_turns_before_touching_evidence() -> None:
    """Per the required priority (current_query > evidence > summary > recent
    turns > older history), a tight token budget must shed recent turns (and,
    if still tight, the summary) while leaving every kept evidence entry intact."""
    history = [make_turn(i) for i in range(1, 5)]
    evidence = make_evidence(6, text_len=200)
    summary = "用户此前多次询问保压压力与翘曲、缩水的关系，已确认保压压力升高可改善两类缺陷。" * 3

    managed = build_llm_context(
        current_query="那对熔体温度呢？",
        query_info={"risk_level": "medium"},
        conversation_history=history,
        conversation_summary=summary,
        reranked_evidence=evidence,
        token_budget=120,  # deliberately far too small to fit everything
    )

    assert len(managed.llm_context["evidence_table"]) == min(6, DEFAULT_TOP_N_EVIDENCE)
    assert managed.context_debug["evidence_kept_ids"] == [
        e["evidence_id"] for e in managed.llm_context["evidence_table"]
    ]
    # history must have been trimmed (recent turns dropped and/or summary compressed/dropped)
    assert managed.context_debug["recent_turns_dropped_indices"] or managed.context_debug[
        "conversation_summary_dropped"
    ] or managed.context_debug["conversation_summary_compressed"]


def test_context_debug_reports_no_drops_when_budget_is_generous() -> None:
    history = [make_turn(i) for i in range(1, 3)]
    managed = build_llm_context(
        current_query="保压压力对翘曲有什么影响？",
        query_info={},
        conversation_history=history,
        conversation_summary="简短摘要",
        reranked_evidence=make_evidence(3),
        token_budget=DEFAULT_TOP_N_EVIDENCE * 1000,
    )
    assert managed.context_debug["over_budget"] is False
    assert managed.context_debug["recent_turns_dropped_indices"] == []
    assert managed.context_debug["conversation_summary_dropped"] is False
    assert managed.context_debug["conversation_summary_compressed"] is False
    assert managed.context_debug["evidence_dropped_ids"] == []


def test_managed_context_to_dict_shape() -> None:
    managed = build_llm_context(
        current_query="q",
        query_info={},
        conversation_history=None,
        conversation_summary=None,
        reranked_evidence=[],
    )
    assert isinstance(managed, ManagedContext)
    payload = managed.to_dict()
    assert set(payload.keys()) == {"llm_context", "context_debug"}


def test_render_prompt_includes_query_evidence_and_rules() -> None:
    managed = build_llm_context(
        current_query="保压压力对缩水有什么影响？",
        query_info={"risk_level": "high"},
        conversation_history=[make_turn(1)],
        conversation_summary="历史摘要内容",
        reranked_evidence=make_evidence(2),
    )
    prompt = render_prompt(managed.llm_context, query_info={"risk_level": "high"})

    assert "保压压力对缩水有什么影响？" in prompt
    assert "[E1]" in prompt
    assert "历史摘要内容" in prompt
    assert HIGH_RISK_RULE in prompt
    assert "结构化查询" in prompt


def test_estimate_tokens_grows_with_text_length() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("保压压力")
    long = estimate_tokens("保压压力" * 20)
    assert long > short
