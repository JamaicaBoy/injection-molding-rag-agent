from pathlib import Path

from src.agent.conversation_state import ConversationState
from src.agent.conversation_summarizer import (
    ConversationSummary,
    rule_summarize_turns,
    should_trigger_summary,
)


def _turn(
    turn_index: int,
    user_question: str,
    cited_paper_ids: list[str] | None = None,
    key_entities: dict[str, list[str]] | None = None,
    need_human_review: bool = False,
    review_reason: str = "",
) -> dict:
    return {
        "turn_index": turn_index,
        "timestamp": f"2026-06-29T10:{turn_index:02d}:00+00:00",
        "user_question": user_question,
        "system_answer_brief": "答案",
        "key_entities": key_entities or {},
        "cited_paper_ids": cited_paper_ids or [],
        "need_human_review": need_human_review,
        "review_reason": review_reason,
    }


def test_should_trigger_summary_on_turn_count() -> None:
    turns = [_turn(i, f"问题{i}") for i in range(1, 10)]
    assert should_trigger_summary(turns) is True


def test_should_trigger_summary_false_for_short_conversation() -> None:
    turns = [_turn(i, f"问题{i}") for i in range(1, 4)]
    assert should_trigger_summary(turns) is False


def test_should_trigger_summary_on_token_threshold() -> None:
    long_question = "保压压力对翘曲的影响机理分析" * 60
    turns = [_turn(1, long_question)]
    assert should_trigger_summary(turns, token_threshold=50) is True


def test_rule_summarize_only_promotes_entities_with_citation() -> None:
    """Requirement #4: a model/turn guess with no supporting citation must
    never land in a confirmed_* field - it should be recorded as an open
    question instead."""
    turns = [
        _turn(
            1,
            "保压压力对翘曲有什么影响？",
            cited_paper_ids=["paper_1"],
            key_entities={"material": ["PP"], "defect_type": ["warpage"], "parameters": ["packing_pressure"]},
        ),
        _turn(
            2,
            "熔体温度会不会导致短射？",
            cited_paper_ids=[],
            key_entities={"material": ["ABS"], "defect_type": ["short_shot"], "parameters": ["melt_temperature"]},
        ),
    ]

    summary = rule_summarize_turns(turns)

    assert summary.confirmed_materials == ["PP"]
    assert summary.confirmed_defects == ["warpage"]
    assert summary.confirmed_parameters == ["packing_pressure"]
    assert summary.cited_papers == ["paper_1"]
    assert "ABS" not in summary.confirmed_materials
    assert "short_shot" not in summary.confirmed_defects
    assert "melt_temperature" not in summary.confirmed_parameters
    assert any("熔体温度" in q for q in summary.open_questions)


def test_rule_summarize_collects_human_review_items() -> None:
    turns = [
        _turn(1, "这个工艺是否安全？", need_human_review=True, review_reason="涉及高风险生产参数"),
    ]
    summary = rule_summarize_turns(turns)
    assert summary.human_review_items
    assert "涉及高风险生产参数" in summary.human_review_items[0]


def test_rule_summarize_merges_with_previous_summary() -> None:
    previous = ConversationSummary(
        user_goal="了解保压压力对翘曲的影响",
        confirmed_materials=["PP"],
        cited_papers=["paper_1"],
    )
    new_turns = [
        _turn(
            5,
            "PC材料呢？",
            cited_paper_ids=["paper_5"],
            key_entities={"material": ["PC"]},
        )
    ]

    summary = rule_summarize_turns(new_turns, previous_summary=previous)

    assert summary.user_goal == "了解保压压力对翘曲的影响"
    assert set(summary.confirmed_materials) == {"PP", "PC"}
    assert set(summary.cited_papers) == {"paper_1", "paper_5"}


def test_long_conversation_compresses_early_turns_keeps_recent(tmp_path: Path) -> None:
    """Requirement #8: once a conversation runs long, early turns must be
    summarized away while the most recent turns remain available verbatim."""
    state = ConversationState(conversation_id="conv-long", storage_dir=tmp_path)

    for i in range(1, 13):
        state.add_turn(
            f"第{i}轮关于保压压力的问题？",
            f"第{i}轮答案",
            evidence_list=[{"paper_id": f"paper_{i}"}],
        )

    # Long conversation should have triggered compression at least once: a
    # rolling summary now exists and the in-memory window never grows past
    # the trigger threshold (compression resets it down to recent_keep each
    # time the threshold is crossed).
    assert state.summary is not None
    assert len(state.turns) <= state.summary_trigger_turns

    # The most recent turns must be the literal tail of the conversation.
    recent_questions = [turn.user_question for turn in state.turns]
    assert recent_questions == [f"第{i}轮关于保压压力的问题？" for i in range(13 - len(state.turns), 13)]

    # Citations from the compressed early turns must have flowed into the summary.
    assert state.summary.cited_papers
    early_question = "第1轮关于保压压力的问题？"
    assert early_question not in recent_questions


def test_summary_dict_is_none_before_trigger(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-short", storage_dir=tmp_path)
    state.add_turn("保压压力对翘曲有什么影响？", "答案", evidence_list=[{"paper_id": "paper_1"}])

    assert state.summary_dict() is None


def test_clear_resets_summary(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-clear", storage_dir=tmp_path)
    for i in range(1, 13):
        state.add_turn(
            f"第{i}轮问题？",
            f"第{i}轮答案",
            evidence_list=[{"paper_id": f"paper_{i}"}],
        )
    assert state.summary is not None

    state.clear()

    assert state.summary is None
    assert state.turns == []
