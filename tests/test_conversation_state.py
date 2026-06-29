from pathlib import Path

from src.agent.conversation_state import (
    ConversationState,
    ensure_current_conversation_state,
    has_followup_marker,
    new_conversation_id,
    resolve_followup_query,
)


class LegacyConversationState:
    def __init__(self, storage_dir: Path) -> None:
        self.conversation_id = "legacy-conversation"
        self.max_turns = 20
        self.storage_dir = storage_dir
        self.turns = []


def test_new_conversation_id_is_unique() -> None:
    assert new_conversation_id() != new_conversation_id()


def test_upgrade_legacy_session_state_adds_summary_without_losing_turns(tmp_path: Path) -> None:
    legacy = LegacyConversationState(tmp_path)
    legacy.turns = [
        {
            "turn_index": 1,
            "timestamp": "2026-06-29T10:00:00+00:00",
            "user_question": "保压压力有什么影响？",
            "system_answer_brief": "候选证据说明保压压力会影响制件质量。",
            "key_entities": {"parameters": ["packing_pressure"]},
            "cited_paper_ids": ["paper_1"],
        }
    ]

    upgraded = ensure_current_conversation_state(legacy)

    assert isinstance(upgraded, ConversationState)
    assert upgraded.summary is None
    assert upgraded.summary_dict() is None
    assert len(upgraded.turns) == 1
    assert upgraded.turns[0].cited_paper_ids == ["paper_1"]


def test_add_turn_persists_jsonl_and_trims_to_max_turns(tmp_path: Path) -> None:
    timestamps = iter(f"2026-06-29T10:0{i}:00+00:00" for i in range(10))
    state = ConversationState(
        conversation_id="conv-1",
        max_turns=2,
        storage_dir=tmp_path,
        clock=lambda: next(timestamps),
    )

    state.add_turn(
        "保压压力对翘曲有什么影响？",
        "保压压力升高通常能降低翘曲。",
        evidence_list=[{"paper_id": "paper_1"}, {"paper_id": "paper_2"}],
    )
    state.add_turn("那对缩水呢？", "缩水也受保压压力影响。", evidence_list=[{"paper_id": "paper_3"}])
    state.add_turn("熔体温度呢？", "熔体温度也有影响。", evidence_list=[{"paper_id": "paper_4"}])

    assert len(state.turns) == 2
    assert [turn.user_question for turn in state.turns] == ["那对缩水呢？", "熔体温度呢？"]

    stored_lines = (tmp_path / "conv-1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(stored_lines) == 3  # full history persisted even though in-memory trims to max_turns


def test_add_turn_does_not_store_full_evidence_text(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-2", storage_dir=tmp_path)
    long_evidence_text = "FULL_PAPER_TEXT " * 50
    state.add_turn(
        "保压压力对缩水有什么影响？",
        "保压压力升高能降低缩水。",
        evidence_list=[
            {"paper_id": "paper_1", "matched_text": long_evidence_text, "full_text": "SECRET"}
        ],
    )
    stored_text = (tmp_path / "conv-2.jsonl").read_text(encoding="utf-8")
    assert "FULL_PAPER_TEXT" not in stored_text
    assert "SECRET" not in stored_text
    assert "paper_1" in stored_text


def test_clear_resets_memory_and_deletes_file(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-3", storage_dir=tmp_path)
    state.add_turn("保压压力对缩水有什么影响？", "答案", evidence_list=[{"paper_id": "paper_1"}])
    assert (tmp_path / "conv-3.jsonl").exists()

    state.clear()

    assert state.turns == []
    assert not (tmp_path / "conv-3.jsonl").exists()


def test_load_reconstructs_state_from_disk(tmp_path: Path) -> None:
    original = ConversationState(conversation_id="conv-4", storage_dir=tmp_path)
    original.add_turn("保压压力对缩水有什么影响？", "答案一", evidence_list=[{"paper_id": "paper_1"}])
    original.add_turn("模具温度呢？", "答案二", evidence_list=[{"paper_id": "paper_2"}])

    reloaded = ConversationState.load("conv-4", storage_dir=tmp_path)

    assert len(reloaded.turns) == 2
    assert reloaded.turns[-1].user_question == "模具温度呢？"
    assert reloaded.turns[0].cited_paper_ids == ["paper_1"]


def test_has_followup_marker_detects_pronoun_questions() -> None:
    assert has_followup_marker("那对缩水呢？")
    assert not has_followup_marker("保压压力对缩水有什么影响？")


def test_resolve_followup_query_fills_parameter_from_previous_turn(tmp_path: Path) -> None:
    """Second-turn question 'that on shrinkage?' must inherit the packing-pressure
    parameter mentioned in the first turn, since it only restates the new defect.
    """
    state = ConversationState(conversation_id="conv-5", storage_dir=tmp_path)
    state.add_turn(
        "保压压力对翘曲有什么影响？",
        "保压压力升高通常能降低翘曲。",
        evidence_list=[{"paper_id": "paper_1"}],
    )

    resolved = resolve_followup_query("那对缩水呢？", state)

    assert resolved.defect_type == ["sink_mark/shrinkage"]
    assert resolved.parameters == ["packing_pressure"]
    assert "packing_pressure" in resolved.must_have_terms


def test_resolve_followup_query_keeps_explicit_entities_untouched(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-6", storage_dir=tmp_path)
    state.add_turn(
        "保压压力对翘曲有什么影响？",
        "保压压力升高通常能降低翘曲。",
        evidence_list=[{"paper_id": "paper_1"}],
    )

    resolved = resolve_followup_query("那模具温度对缩水呢？", state)

    assert resolved.parameters == ["mold_temperature"]
    assert resolved.defect_type == ["sink_mark/shrinkage"]


def test_resolve_followup_query_promotes_general_qa_intent_when_parameter_filled(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-8", storage_dir=tmp_path)
    state.add_turn(
        "保压压力对翘曲有什么影响？",
        "保压压力升高通常能降低翘曲。",
        evidence_list=[{"paper_id": "paper_1"}],
    )

    resolved = resolve_followup_query("这个呢？", state)

    assert resolved.parameters == ["packing_pressure"]
    assert resolved.intent == "parameter_effect"


def test_resolve_followup_query_without_marker_or_history_is_plain_rewrite(tmp_path: Path) -> None:
    state = ConversationState(conversation_id="conv-7", storage_dir=tmp_path)
    resolved_no_history = resolve_followup_query("那对缩水呢？", state)
    assert resolved_no_history.parameters == []

    state.add_turn("保压压力对翘曲有什么影响？", "答案", evidence_list=[{"paper_id": "paper_1"}])
    resolved_no_marker = resolve_followup_query("熔体温度对短射有什么影响？", state)
    assert resolved_no_marker.parameters == ["melt_temperature"]
