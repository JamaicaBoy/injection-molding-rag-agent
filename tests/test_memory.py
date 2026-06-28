from pathlib import Path

from src.agent.memory import AgentMemory, MAX_SUMMARY_LENGTH, main


def test_agent_memory_records_safely_finds_similar_and_exports_stats(tmp_path: Path) -> None:
    path = tmp_path / "agent_memory.jsonl"
    timestamps = iter(("2026-06-27T10:00:00+00:00", "2026-06-27T11:00:00+00:00"))
    memory = AgentMemory(path, clock=lambda: next(timestamps))
    long_text = "Packing pressure reduces shrinkage. " + "FULL_TEXT_SHOULD_NOT_BE_STORED " * 20

    first = memory.record_query(
        query="我的邮箱 test@example.com，保压压力对缩水有什么影响？",
        intent="parameter_effect",
        evidence=[{
            "evidence_id": "E1",
            "text_preview": long_text,
            "full_text": "SECRET_FULL_TEXT",
            "paper_full_text": "SECRET_PAPER_TEXT",
            "raw_text": "SECRET_RAW_TEXT",
        }],
        answer_confidence="low",
        need_human_review=True,
        user_feedback={"comment": "手机号 13812345678，请继续分析", "raw_text": "SECRET_FEEDBACK_RAW_TEXT"},
    )
    memory.record_query(
        query="翘曲和模具温度有关吗？",
        intent="defect_diagnosis",
        evidence=[{"metadata": {"evidence_id": "E2"}, "text_preview": "Mold temperature affects warpage."}],
        answer_confidence=0.8,
        need_human_review=False,
    )

    stored_text = path.read_text(encoding="utf-8")
    assert "test@example.com" not in stored_text
    assert "13812345678" not in stored_text
    assert "SECRET_FULL_TEXT" not in stored_text
    assert "SECRET_PAPER_TEXT" not in stored_text
    assert "SECRET_RAW_TEXT" not in stored_text
    assert "SECRET_FEEDBACK_RAW_TEXT" not in stored_text
    assert "FULL_TEXT_SHOULD_NOT_BE_STORED FULL_TEXT_SHOULD_NOT_BE_STORED FULL_TEXT_SHOULD_NOT_BE_STORED FULL_TEXT_SHOULD_NOT_BE_STORED FULL_TEXT_SHOULD_NOT_BE_STORED FULL_TEXT_SHOULD_NOT_BE_STORED" not in stored_text
    assert first["top_evidence_ids"] == ["E1"]
    assert len(first["evidence_summaries"][0]["summary"]) <= MAX_SUMMARY_LENGTH

    similar = memory.find_recent_similar("保压大一点能减少缩水吗？", intent="parameter_effect", limit=2)
    assert len(similar) == 1
    assert similar[0]["top_evidence_ids"] == ["E1"]
    assert memory.similar_question_hint("保压压力和缩水的关系", intent="parameter_effect") is not None
    recent = memory.read_recent(1)
    assert len(recent) == 1
    assert recent[0]["query"] == "翘曲和模具温度有关吗？"

    stats = memory.export_statistics()
    assert stats["total_queries"] == 2
    assert {item["defect_type"] for item in stats["frequent_defects"]} == {"sink_mark/shrinkage", "warpage"}
    assert {item["parameter"] for item in stats["frequent_parameters"]} == {"packing_pressure", "mold_temperature"}
    assert stats["low_confidence_question_count"] == 1
    assert stats["human_review_count"] == 1


def test_memory_cli_demo_and_stats(tmp_path: Path, capsys) -> None:
    path = tmp_path / "agent_trace.jsonl"

    assert main(["--demo", "--memory_path", str(path), "--recent_n", "3"]) == 0
    demo_output = capsys.readouterr().out
    assert "Memory demo completed." in demo_output
    assert "records_written: 3" in demo_output
    assert "recent_records: 3" in demo_output
    assert "low_confidence_count: 1" in demo_output
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3

    assert main(["--stats", "--memory_path", str(path)]) == 0
    stats_output = capsys.readouterr().out
    assert "Memory statistics." in stats_output
    assert "total_records: 3" in stats_output
    assert "low_confidence_count: 1" in stats_output
