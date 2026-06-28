from src.agent.guardrails import check_answer_guardrails


def test_guardrail_rejects_parameter_range_not_in_evidence() -> None:
    result = check_answer_guardrails(
        "建议保压压力采用 80-120 MPa。",
        [{"text_preview": "The paper discusses packing pressure without a numeric range."}],
    )

    assert not result.passed
    assert result.unsupported_parameter_ranges == ["80-120 MPa"]
    assert "unsupported_process_parameter_range" in result.violations


def test_guardrail_rejects_direct_production_instruction() -> None:
    result = check_answer_guardrails(
        "立即将保压压力调到 120 MPa。",
        [{"text_preview": "The experiment used 120 MPa."}],
    )

    assert "paper_conclusion_presented_as_production_instruction" in result.violations
    assert result.need_human_review


def test_guardrail_rejects_single_certain_answer_for_conflicting_evidence() -> None:
    evidence = [
        {"text_preview": "Evidence A", "metadata": {"effect_direction": "increase_positive"}},
        {"text_preview": "Evidence B", "metadata": {"effect_direction": "increase_negative"}},
    ]
    result = check_answer_guardrails("提高该参数一定会改善质量。", evidence)

    assert result.evidence_conflict
    assert "single_certain_answer_despite_evidence_conflict" in result.violations


def test_guardrail_allows_supported_range_with_non_instructional_language() -> None:
    result = check_answer_guardrails(
        "论文实验覆盖 80-120 MPa，但只能作为候选方向，不能作为直接生产指令。",
        [{"text_preview": "The tested range was 80-120 MPa."}],
    )

    assert result.passed

