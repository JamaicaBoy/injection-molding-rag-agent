import json
from pathlib import Path
from typing import Any

from src.agent.tools import (
    defect_diagnosis_tool,
    evidence_extract_tool,
    human_review_tool,
    knowledge_gap_tool,
    method_compare_tool,
    parameter_effect_tool,
    search_papers_tool,
)


def retrieval_result(
    chunk_id: str = "chunk_1",
    paper_id: str = "paper_1",
    text: str = "Increasing packing pressure reduces shrinkage in the reported experiment.",
    score: float = 0.85,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": "Injection molding evidence",
        "section_name": "Results",
        "chunk_type": "knowledge_card",
        "score": score,
        "source": "bm25",
        "text_preview": text,
        "metadata": {"year": "2024", "page_start": 3, "card_type": "parameter_card"},
    }


def evidence_result(
    evidence_id: str = "E1",
    paper_id: str = "paper_1",
    text: str = "Increasing packing pressure reduces shrinkage in the reported experiment.",
    score: float = 0.85,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "paper_id": paper_id,
        "title": "Injection molding evidence",
        "year": 2024,
        "chunk_id": f"chunk_{evidence_id}",
        "source_location": {"page": 3, "section": "Results", "table": None, "figure": None},
        "matched_text": text,
        "matched_keywords": ["packing pressure", "shrinkage"],
        "relevance_score": score,
        "rerank_score": score,
        "evidence_quality": "high",
        "citation": "Injection molding evidence (2024), Results, paper_1",
    }


class FakeBM25:
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [retrieval_result()]


class FailingDense:
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise RuntimeError("embedding memory pressure")


class StaticRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.results[:top_k]


def fake_search(**kwargs: Any) -> dict[str, Any]:
    return {
        "query": kwargs["query"],
        "search_type": kwargs.get("search_type", "hybrid"),
        "results": [evidence_result("E1", "paper_1"), evidence_result("E2", "paper_2")],
        "overall_confidence": 0.85,
        "warnings": [],
    }


def test_search_papers_tool_returns_auditable_schema() -> None:
    output = search_papers_tool(
        "保压压力对缩水的影响",
        search_type="keyword",
        top_k=1,
        rerank=False,
        _bm25=FakeBM25(),
    )

    result = output["results"][0]
    assert result["evidence_id"] == "E1"
    assert result["paper_id"] == "paper_1"
    assert result["source_location"]["page"] == 3
    assert result["relevance_score"] == 0.85
    assert result["citation"]


def test_hybrid_search_falls_back_to_bm25_when_dense_is_unavailable() -> None:
    output = search_papers_tool(
        "保压压力对缩水的影响",
        search_type="hybrid",
        top_k=1,
        rerank=False,
        _bm25=FakeBM25(),
        _dense=FailingDense(),
    )

    assert len(output["results"]) == 1
    assert output["results"][0]["paper_id"] == "paper_1"
    assert "dense_unavailable_bm25_fallback:RuntimeError" in output["warnings"]


def test_search_papers_tool_deduplicates_identical_chunk_text() -> None:
    repeated = "Ensure sufficient clamping force is applied. Figure 22 Flash at sleeve."
    bm25 = StaticRetriever(
        [
            retrieval_result("chunk_a", "paper_a", repeated, 0.91),
            retrieval_result(
                "chunk_unique",
                "paper_c",
                "Excess injection pressure can promote flash.",
                0.72,
            ),
        ]
    )
    dense = StaticRetriever(
        [retrieval_result("chunk_b", "paper_b", repeated, 0.88)]
    )

    output = search_papers_tool(
        "飞边可能是什么原因",
        search_type="hybrid",
        top_k=3,
        rerank=False,
        _bm25=bm25,
        _dense=dense,
    )

    matched_texts = [item["matched_text"] for item in output["results"]]
    assert len(output["results"]) == 2
    assert matched_texts.count(repeated) == 1


def test_defect_diagnosis_tool_returns_candidates_not_final_decision() -> None:
    output = defect_diagnosis_tool(
        "产品出现缩水，可能是什么原因？",
        defect_type="缩水",
        product_context={"material": "PP", "defect_location": "thick wall"},
        _search_fn=fake_search,
    )

    assert output["standardized_defect_type"] == "sink_mark/shrinkage"
    assert output["possible_causes"]
    assert output["possible_causes"][0]["supporting_evidence_ids"] == ["E1"]
    assert "不能作为最终诊断" in output["not_final_decision_notice"]


def test_parameter_effect_tool_summarizes_supported_trend() -> None:
    output = parameter_effect_tool(
        "packing pressure",
        target_quality_or_defect="shrinkage",
        evidence_scope={"top_k": 5, "min_relevance_score": 0.5},
        _search_fn=fake_search,
    )

    assert output["effect_direction"] == "increase_positive"
    assert len(output["mechanisms"]) == 2
    assert output["can_auto_answer"]
    assert not output["need_human_review"]


def test_method_compare_tool_keeps_evidence_links() -> None:
    output = method_compare_tool(
        [{"method_name": "GA"}, {"method_name": "PSO"}],
        application_context={"task_type": "parameter_optimization", "available_data": ["process parameters"]},
        _search_fn=fake_search,
    )

    assert [row["method_name"] for row in output["comparison_table"]] == ["GA", "PSO"]
    assert output["comparison_table"][0]["supporting_evidence_ids"] == ["E1", "E2"]
    assert "不能替代" in output["recommendation_for_context"]["not_decision_notice"]


def test_evidence_extract_tool_deduplicates_and_normalizes_entities() -> None:
    raw = [
        {"evidence_id": "E1", "paper_id": "paper_1", "matched_text": "保压压力提高可能减少缩水。", "source_location": {"page": 1, "section": "Results"}, "relevance_score": 0.8},
        {"evidence_id": "E2", "paper_id": "paper_2", "matched_text": "保压压力提高可能减少缩水。", "source_location": {"page": 2, "section": "Results"}, "relevance_score": 0.7},
    ]
    output = evidence_extract_tool(raw, extract_schema="parameter_effect")

    assert output["deduplication_report"]["original_count"] == 2
    assert output["deduplication_report"]["final_count"] == 1
    assert output["deduplication_report"]["merged_items"] == ["E2->E1"]
    assert output["evidence_table"][0]["entities"]["parameters"] == ["packing_pressure"]


def test_human_review_tool_creates_pending_audit_ticket(tmp_path: Path) -> None:
    store = tmp_path / "review_tickets.jsonl"
    output = human_review_tool(
        case_id="case_1",
        trigger_reason="high_risk",
        user_question="直接告诉我生产参数怎么调",
        agent_intermediate_result={"candidate": "increase pressure"},
        evidence_ids=["E1"],
        risk_level="high",
        confidence_score=0.5,
        required_expert_role="process_engineer",
        review_questions=["是否适用于当前材料？"],
        _ticket_store=store,
    )

    assert output["status"] == "pending"
    assert output["assigned_role"] == "process_engineer"
    record = json.loads(store.read_text(encoding="utf-8").strip())
    assert record["audit_log"]["evidence_ids"] == ["E1"]


def test_knowledge_gap_tool_records_and_detects_duplicate(tmp_path: Path) -> None:
    store = tmp_path / "knowledge_gaps.jsonl"
    kwargs = {
        "user_question": "新材料 X 的透过率怎么优化？",
        "missing_information_type": "no_relevant_paper",
        "attempted_queries": ["material X transmittance"],
        "retrieved_evidence_ids": [],
        "reason_for_gap": "当前论文库没有相关材料证据。",
        "suggested_next_actions": ["add_papers", "ask_expert"],
        "priority": "high",
        "_gap_store": store,
    }

    first = knowledge_gap_tool(**kwargs)
    second = knowledge_gap_tool(**kwargs)

    assert first["status"] == "recorded"
    assert second["status"] == "duplicate"
    assert second["gap_id"] == first["gap_id"]
    assert len(store.read_text(encoding="utf-8").splitlines()) == 1
