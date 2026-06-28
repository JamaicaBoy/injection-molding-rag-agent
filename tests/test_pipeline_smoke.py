from typing import Any

from src.agent.workflow import AgentWorkflow


def search_evidence(score: float = 0.9, conflict: bool = False) -> dict[str, Any]:
    return {
        "evidence_id": "E1",
        "paper_id": "paper_1",
        "title": "Packing pressure study",
        "authors": [],
        "year": 2024,
        "chunk_id": "chunk_1",
        "source_location": {"page": 3, "section": "Results", "table": None, "figure": None},
        "matched_text": "Packing pressure was associated with shrinkage in the experiment.",
        "matched_keywords": ["packing pressure", "shrinkage"],
        "relevance_score": score,
        "rerank_score": None,
        "evidence_quality": "high",
        "citation": "Packing pressure study (2024), Results, paper_1",
        "conflict": conflict,
    }


class FakeSearch:
    def __init__(self, results: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self.results = results or []
        self.fail = fail
        self.calls = 0

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("search unavailable")
        return {
            "query": kwargs["query"],
            "search_type": kwargs["search_type"],
            "results": self.results,
            "overall_confidence": max((item["relevance_score"] for item in self.results), default=0.0),
            "warnings": [],
        }


class FakeReranker:
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        return [{**item, "score": item["score"], "rerank_score": item["score"], "original_score": item["score"]} for item in candidates[:top_n]]


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, query_rewrite: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        return {
            "answer": "论文证据表明保压压力与缩水有关，但结论取决于实验条件。[E1]",
            "evidence_list": [
                {
                    "evidence_id": "E1",
                    "paper_id": "paper_1",
                    "title": "Packing pressure study",
                    "text_preview": evidence[0]["text_preview"],
                    "metadata": evidence[0].get("metadata", {}),
                }
            ],
            "confidence": "high",
            "limitations": [],
            "need_human_review": False,
        }


def fake_review(**kwargs: Any) -> dict[str, Any]:
    return {"review_ticket_id": "review_1", "status": "pending", "assigned_role": kwargs["required_expert_role"]}


def fake_gap(**kwargs: Any) -> dict[str, Any]:
    return {"gap_id": "gap_1", "status": "recorded", "gap_summary": kwargs["reason_for_gap"]}


def test_pipeline_happy_path_runs_all_answer_nodes() -> None:
    search = FakeSearch([search_evidence()])
    generator = FakeAnswerGenerator()
    workflow = AgentWorkflow(
        search_tool=search,
        reranker=FakeReranker(),
        answer_generator=generator,
        review_tool=fake_review,
        gap_tool=fake_gap,
    )

    output = workflow.run("保压压力对缩水有什么影响？")

    assert "[E1]" in output["answer"]
    assert not output["need_human_review"]
    assert output["node_history"] == [
        "classify_and_rewrite_query",
        "retrieve_evidence",
        "rerank_evidence",
        "decide_answer_or_review",
        "generate_answer",
        "citation_check",
        "final_response",
    ]
    assert output["step_count"] == 7
    assert [call["tool"] for call in output["tool_calls"]] == ["search_papers_tool", "answer_generator"]


def test_pipeline_stops_same_query_without_improvement_and_records_gap() -> None:
    search = FakeSearch([])
    workflow = AgentWorkflow(
        search_tool=search,
        reranker=FakeReranker(),
        answer_generator=FakeAnswerGenerator(),
        review_tool=fake_review,
        gap_tool=fake_gap,
    )

    output = workflow.run("未知新材料如何改善特殊缺陷？")

    assert search.calls == 2
    assert output["knowledge_gap"]["status"] == "recorded"
    assert output["answer"].startswith("当前论文库证据不足")
    assert output["step_count"] == 8
    assert not output["need_human_review"]


def test_pipeline_enters_review_after_more_than_two_tool_errors() -> None:
    search = FakeSearch(fail=True)
    workflow = AgentWorkflow(
        search_tool=search,
        reranker=FakeReranker(),
        answer_generator=FakeAnswerGenerator(),
        review_tool=fake_review,
        gap_tool=fake_gap,
    )

    output = workflow.run("查询相关论文")

    assert search.calls == 3
    assert output["need_human_review"]
    assert output["review_ticket"]["status"] == "pending"
    assert len([call for call in output["tool_calls"] if call["status"] == "error"]) == 3
    assert output["step_count"] <= 8


def test_pipeline_respects_max_tool_calls() -> None:
    generator = FakeAnswerGenerator()
    workflow = AgentWorkflow(
        max_tool_calls=1,
        search_tool=FakeSearch([search_evidence()]),
        reranker=FakeReranker(),
        answer_generator=generator,
        review_tool=fake_review,
        gap_tool=fake_gap,
    )

    output = workflow.run("保压压力对缩水有什么影响？")

    assert len(output["tool_calls"]) == 1
    assert generator.calls == 0
    assert output["need_human_review"]
    assert "max_tool_calls_exceeded: 1" in output["errors"]


def test_pipeline_routes_conflicting_evidence_to_review_before_generation() -> None:
    generator = FakeAnswerGenerator()
    workflow = AgentWorkflow(
        search_tool=FakeSearch([search_evidence(conflict=True)]),
        reranker=FakeReranker(),
        answer_generator=generator,
        review_tool=fake_review,
        gap_tool=fake_gap,
    )

    output = workflow.run("提高保压压力是否一定减少缩水？")

    assert output["need_human_review"]
    assert output["review_ticket"]["status"] == "pending"
    assert generator.calls == 0
    assert "generate_answer" not in output["node_history"]
