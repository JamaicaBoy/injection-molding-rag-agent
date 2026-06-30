from __future__ import annotations

from typing import Any

from src.agent.langgraph_nodes import LangGraphDependencies
from src.agent.langgraph_workflow import LangGraphWorkflow


class MockRetriever:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls = 0

    def invoke(self, query: str) -> list[dict[str, Any]]:
        self.calls += 1
        return self.results


class MockReranker:
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        del query
        return [
            {**item, "rerank_score": item["score"], "original_score": item["score"]}
            for item in candidates[:top_n]
        ]


class MockLLM:
    def __init__(self, answer: str = "论文证据表明保压压力与缩水有关。[E1]") -> None:
        self.answer = answer
        self.calls = 0

    def generate(
        self,
        question: str,
        query_info: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del question, query_info
        self.calls += 1
        return {
            "answer": self.answer,
            "evidence_list": [
                {
                    **item,
                    "evidence_id": f"E{index}",
                }
                for index, item in enumerate(evidence, start=1)
            ],
            "confidence": "high",
            "limitations": [],
            "need_human_review": False,
        }


class MockFallbackLLM(MockLLM):
    def generate(
        self,
        question: str,
        query_info: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        output = super().generate(question, query_info, evidence)
        output.update(
            {
                "llm_mode": "mock",
                "llm_model": "",
                "llm_fallback_reason": "all local models failed",
                "confidence": "low",
            }
        )
        return output


class MockMemory:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record_query(self, **kwargs: Any) -> dict[str, Any]:
        self.records.append(kwargs)
        return kwargs


def evidence(index: int = 1, score: float = 0.9) -> dict[str, Any]:
    return {
        "chunk_id": f"chunk_{index}",
        "paper_id": f"paper_{index}",
        "title": f"Packing pressure study {index}",
        "section_name": "Results",
        "chunk_type": "text",
        "score": score,
        "text_preview": "Packing pressure was associated with shrinkage in the experiment.",
        "metadata": {},
    }


def fake_review(**kwargs: Any) -> dict[str, Any]:
    return {
        "review_ticket_id": "review_1",
        "status": "pending",
        "reason": kwargs["trigger_reason"],
    }


def make_workflow(retriever: MockRetriever, llm: MockLLM, memory: MockMemory) -> LangGraphWorkflow:
    dependencies = LangGraphDependencies(
        retriever=retriever,
        reranker=MockReranker(),
        answer_generator=llm,
        memory=memory,
        review_tool=fake_review,
        retrieval_top_k=5,
        trace_path=None,
    )
    return LangGraphWorkflow(dependencies=dependencies, max_steps=12)


def test_langgraph_happy_path_with_mock_retriever_and_llm() -> None:
    retriever = MockRetriever([evidence(index) for index in range(1, 4)])
    llm = MockLLM()
    memory = MockMemory()
    workflow = make_workflow(retriever, llm, memory)

    output = workflow.run("保压压力对缩水有什么影响？", conversation_id="conversation_1")

    assert output["conversation_id"] == "conversation_1"
    assert output["workflow_backend"] == "langgraph"
    assert output["answer"].endswith("[E1]")
    assert output["citations"] == ["E1"]
    assert output["need_human_review"] is False
    assert output["memory_updated"] is True
    assert output["node_history"] == [
        "query_rewrite_node",
        "retrieve_node",
        "rerank_node",
        "answer_node",
        "citation_guard_node",
        "risk_check_node",
        "memory_update_node",
    ]
    assert retriever.calls == 1
    assert llm.calls == 1
    assert len(memory.records) == 1


def test_langgraph_routes_empty_evidence_to_human_review_without_llm() -> None:
    retriever = MockRetriever([])
    llm = MockLLM()
    memory = MockMemory()
    workflow = make_workflow(retriever, llm, memory)

    output = workflow.run("没有证据的问题")

    assert output["need_human_review"] is True
    assert output["review_ticket"]["status"] == "pending"
    assert "human_review_node" in output["node_history"]
    assert llm.calls == 0
    assert len(memory.records) == 1


def test_langgraph_routes_failed_citation_guard_to_human_review() -> None:
    retriever = MockRetriever([evidence(index) for index in range(1, 4)])
    llm = MockLLM(answer="没有引用的草稿")
    memory = MockMemory()
    workflow = make_workflow(retriever, llm, memory)

    output = workflow.run("保压压力有什么影响？")

    assert output["need_human_review"] is True
    assert "citation_guard_node" in output["node_history"]
    assert "human_review_node" in output["node_history"]
    assert any("answer_has_no_citation" in error for error in output["errors"])


def test_langgraph_never_marks_mock_fallback_as_high_confidence() -> None:
    retriever = MockRetriever([evidence(index) for index in range(1, 6)])
    llm = MockFallbackLLM()
    memory = MockMemory()
    workflow = make_workflow(retriever, llm, memory)

    output = workflow.run("保压压力对缩水有什么影响？")

    assert output["llm_mode"] == "mock"
    assert output["confidence"] == "low"
    assert output["need_human_review"] is True
    assert output["human_review_reason"] == "tool_error"
    assert "human_review_node" in output["node_history"]


def test_langgraph_can_revise_once_after_citation_failure() -> None:
    retriever = MockRetriever([evidence(index) for index in range(1, 4)])
    llm = MockLLM(answer="没有引用的初稿")
    memory = MockMemory()
    dependencies = LangGraphDependencies(
        retriever=retriever,
        reranker=MockReranker(),
        answer_generator=llm,
        memory=memory,
        review_tool=fake_review,
        revision_generator=lambda query, draft, docs: "修订后仅保留有证据的结论。[E1]",
        retrieval_top_k=5,
        max_revisions=1,
        trace_path=None,
    )
    workflow = LangGraphWorkflow(dependencies=dependencies, max_steps=12)

    output = workflow.run("保压压力有什么影响？")

    assert output["answer"].endswith("[E1]")
    assert output["need_human_review"] is False
    assert output["node_history"].count("citation_guard_node") == 2
    assert "revise_answer_node" in output["node_history"]
