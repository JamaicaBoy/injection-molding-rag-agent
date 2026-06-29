from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agent.langgraph_nodes import LangGraphDependencies
from src.agent.langgraph_workflow import LangGraphWorkflow
from src.agent.workflow import AgentWorkflow


class SharedSearch:
    def __init__(self, count: int = 5, score: float = 0.82) -> None:
        self.count = count
        self.score = score

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "results": [
                {
                    "evidence_id": f"E{index}",
                    "paper_id": f"paper_{index}",
                    "title": f"Packing pressure and flash study {index}",
                    "chunk_id": f"chunk_{index}",
                    "chunk_type": "knowledge_card" if index == 1 else "text",
                    "source_location": {"section": "Results", "page": index},
                    "matched_text": "Packing pressure affects flash formation and process quality.",
                    "relevance_score": self.score - index * 0.01,
                    "rerank_score": None,
                }
                for index in range(1, self.count + 1)
            ],
            "overall_confidence": self.score,
            "warnings": [],
        }


class SharedReranker:
    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        del query
        return [
            {
                **item,
                "original_score": float(item["score"]),
                "rerank_score": float(item["score"]) + 0.03,
                "score": float(item["score"]) + 0.03,
            }
            for item in candidates[:top_n]
        ]


class SharedAnswerGenerator:
    def generate(
        self, question: str, query_info: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        del question, query_info
        return {
            "answer": "论文证据表明保压压力会影响飞边风险，具体方向取决于工艺条件。[E1]",
            "evidence_list": [
                {**item, "evidence_id": f"E{index}"}
                for index, item in enumerate(evidence, start=1)
            ],
            "confidence": "high",
            "limitations": [],
            "need_human_review": False,
        }


class MemoryStub:
    def record_query(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


def review_stub(**kwargs: Any) -> dict[str, Any]:
    return {"status": "pending", "reason": kwargs["trigger_reason"]}


def build_workflows(trace_path: Path) -> tuple[AgentWorkflow, LangGraphWorkflow]:
    search = SharedSearch()
    reranker = SharedReranker()
    generator = SharedAnswerGenerator()
    classic = AgentWorkflow(
        search_tool=search,
        reranker=reranker,
        answer_generator=generator,
        retrieval_top_k=5,
    )
    langgraph = LangGraphWorkflow(
        dependencies=LangGraphDependencies(
            retriever=lambda query, top_k: search(query=query, top_k=top_k),
            reranker=reranker,
            answer_generator=generator,
            memory=MemoryStub(),
            review_tool=review_stub,
            retrieval_top_k=5,
            trace_path=trace_path,
        ),
        max_steps=12,
    )
    return classic, langgraph


def test_classic_and_langgraph_retrieve_real_candidates(tmp_path: Path) -> None:
    classic, langgraph = build_workflows(tmp_path / "langgraph_trace.jsonl")
    query = "保压压力对飞边有什么影响"

    classic_output = classic.run(query)
    graph_output = langgraph.run(query)

    assert len(classic_output["evidence_list"]) > 0
    assert len(graph_output["evidence_list"]) > 0
    assert graph_output["retrieved_count"] == 5
    assert graph_output["reranked_count"] == 5
    assert graph_output["need_human_review"] is False
    assert graph_output["human_review_reason"] == "normal_answer"
    assert graph_output["confidence"] in {"medium", "high"}


def test_high_risk_parameter_request_still_requires_review(tmp_path: Path) -> None:
    _, langgraph = build_workflows(tmp_path / "langgraph_trace.jsonl")

    output = langgraph.run("直接给我量产时保压压力设多少")

    assert output["need_human_review"] is True
    assert output["human_review_reason"] == "high_risk_action"


def test_trace_contains_required_nodes_and_safe_summaries(tmp_path: Path) -> None:
    trace_path = tmp_path / "langgraph_trace.jsonl"
    _, langgraph = build_workflows(trace_path)

    output = langgraph.run("保压压力对飞边有什么影响")
    executed = set(output["node_history"])

    assert {
        "retrieve_node",
        "rerank_node",
        "answer_node",
        "citation_guard_node",
    }.issubset(executed)
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert {record["node_name"] for record in records}.issuperset(executed)
    assert all("retrieved_count" in record and "top_score" in record for record in records)
    assert all("text_preview" not in record for record in records)


def test_none_rerank_score_does_not_erase_relevance_score(tmp_path: Path) -> None:
    _, langgraph = build_workflows(tmp_path / "langgraph_trace.jsonl")

    output = langgraph.run("保压压力对飞边有什么影响")

    rerank_trace = next(
        item for item in output["trace"] if item["node_name"] == "rerank_node"
    )
    assert rerank_trace["top_score"] > 0
