from __future__ import annotations

import uuid
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from src.agent.langgraph_nodes import (
    LangGraphDependencies,
    answer_node,
    citation_guard_node,
    human_review_node,
    memory_update_node,
    query_rewrite_node,
    rerank_node,
    retrieve_node,
    revise_answer_node,
    risk_check_node,
)
from src.agent.langgraph_state import AgentState


class LangGraphWorkflow:
    def __init__(
        self,
        *,
        max_steps: int = 12,
        dependencies: LangGraphDependencies | None = None,
        **dependency_overrides: Any,
    ) -> None:
        if max_steps < 8:
            raise ValueError("max_steps must be at least 8 for the full workflow")
        self.max_steps = max_steps
        self.dependencies = dependencies or LangGraphDependencies(**dependency_overrides)
        self.graph = self._build_graph()

    def _build_graph(self):
        deps = self.dependencies
        builder = StateGraph(AgentState)
        builder.add_node("query_rewrite", lambda state: query_rewrite_node(state, deps))
        builder.add_node("retrieve", lambda state: retrieve_node(state, deps))
        builder.add_node("rerank", lambda state: rerank_node(state, deps))
        builder.add_node("answer", lambda state: answer_node(state, deps))
        builder.add_node("citation_guard", lambda state: citation_guard_node(state, deps))
        builder.add_node("revise_answer", lambda state: revise_answer_node(state, deps))
        builder.add_node("risk_check", lambda state: risk_check_node(state, deps))
        builder.add_node("human_review", lambda state: human_review_node(state, deps))
        builder.add_node("memory_update", lambda state: memory_update_node(state, deps))

        builder.add_edge(START, "query_rewrite")
        builder.add_conditional_edges(
            "query_rewrite",
            self._route_after_rewrite,
            {"retrieve": "retrieve", "human_review": "human_review"},
        )
        builder.add_edge("retrieve", "rerank")
        builder.add_conditional_edges(
            "rerank",
            self._route_after_rerank,
            {"answer": "answer", "human_review": "human_review"},
        )
        builder.add_edge("answer", "citation_guard")
        builder.add_conditional_edges(
            "citation_guard",
            self._route_after_citation,
            {
                "risk_check": "risk_check",
                "revise_answer": "revise_answer",
                "human_review": "human_review",
            },
        )
        builder.add_edge("revise_answer", "citation_guard")
        builder.add_conditional_edges(
            "risk_check",
            self._route_after_risk,
            {"memory_update": "memory_update", "human_review": "human_review"},
        )
        builder.add_edge("human_review", "memory_update")
        builder.add_edge("memory_update", END)
        return builder.compile(name="injection-molding-rag-langgraph")

    @staticmethod
    def _route_after_rewrite(state: AgentState) -> Literal["retrieve", "human_review"]:
        return "human_review" if state.get("need_human_review") else "retrieve"

    @staticmethod
    def _route_after_rerank(state: AgentState) -> Literal["answer", "human_review"]:
        return "human_review" if state.get("need_human_review") else "answer"

    def _route_after_citation(
        self, state: AgentState
    ) -> Literal["risk_check", "revise_answer", "human_review"]:
        if state.get("citation_passed"):
            return "risk_check"
        can_revise = (
            self.dependencies.revision_generator is not None
            and int(state.get("revision_count", 0)) < self.dependencies.max_revisions
            and state.get("human_review_reason") != "tool_error"
        )
        return "revise_answer" if can_revise else "human_review"

    @staticmethod
    def _route_after_risk(state: AgentState) -> Literal["memory_update", "human_review"]:
        return "memory_update" if state.get("risk_passed") else "human_review"

    def run(
        self,
        query: str,
        conversation_id: str | None = None,
        messages: list[Any] | None = None,
    ) -> dict[str, Any]:
        initial: AgentState = {
            "query": query,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "messages": messages or [HumanMessage(content=query)],
            "query_info": {},
            "retrieved_docs": [],
            "reranked_docs": [],
            "evidence_list": [],
            "draft_answer": "",
            "final_answer": "",
            "citations": [],
            "confidence": "low",
            "confidence_reason": "Workflow has not evaluated evidence yet.",
            "need_human_review": False,
            "human_review_reason": "",
            "errors": [],
            "step_count": 0,
            "limitations": [],
            "citation_passed": False,
            "risk_passed": False,
            "revision_count": 0,
            "review_ticket": None,
            "memory_updated": False,
            "node_history": [],
            "trace": [],
        }
        try:
            final_state = self.graph.invoke(initial, config={"recursion_limit": self.max_steps})
        except GraphRecursionError:
            final_state = {
                **initial,
                "final_answer": "工作流达到最大步骤限制，已停止并转人工复核。",
                "need_human_review": True,
                "human_review_reason": "tool_error",
                "confidence": "low",
                "confidence_reason": "The workflow exceeded its step limit.",
                "errors": [f"max_steps_exceeded: {self.max_steps}"],
                "step_count": self.max_steps,
            }
        return self._to_output(final_state)

    def invoke(
        self, input: str | dict[str, Any], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        del config
        if isinstance(input, str):
            return self.run(input)
        return self.run(
            str(input.get("query", "")),
            conversation_id=input.get("conversation_id"),
            messages=input.get("messages"),
        )

    @staticmethod
    def _to_output(state: AgentState) -> dict[str, Any]:
        query_info = dict(state.get("query_info", {}))
        answer = (
            state.get("final_answer")
            or state.get("draft_answer")
            or "当前论文库证据不足。"
        )
        retrieved = list(state.get("retrieved_docs", []))
        reranked = list(state.get("reranked_docs", []))
        evidence = list(state.get("evidence_list") or reranked)
        top_score = max(
            (
                float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
                for item in reranked
            ),
            default=0.0,
        )
        return {
            "query": state.get("query", ""),
            "conversation_id": state.get("conversation_id", ""),
            "query_info": query_info,
            "normalized_query": query_info.get("normalized_query", state.get("query", "")),
            "intent": query_info.get("intent", "general_qa"),
            "answer": answer,
            "evidence_list": evidence,
            "citations": list(state.get("citations", [])),
            "confidence": state.get("confidence", "low"),
            "confidence_reason": state.get("confidence_reason", ""),
            "need_human_review": bool(state.get("need_human_review", False)),
            "human_review_reason": state.get("human_review_reason", ""),
            "limitations": list(state.get("limitations", [])),
            "errors": list(state.get("errors", [])),
            "step_count": int(state.get("step_count", 0)),
            "node_history": list(state.get("node_history", [])),
            "trace": list(state.get("trace", [])),
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "top_score": top_score,
            "review_ticket": state.get("review_ticket"),
            "memory_updated": bool(state.get("memory_updated", False)),
            "workflow_backend": "langgraph",
        }
