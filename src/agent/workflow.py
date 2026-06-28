from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

from src.agent.guardrails import check_answer_guardrails, evidence_has_conflict
from src.agent.state import AgentState
from src.agent.tools import human_review_tool, knowledge_gap_tool, search_papers_tool
from src.rag.answer_generator import AnswerGenerator
from src.rag.citation_guard import check_citations
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.reranker import Reranker


class AgentWorkflow:
    """A small, auditable state machine for the paper RAG Agent."""

    def __init__(
        self,
        max_steps: int = 8,
        max_tool_calls: int = 5,
        evidence_score_threshold: float = 0.30,
        retrieval_top_k: int = 10,
        *,
        rewriter: Callable[[str], Any] = rewrite_query,
        search_tool: Callable[..., dict[str, Any]] = search_papers_tool,
        reranker: Any | None = None,
        answer_generator: Any | None = None,
        citation_checker: Callable[[str, list[dict[str, Any]]], Any] = check_citations,
        guardrail_checker: Callable[[str, list[dict[str, Any]]], Any] = check_answer_guardrails,
        review_tool: Callable[..., dict[str, Any]] = human_review_tool,
        gap_tool: Callable[..., dict[str, Any]] = knowledge_gap_tool,
    ) -> None:
        if max_steps <= 0 or max_tool_calls <= 0:
            raise ValueError("max_steps and max_tool_calls must be positive")
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.evidence_score_threshold = evidence_score_threshold
        self.retrieval_top_k = retrieval_top_k
        self.rewriter = rewriter
        self.search_tool = search_tool
        self.reranker = reranker or Reranker(mode="rule")
        self.answer_generator = answer_generator or AnswerGenerator(mode="ollama")
        self.citation_checker = citation_checker
        self.guardrail_checker = guardrail_checker
        self.review_tool = review_tool
        self.gap_tool = gap_tool

    def run(self, query: str) -> dict[str, Any]:
        state = AgentState(query=query)
        if not self.classify_and_rewrite_query(state):
            return self.final_response(state)

        while state.step_count < self.max_steps:
            retrieved = self.retrieve_evidence(state)
            if not retrieved:
                if state.tool_error_count > 2:
                    self.decide_answer_or_review(state)
                    break
                if state.step_count >= self.max_steps - 1:
                    state.need_human_review = True
                    break
                continue

            self.rerank_evidence(state)
            decision = self.decide_answer_or_review(state)
            if decision == "retry_retrieval":
                continue
            if decision == "generate_answer":
                if self.generate_answer(state):
                    self.citation_check(state)
            break

        return self.final_response(state)

    def classify_and_rewrite_query(self, state: AgentState) -> bool:
        if not self._enter_node(state, "classify_and_rewrite_query"):
            return False
        try:
            rewritten = self.rewriter(state.query)
            data = self._as_dict(rewritten)
            state.query_rewrite = data
            state.normalized_query = str(data.get("normalized_query") or state.query)
            state.intent = str(data.get("intent") or "general_qa")
            state.risk_level = str(data.get("risk_level") or "low")
            return True
        except Exception as exc:
            state.errors.append(f"classify_and_rewrite_query: {type(exc).__name__}: {exc}")
            state.need_human_review = True
            return False

    def retrieve_evidence(self, state: AgentState) -> bool:
        if not self._enter_node(state, "retrieve_evidence"):
            return False
        retrieval_query = state.normalized_query or state.query
        result = self._call_tool(
            state,
            "search_papers_tool",
            lambda: self.search_tool(
                query=retrieval_query,
                search_type="hybrid",
                filters={},
                top_k=self.retrieval_top_k,
                rerank=False,
                return_chunks=True,
                language="auto",
            ),
            {"query": retrieval_query, "top_k": self.retrieval_top_k},
        )
        if result is None:
            state.retrieved_evidence = []
            return False

        results = list(result.get("results", []))
        candidates = [self._search_result_to_candidate(item) for item in results]
        state.retrieved_evidence = candidates
        state.confidence = float(result.get("overall_confidence", self._evidence_confidence(candidates)))
        self._record_retrieval(state, retrieval_query, candidates)
        return True

    def rerank_evidence(self, state: AgentState) -> bool:
        if not self._enter_node(state, "rerank_evidence"):
            return False
        if not state.retrieved_evidence:
            state.confidence = 0.0
            return True
        try:
            state.retrieved_evidence = self.reranker.rerank(
                state.query,
                state.retrieved_evidence,
                top_n=min(self.retrieval_top_k, len(state.retrieved_evidence)),
            )
            state.confidence = self._evidence_confidence(state.retrieved_evidence)
            return True
        except Exception as exc:
            state.tool_error_count += 1
            state.errors.append(f"rerank_evidence: {type(exc).__name__}: {exc}")
            if state.tool_error_count > 2:
                state.need_human_review = True
            return False

    def decide_answer_or_review(self, state: AgentState) -> str:
        if not self._enter_node(state, "decide_answer_or_review"):
            state.need_human_review = True
            return "human_review"

        if state.tool_error_count > 2:
            self._enter_human_review(state, "low_confidence", "工具连续报错超过 2 次。")
            return "human_review"
        if state.risk_level in {"high", "critical"}:
            self._enter_human_review(state, "high_risk", "问题涉及直接生产参数或高风险决策。")
            return "human_review"
        if evidence_has_conflict(state.retrieved_evidence):
            self._enter_human_review(state, "evidence_conflict", "检索证据存在冲突，不能给出单一确定答案。")
            return "human_review"

        low_evidence = not state.retrieved_evidence or self._evidence_confidence(state.retrieved_evidence) < self.evidence_score_threshold
        if low_evidence:
            if len(state.retrieval_history) < 2 and not state.retrieval_stalled and state.step_count < self.max_steps - 3:
                return "retry_retrieval"
            self._record_knowledge_gap(state)
            return "knowledge_gap"
        return "generate_answer"

    def generate_answer(self, state: AgentState) -> bool:
        if not self._enter_node(state, "generate_answer"):
            return False
        result = self._call_tool(
            state,
            "answer_generator",
            lambda: self.answer_generator.generate(state.query, state.query_rewrite, state.retrieved_evidence),
            {"evidence_count": len(state.retrieved_evidence)},
        )
        if result is None:
            state.need_human_review = True
            return False
        data = self._as_dict(result)
        state.answer = str(data.get("answer", ""))
        state.generated_evidence = list(data.get("evidence_list", []))
        state.confidence = data.get("confidence", "low")
        state.limitations = list(data.get("limitations", []))
        if data.get("need_human_review"):
            self._enter_human_review(state, "low_confidence", "答案生成器要求人工复核。")
        return bool(state.answer)

    def citation_check(self, state: AgentState) -> bool:
        if not self._enter_node(state, "citation_check"):
            return False
        evidence = state.generated_evidence or state.retrieved_evidence
        try:
            citation_result = self.citation_checker(state.answer, evidence)
            guardrail_result = self.guardrail_checker(state.answer, evidence)
            citation_data = self._as_dict(citation_result)
            guardrail_data = self._as_dict(guardrail_result)
            violations = [*citation_data.get("issues", []), *guardrail_data.get("violations", [])]
            state.guardrail_violations = list(dict.fromkeys(str(item) for item in violations))
            if citation_data.get("high_risk") or guardrail_data.get("need_human_review"):
                state.errors.extend(f"guardrail: {item}" for item in state.guardrail_violations)
                self._enter_human_review(state, "safety_risk", "答案未通过引用或生产安全校验。")
                return False
            return True
        except Exception as exc:
            state.errors.append(f"citation_check: {type(exc).__name__}: {exc}")
            self._enter_human_review(state, "safety_risk", "引用校验工具执行失败。")
            return False

    def final_response(self, state: AgentState) -> dict[str, Any]:
        self._enter_node(state, "final_response")
        answer = state.answer
        if state.guardrail_violations:
            answer = "生成草稿未通过引用或生产安全校验，不能直接输出；该问题已转人工复核。"
        elif state.knowledge_gap is not None:
            answer = "当前论文库证据不足。该问题已记录为知识缺口。"
        elif state.need_human_review and not answer:
            answer = "该问题涉及高风险、证据冲突或工具异常，不能由模型给出最终结论，已转人工复核。"
        elif not answer:
            answer = "当前论文库证据不足。"

        output = {
            "query": state.query,
            "normalized_query": state.normalized_query,
            "intent": state.intent,
            "answer": answer,
            "evidence_list": state.generated_evidence or state.retrieved_evidence,
            "confidence": state.confidence,
            "need_human_review": state.need_human_review,
            "limitations": state.limitations,
            "errors": state.errors,
            "tool_calls": state.tool_calls,
            "step_count": state.step_count,
            "node_history": state.node_history,
            "knowledge_gap": state.knowledge_gap,
            "review_ticket": state.review_ticket,
        }
        state.answer = answer
        state.final_output = output
        return output

    def _enter_node(self, state: AgentState, node_name: str) -> bool:
        if state.step_count >= self.max_steps:
            message = f"max_steps_exceeded: {self.max_steps}"
            if message not in state.errors:
                state.errors.append(message)
            state.need_human_review = True
            return False
        state.step_count += 1
        state.node_history.append(node_name)
        return True

    def _call_tool(
        self,
        state: AgentState,
        tool_name: str,
        call: Callable[[], Any],
        arguments: dict[str, Any],
    ) -> Any | None:
        if state.tool_call_count >= self.max_tool_calls:
            state.errors.append(f"max_tool_calls_exceeded: {self.max_tool_calls}")
            state.need_human_review = True
            return None
        record: dict[str, Any] = {"tool": tool_name, "arguments": arguments}
        try:
            result = call()
            record["status"] = "success"
            state.tool_calls.append(record)
            return result
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            state.tool_calls.append(record)
            state.tool_error_count += 1
            state.errors.append(f"{tool_name}: {record['error']}")
            if state.tool_error_count > 2:
                state.need_human_review = True
            return None

    def _record_retrieval(self, state: AgentState, query: str, evidence: list[dict[str, Any]]) -> None:
        current = {
            "query": query,
            "result_count": len(evidence),
            "best_score": max((float(item.get("score", 0.0)) for item in evidence), default=0.0),
        }
        if state.retrieval_history:
            previous = state.retrieval_history[-1]
            same_query = previous["query"] == current["query"]
            improved = current["result_count"] > previous["result_count"] or current["best_score"] > previous["best_score"] + 1e-9
            state.retrieval_stalled = same_query and not improved
        state.retrieval_history.append(current)

    def _record_knowledge_gap(self, state: AgentState) -> None:
        result = self._call_tool(
            state,
            "knowledge_gap_tool",
            lambda: self.gap_tool(
                user_question=state.query,
                missing_information_type="no_relevant_paper" if not state.retrieved_evidence else "insufficient_evidence",
                attempted_queries=[item["query"] for item in state.retrieval_history],
                retrieved_evidence_ids=[str(item.get("metadata", {}).get("evidence_id", "")) for item in state.retrieved_evidence if item.get("metadata", {}).get("evidence_id")],
                reason_for_gap="当前检索为空或证据分数低于自动回答阈值。",
                suggested_next_actions=["add_papers", "add_synonyms", "build_eval_question"],
                priority="medium",
            ),
            {"query": state.query, "retrieval_attempts": len(state.retrieval_history)},
        )
        state.knowledge_gap = result or {"status": "record_failed"}

    def _enter_human_review(self, state: AgentState, trigger_reason: str, reason: str) -> None:
        state.need_human_review = True
        if state.review_ticket is not None:
            return
        case_hash = hashlib.sha1(state.query.encode("utf-8")).hexdigest()[:12]
        result = self._call_tool(
            state,
            "human_review_tool",
            lambda: self.review_tool(
                case_id=f"case_{case_hash}",
                trigger_reason=trigger_reason,
                user_question=state.query,
                agent_intermediate_result={
                    "intent": state.intent,
                    "reason": reason,
                    "evidence_count": len(state.retrieved_evidence),
                    "errors": state.errors,
                },
                evidence_ids=[str(item.get("metadata", {}).get("evidence_id", "")) for item in state.retrieved_evidence if item.get("metadata", {}).get("evidence_id")],
                risk_level=state.risk_level if state.risk_level in {"low", "medium", "high", "critical"} else "high",
                confidence_score=self._confidence_float(state.confidence),
                required_expert_role="process_engineer",
                review_questions=[reason],
            ),
            {"trigger_reason": trigger_reason, "risk_level": state.risk_level},
        )
        state.review_ticket = result or {"status": "creation_failed", "reason": reason}

    @staticmethod
    def _search_result_to_candidate(item: dict[str, Any]) -> dict[str, Any]:
        source = item.get("source_location") or {}
        return {
            "chunk_id": str(item.get("chunk_id", "")),
            "paper_id": str(item.get("paper_id", "")),
            "title": str(item.get("title", "")),
            "section_name": str(source.get("section") or ""),
            "chunk_type": str(item.get("chunk_type", "")),
            "score": float(item.get("relevance_score", 0.0)),
            "source": "agent_search",
            "text_preview": str(item.get("matched_text", ""))[:200],
            "metadata": {
                "evidence_id": item.get("evidence_id"),
                "year": item.get("year"),
                "page_start": source.get("page"),
                "table": source.get("table"),
                "figure": source.get("figure"),
                "citation": item.get("citation"),
                "evidence_quality": item.get("evidence_quality"),
                "conflict": item.get("conflict", False),
                "effect_direction": item.get("effect_direction"),
            },
            "conflict": item.get("conflict", False),
            "effect_direction": item.get("effect_direction"),
        }

    @staticmethod
    def _evidence_confidence(evidence: list[dict[str, Any]]) -> float:
        if not evidence:
            return 0.0
        scores = [float(item.get("rerank_score", item.get("score", 0.0))) for item in evidence[:3]]
        return sum(scores) / len(scores)

    @staticmethod
    def _confidence_float(confidence: str | float) -> float:
        if isinstance(confidence, (int, float)):
            return max(0.0, min(float(confidence), 1.0))
        return {"high": 0.85, "medium": 0.7, "low": 0.4}.get(str(confidence).lower(), 0.0)

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(f"Expected a dictionary-like result, got {type(value).__name__}")


def run_agent(query: str, **workflow_kwargs: Any) -> dict[str, Any]:
    return AgentWorkflow(**workflow_kwargs).run(query)
