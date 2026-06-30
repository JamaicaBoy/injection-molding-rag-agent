from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agent.confidence import ConfidenceAssessment, assess_confidence
from src.agent.guardrails import check_answer_guardrails
from src.agent.langgraph_state import AgentState
from src.agent.memory import AgentMemory
from src.agent.tools import human_review_tool, search_papers_tool
from src.rag.answer_generator import AnswerGenerator
from src.rag.citation_guard import check_citations
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.reranker import Reranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = PROJECT_ROOT / "data" / "logs" / "langgraph_trace.jsonl"
_TRACE_LOCK = threading.Lock()


def _default_retriever(query: str, top_k: int) -> dict[str, Any]:
    return search_papers_tool(
        query=query,
        search_type="hybrid",
        filters={},
        top_k=top_k,
        rerank=False,
        return_chunks=True,
        language="auto",
    )


@dataclass
class LangGraphDependencies:
    rewriter: Callable[[str], Any] = rewrite_query
    retriever: Any = _default_retriever
    reranker: Any = None
    answer_generator: Any = None
    citation_checker: Callable[[str, list[dict[str, Any]]], Any] = check_citations
    guardrail_checker: Callable[[str, list[dict[str, Any]]], Any] = check_answer_guardrails
    memory: Any = None
    review_tool: Callable[..., dict[str, Any]] = human_review_tool
    revision_generator: Callable[[str, str, list[dict[str, Any]]], str] | None = None
    retrieval_top_k: int = 10
    max_revisions: int = 1
    trace_path: Path | None = DEFAULT_TRACE_PATH

    def __post_init__(self) -> None:
        if self.reranker is None:
            self.reranker = Reranker(mode="rule")
        if self.answer_generator is None:
            self.answer_generator = AnswerGenerator(mode="ollama")
        if self.memory is None:
            self.memory = AgentMemory()
        if self.trace_path is not None:
            self.trace_path = Path(self.trace_path)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Expected dictionary-like output, got {type(value).__name__}")


def _score(data: dict[str, Any]) -> float:
    # A non-reranked search result intentionally contains rerank_score=None.
    # Skip None so the real relevance_score is retained for the rule reranker.
    for key in ("rerank_score", "relevance_score", "score"):
        value = data.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _normalize_document(item: Any) -> dict[str, Any]:
    if isinstance(item, Document):
        metadata = dict(item.metadata)
        data = {
            "chunk_id": metadata.get("chunk_id", ""),
            "paper_id": metadata.get("paper_id", ""),
            "title": metadata.get("title", ""),
            "section_name": metadata.get("section") or metadata.get("section_name") or "",
            "chunk_type": metadata.get("chunk_type", ""),
            "score": metadata.get("score", 0.0),
            "source": "langchain_document",
            "text_preview": item.page_content[:600],
            "metadata": metadata,
        }
        return data

    data = dict(item)
    source_location = data.get("source_location") or {}
    metadata = dict(data.get("metadata") or {})
    evidence_id = data.get("evidence_id") or metadata.get("evidence_id")
    if evidence_id:
        metadata["evidence_id"] = evidence_id
    return {
        "chunk_id": str(data.get("chunk_id", "")),
        "paper_id": str(data.get("paper_id", "")),
        "title": str(data.get("title", "")),
        "section_name": str(data.get("section_name") or source_location.get("section") or ""),
        "chunk_type": str(data.get("chunk_type", "")),
        "score": _score(data),
        "source": str(data.get("source", "agent_search")),
        "text_preview": str(data.get("text_preview") or data.get("matched_text") or "")[:600],
        "metadata": metadata,
        "conflict": bool(data.get("conflict", False)),
        "effect_direction": data.get("effect_direction"),
    }


def _invoke_retriever(retriever: Any, query: str, top_k: int) -> list[dict[str, Any]]:
    if hasattr(retriever, "invoke"):
        output = retriever.invoke(query)
    else:
        try:
            output = retriever(query, top_k)
        except TypeError:
            output = retriever(query)
    if isinstance(output, dict):
        output = output.get("results", [])
    return [_normalize_document(item) for item in list(output or [])]


def _assessment_update(assessment: ConfidenceAssessment) -> dict[str, Any]:
    return {
        "confidence": assessment.level,
        "confidence_reason": assessment.reason,
        "need_human_review": assessment.need_human_review,
        "human_review_reason": assessment.human_review_reason,
    }


def _resulting_list(state: AgentState, update: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return list(update[key] if key in update else state.get(key, []))


def _finalize_node(
    state: AgentState,
    node_name: str,
    update: dict[str, Any],
    deps: LangGraphDependencies,
    *,
    input_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    update["step_count"] = int(state.get("step_count", 0)) + 1
    update["node_history"] = [node_name]
    retrieved = _resulting_list(state, update, "retrieved_docs")
    reranked = _resulting_list(state, update, "reranked_docs")
    evidence = _resulting_list(state, update, "evidence_list")
    top_score = max((_score(item) for item in reranked), default=0.0)
    errors = list(update.get("errors", []))
    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": str(state.get("conversation_id", "")),
        "node_name": node_name,
        "input_summary": input_summary or {
            "intent": state.get("query_info", {}).get("intent", ""),
            "retrieved_count": len(state.get("retrieved_docs", [])),
            "reranked_count": len(state.get("reranked_docs", [])),
        },
        "output_summary": {
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "evidence_count": len(evidence),
            "citation_passed": update.get("citation_passed", state.get("citation_passed")),
            "human_review_reason": update.get(
                "human_review_reason", state.get("human_review_reason", "")
            ),
        },
        "retrieved_count": len(retrieved),
        "reranked_count": len(reranked),
        "top_score": round(top_score, 6),
        "confidence_before": state.get("confidence", "low"),
        "confidence_after": update.get("confidence", state.get("confidence", "low")),
        "confidence_reason": update.get(
            "confidence_reason", state.get("confidence_reason", "")
        ),
        "need_human_review": bool(
            update.get("need_human_review", state.get("need_human_review", False))
        ),
        "error": errors[-1] if errors else "",
    }
    update["trace"] = [trace_entry]
    if deps.trace_path is not None:
        deps.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_LOCK, deps.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
    return update


def query_rewrite_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    try:
        update["query_info"] = _as_dict(deps.rewriter(state["query"]))
    except Exception as exc:
        update.update(
            {
                "query_info": {},
                "need_human_review": True,
                "human_review_reason": "tool_error",
                "confidence": "low",
                "confidence_reason": "Query rewrite failed.",
                "errors": [f"query_rewrite_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "query_rewrite_node", update, deps, input_summary={"query_length": len(state["query"])})


def retrieve_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    query_info = state.get("query_info", {})
    query = str(query_info.get("normalized_query") or state["query"])
    try:
        update["retrieved_docs"] = _invoke_retriever(deps.retriever, query, deps.retrieval_top_k)
    except Exception as exc:
        update.update(
            {
                "retrieved_docs": [],
                "need_human_review": True,
                "human_review_reason": "tool_error",
                "confidence": "low",
                "confidence_reason": "Retrieval failed.",
                "errors": [f"retrieve_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(
        state,
        "retrieve_node",
        update,
        deps,
        input_summary={"normalized_query_length": len(query), "intent": query_info.get("intent", "")},
    )


def rerank_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    retrieved = list(state.get("retrieved_docs", []))
    if not retrieved:
        assessment = assess_confidence(state["query"], state.get("query_info", {}), [])
        update.update({"reranked_docs": [], **_assessment_update(assessment)})
        return _finalize_node(state, "rerank_node", update, deps)
    try:
        reranked = deps.reranker.rerank(
            state["query"], retrieved, top_n=min(deps.retrieval_top_k, len(retrieved))
        )
        assessment = assess_confidence(state["query"], state.get("query_info", {}), reranked)
        update.update({"reranked_docs": reranked, **_assessment_update(assessment)})
    except Exception as exc:
        assessment = assess_confidence(
            state["query"], state.get("query_info", {}), [], tool_error=True
        )
        update.update(
            {
                "reranked_docs": [],
                **_assessment_update(assessment),
                "errors": [f"rerank_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "rerank_node", update, deps)


def answer_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    reranked = list(state.get("reranked_docs", []))
    try:
        output = _as_dict(
            deps.answer_generator.generate(state["query"], state.get("query_info", {}), reranked)
        )
        answer = str(output.get("answer", ""))
        evidence = list(output.get("evidence_list") or reranked)
        update.update(
            {
                "draft_answer": answer,
                "evidence_list": evidence,
                "limitations": list(output.get("limitations", [])),
                "llm_mode": str(output.get("llm_mode", "ollama")),
                "llm_model": str(output.get("llm_model", "")),
                "llm_fallback_reason": str(output.get("llm_fallback_reason", "")),
                "messages": [AIMessage(content=answer)] if answer else [],
            }
        )
        if update["llm_mode"] == "mock":
            update.update(
                {
                    "confidence": "low",
                    "confidence_reason": "Local LLM generation failed and Mock fallback was used.",
                    "need_human_review": True,
                    "human_review_reason": "tool_error",
                }
            )
    except Exception as exc:
        assessment = assess_confidence(
            state["query"], state.get("query_info", {}), reranked, tool_error=True
        )
        update.update(
            {
                "draft_answer": "",
                "evidence_list": reranked,
                **_assessment_update(assessment),
                "errors": [f"answer_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "answer_node", update, deps)


def citation_guard_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    evidence = list(state.get("evidence_list", []))
    try:
        result = _as_dict(deps.citation_checker(state.get("draft_answer", ""), evidence))
        passed = bool(result.get("passed", not result.get("high_risk", False)))
        update.update(
            {
                "citation_passed": passed,
                "citations": [str(item) for item in result.get("citations", [])],
            }
        )
        if not passed:
            update.update(
                {
                    "human_review_reason": "citation_failed",
                    "confidence": "low",
                    "confidence_reason": "The answer did not pass citation validation.",
                    "errors": [f"citation_guard: {issue}" for issue in result.get("issues", [])],
                }
            )
    except Exception as exc:
        update.update(
            {
                "citation_passed": False,
                "need_human_review": True,
                "human_review_reason": "tool_error",
                "confidence": "low",
                "confidence_reason": "Citation validation failed to run.",
                "errors": [f"citation_guard_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "citation_guard_node", update, deps)


def revise_answer_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    if deps.revision_generator is None:
        update.update({"need_human_review": True, "human_review_reason": "citation_failed"})
        return _finalize_node(state, "revise_answer_node", update, deps)
    try:
        revised = deps.revision_generator(
            state["query"], state.get("draft_answer", ""), list(state.get("evidence_list", []))
        )
        update.update(
            {
                "draft_answer": str(revised),
                "revision_count": int(state.get("revision_count", 0)) + 1,
                "human_review_reason": "",
            }
        )
    except Exception as exc:
        update.update(
            {
                "need_human_review": True,
                "human_review_reason": "tool_error",
                "errors": [f"revise_answer_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "revise_answer_node", update, deps)


def risk_check_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    evidence = list(state.get("evidence_list", []))
    try:
        guard = _as_dict(deps.guardrail_checker(state.get("draft_answer", ""), evidence))
        assessment = assess_confidence(
            state["query"],
            state.get("query_info", {}),
            evidence,
            citation_passed=bool(state.get("citation_passed", False)),
            tool_error=state.get("llm_mode") == "mock",
        )
        update.update(_assessment_update(assessment))
        violations = list(guard.get("violations", []))
        guard_passed = bool(guard.get("passed", not guard.get("need_human_review", False)))
        if not guard_passed:
            update.update(
                {
                    "risk_passed": False,
                    "need_human_review": True,
                    "human_review_reason": "high_risk_action",
                    "errors": [f"risk_guard: {item}" for item in violations],
                }
            )
        else:
            update["risk_passed"] = not assessment.need_human_review
            update["final_answer"] = state.get("draft_answer", "")
    except Exception as exc:
        assessment = assess_confidence(
            state["query"], state.get("query_info", {}), evidence, tool_error=True
        )
        update.update(
            {
                "risk_passed": False,
                **_assessment_update(assessment),
                "errors": [f"risk_check_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "risk_check_node", update, deps)


def human_review_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    reason = str(state.get("human_review_reason") or "evidence_insufficient")
    trigger = {
        "evidence_insufficient": "low_confidence",
        "high_risk_action": "high_risk",
        "citation_failed": "safety_risk",
        "tool_error": "other",
    }.get(reason, "other")
    risk_level = str(state.get("query_info", {}).get("risk_level", "medium")).lower()
    if risk_level not in {"low", "medium", "high", "critical"}:
        risk_level = "medium"
    evidence = list(state.get("evidence_list") or state.get("reranked_docs", []))
    evidence_ids = [
        str(item.get("evidence_id") or item.get("metadata", {}).get("evidence_id"))
        for item in evidence
        if item.get("evidence_id") or item.get("metadata", {}).get("evidence_id")
    ]
    try:
        update["review_ticket"] = deps.review_tool(
            case_id=f"case_{hashlib.sha1(state['query'].encode('utf-8')).hexdigest()[:12]}",
            trigger_reason=trigger,
            user_question=state["query"],
            agent_intermediate_result={"reason": reason, "evidence_count": len(evidence)},
            evidence_ids=evidence_ids,
            risk_level=risk_level,
            confidence_score=_confidence_float(state.get("confidence", "low")),
            required_expert_role="process_engineer",
            review_questions=[reason],
        )
    except Exception as exc:
        update["errors"] = [f"human_review_node: {type(exc).__name__}: {exc}"]
    fallback = {
        "evidence_insufficient": "当前论文库证据不足，已转人工复核。",
        "citation_failed": "答案引用校验未通过，已转人工复核。",
        "tool_error": "工作流工具执行失败，已转人工复核。",
        "high_risk_action": "该问题涉及直接生产参数或设备操作，答案只能作为候选方向，需人工复核。",
    }.get(reason, "该问题需要人工复核。")
    update.update(
        {
            "need_human_review": True,
            "human_review_reason": reason,
            "final_answer": fallback,
        }
    )
    return _finalize_node(state, "human_review_node", update, deps)


def memory_update_node(state: AgentState, deps: LangGraphDependencies) -> dict[str, Any]:
    update: dict[str, Any] = {}
    evidence = list(state.get("evidence_list") or state.get("reranked_docs", []))
    try:
        deps.memory.record_query(
            query=state["query"],
            intent=str(state.get("query_info", {}).get("intent", "general_qa")),
            evidence=evidence,
            answer_confidence=state.get("confidence", "low"),
            need_human_review=bool(state.get("need_human_review", False)),
            user_feedback=None,
        )
        update["memory_updated"] = True
    except Exception as exc:
        update.update(
            {
                "memory_updated": False,
                "errors": [f"memory_update_node: {type(exc).__name__}: {exc}"],
            }
        )
    return _finalize_node(state, "memory_update_node", update, deps)


def _confidence_float(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    return {"high": 0.85, "medium": 0.68, "low": 0.4}.get(str(value).lower(), 0.0)
