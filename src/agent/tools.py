from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import merge_results
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.reranker import Reranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_TICKETS = PROJECT_ROOT / "data" / "manual_review" / "agent_review_tickets.jsonl"
DEFAULT_KNOWLEDGE_GAPS = PROJECT_ROOT / "data" / "manual_review" / "knowledge_gaps.jsonl"

SEARCH_TYPES = {"keyword", "semantic", "hybrid"}
RISK_LEVELS = {"low", "medium", "high", "critical"}
EXTRACT_SCHEMAS = {"defect", "parameter_effect", "method", "experiment", "metric", "general_evidence"}
ANALYSIS_TYPES = {"trend", "mechanism", "sensitivity", "optimization_hint", "all"}
COMPARISON_DIMENSIONS = {"task", "input_data", "output", "model_structure", "optimization_objective", "dataset", "metrics", "advantages", "limitations", "engineering_feasibility", "interpretability"}
TRIGGER_REASONS = {"low_confidence", "high_risk", "evidence_conflict", "production_decision", "missing_context", "user_requested_final_decision", "safety_risk", "other"}
EXPERT_ROLES = {"process_engineer", "quality_engineer", "mold_engineer", "material_engineer", "algorithm_engineer", "manager", "unknown"}
MISSING_INFORMATION_TYPES = {"no_relevant_paper", "insufficient_evidence", "missing_material_context", "missing_parameter_range", "missing_defect_case", "missing_method_detail", "conflicting_evidence", "other"}
NEXT_ACTIONS = {"add_papers", "ask_expert", "collect_factory_cases", "add_metadata", "improve_chunking", "add_synonyms", "build_eval_question", "other"}
METHOD_TERMS = ("machine learning", "deep learning", "random forest", "cnn", "rag", "llm", "doe", "ga", "pso", "moldflow", "simulation")

SearchFunction = Callable[..., dict[str, Any]]


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _evidence_quality(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _filter_terms(value: str) -> list[str]:
    rewrite = rewrite_query(value)
    return list(dict.fromkeys([value, *rewrite.must_have_terms, *rewrite.expanded_terms]))


def _matches_filters(result: dict[str, Any], filters: dict[str, Any]) -> bool:
    metadata = result.get("metadata") or {}
    blob = " ".join(
        (
            str(result.get("title", "")),
            str(result.get("text_preview", "")),
            json.dumps(metadata, ensure_ascii=False, default=str),
        )
    ).lower().replace("_", " ")

    paper_ids = filters.get("paper_ids") or []
    if paper_ids and str(result.get("paper_id", "")) not in {str(item) for item in paper_ids}:
        return False

    year_range = filters.get("year_range") or []
    if year_range:
        year = _safe_int(metadata.get("year"))
        start = _safe_int(year_range[0]) if len(year_range) > 0 else None
        end = _safe_int(year_range[1]) if len(year_range) > 1 else None
        if (start is not None or end is not None) and (
            year is None or (start is not None and year < start) or (end is not None and year > end)
        ):
            return False

    for field in ("material", "defect_type", "process_parameter", "method_type"):
        value = filters.get(field)
        if value:
            terms = [term.lower().replace("_", " ") for term in _filter_terms(str(value))]
            if not any(term in blob for term in terms):
                return False
    return True


def _matched_keywords(result: dict[str, Any], query_terms: list[str]) -> list[str]:
    text = f"{result.get('title', '')} {result.get('text_preview', '')}".lower().replace("_", " ")
    return [term for term in query_terms if term.lower().replace("_", " ") in text]


def _source_location(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    preview = str(result.get("text_preview", ""))
    table_match = re.search(r"\b(?:table|表)\s*([A-Za-z0-9.-]+)", preview, flags=re.IGNORECASE)
    figure_match = re.search(r"\b(?:figure|fig\.?|图)\s*([A-Za-z0-9.-]+)", preview, flags=re.IGNORECASE)
    return {
        "page": _safe_int(metadata.get("page_start")),
        "section": result.get("section_name") or metadata.get("section") or None,
        "table": table_match.group(0) if table_match else None,
        "figure": figure_match.group(0) if figure_match else None,
    }


def search_papers_tool(
    query: str,
    search_type: str = "hybrid",
    filters: dict[str, Any] | None = None,
    top_k: int = 10,
    rerank: bool = True,
    return_chunks: bool = True,
    language: str = "auto",
    *,
    _bm25: Any | None = None,
    _dense: Any | None = None,
    _reranker: Any | None = None,
) -> dict[str, Any]:
    """Search the paper chunk store and return auditable evidence records."""
    if search_type not in SEARCH_TYPES:
        raise ValueError(f"Unsupported search_type: {search_type}")
    if language not in {"zh", "en", "auto"}:
        raise ValueError(f"Unsupported language: {language}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    filters = filters or {}
    rewritten = rewrite_query(query)
    retrieval_query = rewritten.normalized_query
    candidate_k = max(top_k * 5 if filters else top_k * 2, top_k)

    if search_type == "keyword":
        bm25 = _bm25 or BM25Retriever()
        candidates = bm25.search(retrieval_query, top_k=candidate_k)
    elif search_type == "semantic":
        dense = _dense or DenseRetriever()
        candidates = dense.search(retrieval_query, top_k=candidate_k)
    else:
        bm25 = _bm25 or BM25Retriever()
        dense = _dense or DenseRetriever()
        candidates = merge_results(
            bm25.search(retrieval_query, top_k=candidate_k),
            dense.search(retrieval_query, top_k=candidate_k),
            top_k=candidate_k,
        )

    candidates = [candidate for candidate in candidates if _matches_filters(candidate, filters)]
    if rerank and candidates:
        reranker = _reranker or Reranker(mode="rule")
        candidates = reranker.rerank(query, candidates, top_n=top_k)
    else:
        candidates = candidates[:top_k]

    query_terms = list(dict.fromkeys([*rewritten.must_have_terms, *rewritten.expanded_terms]))
    results: list[dict[str, Any]] = []
    for index, result in enumerate(candidates, start=1):
        metadata = result.get("metadata") or {}
        score = _clamp(result.get("rerank_score", result.get("score", 0.0)))
        year = _safe_int(metadata.get("year"))
        authors = metadata.get("authors") or []
        if isinstance(authors, str):
            authors = [item.strip() for item in re.split(r"[;,]", authors) if item.strip()]
        title = str(result.get("title", ""))
        location = _source_location(result)
        results.append(
            {
                "evidence_id": f"E{index}",
                "paper_id": str(result.get("paper_id", "")),
                "title": title,
                "authors": authors,
                "year": year,
                "chunk_id": str(result.get("chunk_id", "")),
                "source_location": location,
                "matched_text": str(result.get("text_preview", ""))[:200] if return_chunks else "",
                "matched_keywords": _matched_keywords(result, query_terms),
                "relevance_score": score,
                "rerank_score": score if rerank else None,
                "evidence_quality": _evidence_quality(score),
                "citation": f"{title} ({year or 'n.d.'}), {location['section'] or 'unknown section'}, {result.get('paper_id', '')}",
            }
        )

    overall_confidence = (
        sum(item["relevance_score"] for item in results[:3]) / min(len(results), 3) if results else 0.0
    )
    warnings: list[str] = []
    if not results:
        warnings.append("no_relevant_evidence")
    elif overall_confidence < 0.65:
        warnings.append("low_overall_confidence")
    if results and not any(item["authors"] for item in results):
        warnings.append("authors_unavailable_in_current_chunk_metadata")
    return {
        "query": query,
        "search_type": search_type,
        "results": results,
        "overall_confidence": _clamp(overall_confidence),
        "warnings": warnings,
    }


def defect_diagnosis_tool(
    defect_description: str,
    defect_type: str | None = None,
    product_context: dict[str, Any] | None = None,
    process_context: dict[str, Any] | None = None,
    retrieval_top_k: int = 10,
    risk_level: str = "medium",
    *,
    _search_fn: SearchFunction | None = None,
) -> dict[str, Any]:
    """Build a candidate defect diagnosis from paper evidence, never a final production decision."""
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Unsupported risk_level: {risk_level}")
    product_context = product_context or {}
    process_context = process_context or {}
    rewrite = rewrite_query(f"{defect_type or ''} {defect_description}")
    standardized = rewrite.defect_type[0] if rewrite.defect_type else (defect_type or "unknown")
    material = product_context.get("material")
    process_summary = " ".join(f"{key}={value}" for key, value in process_context.items() if value not in (None, ""))
    search_fn = _search_fn or search_papers_tool
    search = search_fn(
        query=f"{standardized} possible causes related parameters {defect_description} {process_summary}",
        search_type="hybrid",
        filters={"material": material, "defect_type": standardized},
        top_k=retrieval_top_k,
        rerank=True,
    )

    possible_causes: list[dict[str, Any]] = []
    for index, evidence in enumerate(search.get("results", [])[:5], start=1):
        entity_rewrite = rewrite_query(evidence.get("matched_text", "") or defect_description)
        related_parameters = entity_rewrite.parameters
        possible_causes.append(
            {
                "cause_id": f"C{index}",
                "cause_category": "process_parameter" if related_parameters else "unknown",
                "cause_description": evidence.get("matched_text", ""),
                "related_parameters": related_parameters,
                "mechanism_explanation": evidence.get("matched_text", ""),
                "supporting_evidence_ids": [evidence["evidence_id"]],
                "confidence_score": evidence.get("relevance_score", 0.0),
                "applicability_conditions": [f"material={material}"] if material else [],
                "limitations": ["Candidate cause inferred from a retrieved paper snippet;现场条件仍需核验。"],
            }
        )

    suggested_checks: list[dict[str, Any]] = []
    if not material:
        suggested_checks.append({"check_item": "补充材料牌号", "reason": "缺陷机制可能随材料变化", "priority": "high", "requires_human": True})
    if not product_context.get("defect_location"):
        suggested_checks.append({"check_item": "确认缺陷位置与形态", "reason": "用于区分材料、模具与参数因素", "priority": "high", "requires_human": True})
    for parameter in list(dict.fromkeys(item for cause in possible_causes for item in cause["related_parameters"]))[:3]:
        suggested_checks.append({"check_item": f"核对 {parameter} 的当前记录", "reason": "论文证据提示其可能相关", "priority": "medium", "requires_human": True})

    paper_count = len({item.get("paper_id") for item in search.get("results", []) if item.get("paper_id")})
    confidence = max((item["confidence_score"] for item in possible_causes), default=0.0)
    knowledge_gap = not possible_causes
    need_human_review = risk_level in {"high", "critical"} or confidence < 0.65 or paper_count < 2
    return {
        "standardized_defect_type": standardized,
        "possible_causes": possible_causes,
        "suggested_checks": suggested_checks,
        "not_final_decision_notice": "以上仅为论文证据支持的候选原因和排查方向，不能作为最终诊断或直接生产调参指令。",
        "need_human_review": need_human_review,
        "knowledge_gap": knowledge_gap,
    }


def _effect_direction(texts: list[str]) -> str:
    text = " ".join(texts).lower()
    increase = r"(?:increas(?:e|ed|ing)|提高|增大)"
    positive = bool(re.search(rf"{increase}.{{0,60}}(?:reduc(?:e|ed|ing)|decreas(?:e|ed|ing)|降低|减少).{{0,30}}(?:defect|warpage|shrinkage|缺陷|翘曲|缩水|缩痕)", text))
    negative = bool(re.search(rf"{increase}.{{0,60}}(?:increas(?:e|ed|ing)|worsen(?:ed|ing)?|增多|增加|恶化).{{0,30}}(?:defect|warpage|shrinkage|缺陷|翘曲|缩水|缩痕)", text))
    if positive and negative:
        return "condition_dependent"
    if positive:
        return "increase_positive"
    if negative:
        return "increase_negative"
    return "unclear"


def parameter_effect_tool(
    parameter_name: str,
    target_quality_or_defect: str | None = None,
    material: str | None = None,
    process_stage: str | None = None,
    analysis_type: str = "all",
    evidence_scope: dict[str, Any] | None = None,
    risk_level: str = "medium",
    *,
    _search_fn: SearchFunction | None = None,
) -> dict[str, Any]:
    """Summarize a parameter effect as an evidence-backed trend, not a set point."""
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Unsupported risk_level: {risk_level}")
    if analysis_type not in ANALYSIS_TYPES:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")
    evidence_scope = evidence_scope or {}
    top_k = int(evidence_scope.get("top_k", 10))
    search_fn = _search_fn or search_papers_tool
    search = search_fn(
        query=f"{parameter_name} effect on {target_quality_or_defect or 'quality'} {material or ''}",
        search_type="hybrid",
        filters={
            "material": material,
            "process_parameter": parameter_name,
            "defect_type": target_quality_or_defect,
            "paper_ids": evidence_scope.get("paper_ids", []),
        },
        top_k=top_k,
        rerank=True,
    )
    min_score = float(evidence_scope.get("min_relevance_score", 0.5))
    evidence = [item for item in search.get("results", []) if item.get("relevance_score", 0.0) >= min_score]
    direction = _effect_direction([item.get("matched_text", "") for item in evidence])
    mechanisms = [
        {
            "mechanism": item.get("matched_text", ""),
            "process_stage": process_stage,
            "supporting_evidence_ids": [item["evidence_id"]],
            "confidence_score": item.get("relevance_score", 0.0),
        }
        for item in evidence[:5]
    ]
    interactions: list[dict[str, Any]] = []
    target_parameters = set(rewrite_query(parameter_name).parameters)
    for item in evidence:
        related = rewrite_query(item.get("matched_text", "")).parameters
        for parameter in related:
            if parameter not in target_parameters and parameter not in {entry["related_parameter"] for entry in interactions}:
                interactions.append(
                    {
                        "related_parameter": parameter,
                        "interaction_description": item.get("matched_text", ""),
                        "supporting_evidence_ids": [item["evidence_id"]],
                    }
                )

    paper_count = len({item.get("paper_id") for item in evidence if item.get("paper_id")})
    confidence = sum(item.get("relevance_score", 0.0) for item in evidence) / len(evidence) if evidence else 0.0
    high_risk = risk_level in {"high", "critical"}
    can_auto_answer = bool(evidence) and confidence >= 0.65 and paper_count >= 2 and not high_risk and direction != "unclear"
    unsafe = []
    if high_risk:
        unsafe.append("真实生产参数值和调整幅度未经现场工程师验证。")
    if not evidence:
        unsafe.append("当前论文库证据不足。")
    return {
        "parameter_name": parameter_name,
        "target_quality_or_defect": target_quality_or_defect,
        "effect_summary": " ".join(item.get("matched_text", "") for item in evidence[:3]) or "当前论文库证据不足。",
        "effect_direction": direction,
        "mechanisms": mechanisms,
        "parameter_interactions": interactions,
        "applicability_conditions": [value for value in (f"material={material}" if material else "", f"process_stage={process_stage}" if process_stage else "") if value],
        "unsafe_or_unverified_claims": unsafe,
        "can_auto_answer": can_auto_answer,
        "need_human_review": high_risk or not can_auto_answer,
    }


def method_compare_tool(
    methods: list[dict[str, Any]],
    comparison_dimensions: list[str] | None = None,
    application_context: dict[str, Any] | None = None,
    retrieval_top_k: int = 10,
    output_format: str = "both",
    *,
    _search_fn: SearchFunction | None = None,
) -> dict[str, Any]:
    """Build a qualitative, evidence-linked comparison without making a deployment decision."""
    if not methods:
        raise ValueError("methods must not be empty")
    if output_format not in {"table", "narrative", "both"}:
        raise ValueError(f"Unsupported output_format: {output_format}")
    invalid_dimensions = set(comparison_dimensions or []) - COMPARISON_DIMENSIONS
    if invalid_dimensions:
        raise ValueError(f"Unsupported comparison dimensions: {sorted(invalid_dimensions)}")
    application_context = application_context or {}
    search_fn = _search_fn or search_papers_tool
    rows: list[dict[str, Any]] = []
    conflicts: list[str] = []
    for method in methods:
        name = str(method.get("method_name", "")).strip()
        if not name:
            continue
        search = search_fn(
            query=f"{name} injection molding method task advantages limitations",
            search_type="hybrid",
            filters={"method_type": name, "paper_ids": [method["paper_id"]] if method.get("paper_id") else []},
            top_k=retrieval_top_k,
            rerank=True,
        )
        evidence = search.get("results", [])
        if not evidence:
            conflicts.append(f"{name}: no comparable evidence")
        confidence = max((item.get("relevance_score", 0.0) for item in evidence), default=0.0)
        rows.append(
            {
                "method_name": name,
                "task_type": application_context.get("task_type", "unknown"),
                "input_data": list(application_context.get("available_data", [])),
                "output": [application_context["task_type"]] if application_context.get("task_type") not in (None, "unknown") else [],
                "core_idea": evidence[0].get("matched_text", "") if evidence else "当前论文库证据不足。",
                "advantages": [],
                "limitations": [] if evidence else ["No matching paper evidence."],
                "required_data_volume": "unclear",
                "engineering_feasibility": "unclear",
                "interpretability": "unclear",
                "supporting_evidence_ids": [item["evidence_id"] for item in evidence[:5]],
                "confidence_score": confidence,
            }
        )

    ranked_rows = sorted(rows, key=lambda item: item["confidence_score"], reverse=True)
    best = ranked_rows[0] if ranked_rows else None
    unique_best = len(ranked_rows) == 1 or ranked_rows[0]["confidence_score"] - ranked_rows[1]["confidence_score"] >= 0.05
    deployment = str(application_context.get("deployment_requirement", "")).strip()
    best_fit = best["method_name"] if best and unique_best and best["confidence_score"] >= 0.8 and not deployment else None
    need_human = bool(deployment) or any(row["confidence_score"] < 0.65 for row in rows)
    return {
        "comparison_table": rows,
        "recommendation_for_context": {
            "best_fit_method": best_fit,
            "reason": "按当前论文证据覆盖度排序；不同数据集和指标不可直接横向比较。" if best else "当前论文库证据不足。",
            "not_decision_notice": "该对比仅用于论文调研，不能替代企业技术选型、采购或上线决策。",
        },
        "evidence_conflicts": conflicts,
        "need_human_review": need_human,
    }


def _evidence_type(extract_schema: str, section: str) -> str:
    section_lower = section.lower()
    if "conclusion" in section_lower:
        return "conclusion"
    if "result" in section_lower or "experiment" in section_lower:
        return "experiment_result"
    if extract_schema == "parameter_effect":
        return "mechanism"
    if extract_schema == "method":
        return "method_description"
    if extract_schema == "metric":
        return "metric"
    if extract_schema == "experiment":
        return "dataset"
    return "mechanism"


def evidence_extract_tool(
    raw_results: list[dict[str, Any]],
    extract_schema: str = "general_evidence",
    target_fields: list[str] | None = None,
    deduplicate: bool = True,
    normalize_terms: bool = True,
    language: str = "zh",
) -> dict[str, Any]:
    """Convert retrieved snippets into a compact, traceable evidence table."""
    if extract_schema not in EXTRACT_SCHEMAS:
        raise ValueError(f"Unsupported extract_schema: {extract_schema}")
    if language not in {"zh", "en", "bilingual"}:
        raise ValueError(f"Unsupported language: {language}")

    evidence_table: list[dict[str, Any]] = []
    merged_items: list[str] = []
    seen: dict[str, str] = {}
    for index, raw in enumerate(raw_results, start=1):
        text = str(raw.get("text") or raw.get("matched_text") or raw.get("text_preview") or "").strip()
        if not text:
            continue
        claim = re.split(r"(?<=[。！？.!?])\s+", text, maxsplit=1)[0][:240]
        key = re.sub(r"\W+", "", claim.lower())
        evidence_id = str(raw.get("evidence_id") or f"E{index}")
        if deduplicate and key in seen:
            merged_items.append(f"{evidence_id}->{seen[key]}")
            continue
        seen[key] = evidence_id

        entities = {"materials": [], "defects": [], "parameters": [], "quality_metrics": [], "methods": []}
        if normalize_terms:
            rewrite = rewrite_query(text)
            entities = {
                "materials": rewrite.material,
                "defects": rewrite.defect_type,
                "parameters": rewrite.parameters,
                "quality_metrics": rewrite.quality_metric,
                "methods": [method for method in METHOD_TERMS if method in text.lower()],
            }
        source = raw.get("source_location") or {
            "page": (raw.get("metadata") or {}).get("page_start"),
            "section": raw.get("section_name"),
            "table": None,
            "figure": None,
        }
        score = _clamp(raw.get("rerank_score", raw.get("relevance_score", raw.get("confidence_score", 0.6))))
        evidence_table.append(
            {
                "evidence_id": evidence_id,
                "paper_id": str(raw.get("paper_id", "")),
                "claim": claim,
                "evidence_text": text[:600],
                "evidence_type": _evidence_type(extract_schema, str(source.get("section") or "")),
                "entities": entities,
                "source_location": {
                    "page": _safe_int(source.get("page")),
                    "section": source.get("section"),
                    "table": source.get("table"),
                    "figure": source.get("figure"),
                },
                "applicability_conditions": [],
                "limitations": [] if score >= 0.65 else ["Low-confidence source snippet; manual validation recommended."],
                "confidence_score": score,
            }
        )

    low_confidence = [item["evidence_id"] for item in evidence_table if item["confidence_score"] < 0.65]
    return {
        "evidence_table": evidence_table,
        "deduplication_report": {
            "original_count": len(raw_results),
            "final_count": len(evidence_table),
            "merged_items": merged_items,
        },
        "low_confidence_items": low_confidence,
        "need_human_review": bool(evidence_table) and len(low_confidence) / len(evidence_table) >= 0.5,
    }


def human_review_tool(
    case_id: str,
    trigger_reason: str,
    user_question: str,
    agent_intermediate_result: dict[str, Any],
    evidence_ids: list[str],
    risk_level: str,
    confidence_score: float,
    required_expert_role: str = "unknown",
    review_questions: list[str] | None = None,
    deadline: str | None = None,
    *,
    _ticket_store: Path = DEFAULT_REVIEW_TICKETS,
) -> dict[str, Any]:
    """Create a local, auditable pending ticket for mandatory expert review."""
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"Unsupported risk_level: {risk_level}")
    if trigger_reason not in TRIGGER_REASONS:
        raise ValueError(f"Unsupported trigger_reason: {trigger_reason}")
    if required_expert_role not in EXPERT_ROLES:
        raise ValueError(f"Unsupported required_expert_role: {required_expert_role}")
    created_at = _now()
    ticket_id = _stable_id("review", f"{case_id}|{user_question}|{created_at}")
    additional = ["请补充材料、设备、模具、产品结构和当前工艺参数。"] if trigger_reason == "missing_context" else []
    output = {
        "review_ticket_id": ticket_id,
        "status": "pending",
        "assigned_role": required_expert_role,
        "review_summary": None,
        "expert_comments": [],
        "approved_actions": [],
        "rejected_actions": [],
        "additional_information_needed": additional,
        "final_decision_owner": required_expert_role,
        "audit_log": {
            "created_at": created_at,
            "trigger_reason": trigger_reason,
            "risk_level": risk_level,
            "evidence_ids": evidence_ids,
        },
    }
    _append_jsonl(
        Path(_ticket_store),
        {
            **output,
            "case_id": case_id,
            "user_question": user_question,
            "confidence_score": _clamp(confidence_score),
            "review_questions": review_questions or [],
            "deadline": deadline,
            "agent_intermediate_result": agent_intermediate_result,
        },
    )
    return output


def knowledge_gap_tool(
    user_question: str,
    missing_information_type: str,
    attempted_queries: list[str],
    retrieved_evidence_ids: list[str],
    reason_for_gap: str,
    suggested_next_actions: list[str],
    priority: str = "medium",
    gap_id: str | None = None,
    *,
    _gap_store: Path = DEFAULT_KNOWLEDGE_GAPS,
) -> dict[str, Any]:
    """Record a deduplicated local knowledge gap instead of inventing an answer."""
    if priority not in {"high", "medium", "low"}:
        raise ValueError(f"Unsupported priority: {priority}")
    if missing_information_type not in MISSING_INFORMATION_TYPES:
        raise ValueError(f"Unsupported missing_information_type: {missing_information_type}")
    invalid_actions = set(suggested_next_actions) - NEXT_ACTIONS
    if invalid_actions:
        raise ValueError(f"Unsupported suggested_next_actions: {sorted(invalid_actions)}")
    normalized_question = re.sub(r"\s+", "", user_question).lower()
    gap_key = f"{missing_information_type}|{normalized_question}"
    existing = _read_jsonl(Path(_gap_store))
    for record in existing:
        if record.get("_gap_key") == gap_key:
            duplicate = {key: value for key, value in record.items() if not key.startswith("_")}
            duplicate["status"] = "duplicate"
            return duplicate

    owner_map = {
        "missing_material_context": "material_engineer",
        "missing_defect_case": "quality_engineer",
        "missing_parameter_range": "process_engineer",
        "missing_method_detail": "algorithm_engineer",
    }
    created_at = _now()
    output = {
        "gap_id": gap_id or _stable_id("gap", gap_key),
        "status": "recorded",
        "gap_summary": reason_for_gap,
        "priority": priority,
        "recommended_actions": suggested_next_actions,
        "linked_questions": [user_question],
        "linked_evidence_ids": retrieved_evidence_ids,
        "created_at": created_at,
        "owner_role": owner_map.get(missing_information_type, "knowledge_engineer"),
    }
    _append_jsonl(Path(_gap_store), {**output, "_gap_key": gap_key, "attempted_queries": attempted_queries})
    return output
