from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass
from typing import Any


HIGH_RISK_INTENTS = {"parameter_recommendation", "production_instruction", "equipment_operation"}


@dataclass(frozen=True)
class ConfidenceAssessment:
    level: str
    score: float
    reason: str
    evidence_count: int
    unique_paper_count: int
    top_score: float
    relative_score_strength: float
    term_coverage: float
    citation_passed: bool | None
    need_human_review: bool
    human_review_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_score(item: dict[str, Any]) -> float:
    for key in ("rerank_score", "score", "relevance_score"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _normalized_terms(query: str, query_info: dict[str, Any]) -> list[str]:
    configured = [
        *query_info.get("must_have_terms", []),
        *query_info.get("expanded_terms", []),
    ]
    terms: list[str] = []
    for value in configured:
        normalized = str(value).lower().replace("_", " ").strip()
        if normalized and normalized not in terms:
            terms.append(normalized)
    if terms:
        return terms
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z-]+|[\u4e00-\u9fff]{2,}", query)]


def _term_coverage(query: str, query_info: dict[str, Any], evidence: list[dict[str, Any]]) -> float:
    terms = _normalized_terms(query, query_info)
    if not terms:
        return 1.0 if evidence else 0.0
    haystack = " ".join(
        f"{item.get('title', '')} {item.get('section_name', '')} {item.get('text_preview', '')}"
        for item in evidence[:10]
    ).lower().replace("_", " ")
    return sum(term in haystack for term in terms) / len(terms)


def _relative_strength(scores: list[float]) -> float:
    positive = [score for score in scores if score > 0]
    if not positive:
        return 0.0
    if len(positive) == 1:
        return 0.5
    top = max(positive)
    median = statistics.median(positive)
    spread = max(0.0, min((top - median) / max(abs(top), 1e-9), 1.0))
    positive_ratio = len(positive) / len(scores)
    return round(0.5 * positive_ratio + 0.5 * spread, 4)


def assess_confidence(
    query: str,
    query_info: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    citation_passed: bool | None = None,
    tool_error: bool = False,
) -> ConfidenceAssessment:
    evidence_count = len(evidence)
    unique_papers = len({str(item.get("paper_id", "")) for item in evidence if item.get("paper_id")})
    scores = [_candidate_score(item) for item in evidence]
    top_score = max(scores, default=0.0)
    relative_strength = _relative_strength(scores)
    coverage = _term_coverage(query, query_info, evidence)
    risk_level = str(query_info.get("risk_level", "low")).lower()
    intent = str(query_info.get("intent", "general_qa")).lower()
    high_risk = risk_level in {"high", "critical"} or intent in HIGH_RISK_INTENTS

    if tool_error:
        return ConfidenceAssessment(
            "low", 0.0, "A retrieval or generation tool failed.", evidence_count,
            unique_papers, top_score, relative_strength, coverage, citation_passed, True, "tool_error"
        )
    if not evidence:
        return ConfidenceAssessment(
            "low", 0.0, "No evidence was retrieved.", 0, 0, 0.0, 0.0, 0.0,
            citation_passed, True, "evidence_insufficient"
        )
    if citation_passed is False:
        return ConfidenceAssessment(
            "low", 0.25, "The answer did not pass citation validation.", evidence_count,
            unique_papers, top_score, relative_strength, coverage, False, True, "citation_failed"
        )

    relevant = coverage > 0 or any(score > 0 for score in scores)
    enough = evidence_count >= 3 and relevant
    broad = evidence_count >= 5 and unique_papers >= 3

    if not enough:
        return ConfidenceAssessment(
            "low", 0.4, "Fewer than three relevant evidence items were available.", evidence_count,
            unique_papers, top_score, relative_strength, coverage, citation_passed, True,
            "evidence_insufficient"
        )

    level = "high" if citation_passed is True and broad and relative_strength >= 0.5 else "medium"
    score = 0.85 if level == "high" else 0.68
    reason = (
        "At least five cited results from multiple papers support the answer."
        if level == "high"
        else "At least three relevant evidence items support the answer."
    )
    if high_risk:
        return ConfidenceAssessment(
            level, score, reason + " The query still requests a high-risk production action.",
            evidence_count, unique_papers, top_score, relative_strength, coverage,
            citation_passed, True, "high_risk_action"
        )
    return ConfidenceAssessment(
        level, score, reason, evidence_count, unique_papers, top_score, relative_strength,
        coverage, citation_passed, False, "normal_answer"
    )
