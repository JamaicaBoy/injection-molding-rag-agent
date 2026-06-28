from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "manual_review" / "review_queue.csv"
DEFAULT_REVIEW_FIELDS = ["card_type", "evidence_paper_id", "item_name", "confidence", "reason"]

CITATION_PATTERN = re.compile(r"\[E(\d+)\]", flags=re.IGNORECASE)
SPECIFIC_VALUE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"\d+(?:\.\d+)?\s*(?:-|–|~|～|至)\s*\d+(?:\.\d+)?\s*(?:%|mpa|bar|℃|°c|mm|μm|um|ms|s|秒|分钟|小时)?"
    r"|\d+(?:\.\d+)?\s*(?:%|mpa|bar|℃|°c|mm|μm|um|ms|秒|分钟|小时)"
    r"|\d+\.\d+"
    r"|\d{2,}"
    r")(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
TITLE_PATTERNS = (
    re.compile(r"《([^》]{4,})》"),
    re.compile(r"(?:论文|文献|研究)[“\"]([^”\"]{4,})[”\"]"),
    re.compile(r"(?:paper|study)\s+(?:titled|named)\s+[\"']([^\"']{4,})[\"']", flags=re.IGNORECASE),
)


@dataclass(frozen=True)
class CitationGuardResult:
    passed: bool
    citations: list[str]
    invalid_citations: list[str]
    unsupported_values: list[str]
    unsupported_titles: list[str]
    issues: list[str]
    high_risk: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", "", value).lower().replace("–", "-").replace("～", "~")


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", title.lower())


def mentioned_titles(answer: str) -> list[str]:
    return list(dict.fromkeys(match.group(1).strip() for pattern in TITLE_PATTERNS for match in pattern.finditer(answer)))


def check_citations(answer: str, evidence_list: list[dict[str, Any]]) -> CitationGuardResult:
    valid_ids = {str(item.get("evidence_id", "")).upper() for item in evidence_list}
    citations = list(dict.fromkeys(f"E{number}".upper() for number in CITATION_PATTERN.findall(answer)))
    invalid_citations = [citation for citation in citations if citation not in valid_ids]

    issues: list[str] = []
    if evidence_list and not citations:
        issues.append("answer_has_no_citation")
    if invalid_citations:
        issues.append("citation_id_not_in_evidence")

    answer_without_citations = CITATION_PATTERN.sub("", answer)
    evidence_text = " ".join(
        f"{item.get('title', '')} {item.get('text_preview', '')}" for item in evidence_list
    )
    evidence_values = {normalize_value(value) for value in SPECIFIC_VALUE_PATTERN.findall(evidence_text)}
    answer_values = list(dict.fromkeys(SPECIFIC_VALUE_PATTERN.findall(answer_without_citations)))
    unsupported_values = [value for value in answer_values if normalize_value(value) not in evidence_values]
    if unsupported_values:
        issues.append("specific_value_not_in_evidence")

    allowed_titles = [normalize_title(str(item.get("title", ""))) for item in evidence_list]
    titles = mentioned_titles(answer)
    unsupported_titles = []
    for title in titles:
        normalized = normalize_title(title)
        if not normalized or normalized not in allowed_titles:
            unsupported_titles.append(title)
    if unsupported_titles:
        issues.append("paper_title_not_in_evidence")

    high_risk = bool(issues)
    return CitationGuardResult(
        passed=not high_risk,
        citations=citations,
        invalid_citations=invalid_citations,
        unsupported_values=unsupported_values,
        unsupported_titles=unsupported_titles,
        issues=issues,
        high_risk=high_risk,
    )


def append_review_queue(
    question: str,
    evidence_list: list[dict[str, Any]],
    confidence: str,
    reasons: list[str],
    review_queue: Path = DEFAULT_REVIEW_QUEUE,
) -> None:
    review_queue = Path(review_queue)
    review_queue.parent.mkdir(parents=True, exist_ok=True)
    file_exists = review_queue.exists() and review_queue.stat().st_size > 0
    fieldnames = DEFAULT_REVIEW_FIELDS
    if file_exists:
        with review_queue.open("r", encoding="utf-8", newline="") as file:
            fieldnames = next(csv.reader(file), DEFAULT_REVIEW_FIELDS)

    paper_ids = list(dict.fromkeys(str(item.get("paper_id", "")) for item in evidence_list if item.get("paper_id")))
    known_values = {
        "card_type": "rag_answer",
        "evidence_paper_id": ";".join(paper_ids),
        "item_name": question[:200],
        "confidence": confidence,
        "reason": ";".join(reasons),
    }
    row = {field: known_values.get(field, "") for field in fieldnames}
    with review_queue.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
