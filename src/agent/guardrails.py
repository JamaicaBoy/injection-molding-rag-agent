from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


PARAMETER_RANGE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:-|–|~|～|至|到)\s*\d+(?:\.\d+)?\s*(?:MPa|bar|℃|°C|mm/s|cm3/s|s|秒|分钟|%)",
    flags=re.IGNORECASE,
)

PRODUCTION_DIRECTIVE_PATTERNS = (
    re.compile(r"(?:立即|直接|马上|必须|务必)(?:把|将)?[^。；\n]{0,50}(?:调到|设为|设置为|改为|提高到|降低到)"),
    re.compile(r"(?:把|将)(?:保压|注射|熔体|模具|冷却|速度|压力|温度)[^。；\n]{0,30}(?:调到|设为|设置为|改为)"),
    re.compile(r"(?:set|adjust|change)\s+(?:the\s+)?[^.;\n]{0,40}\s+to\s+\d", flags=re.IGNORECASE),
)

UNCERTAINTY_MARKERS = ("可能", "取决于", "条件", "冲突", "不同论文", "无法确定", "不一致", "需进一步", "uncertain", "depends", "conflict")


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    violations: list[str]
    unsupported_parameter_ranges: list[str]
    evidence_conflict: bool
    need_human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower().replace("–", "-").replace("～", "~").replace("到", "至")


def evidence_has_conflict(evidence: list[dict[str, Any]]) -> bool:
    directions: set[str] = set()
    for item in evidence:
        metadata = item.get("metadata") or {}
        if item.get("conflict") is True or metadata.get("conflict") is True or metadata.get("evidence_conflict") is True:
            return True
        direction = item.get("effect_direction") or metadata.get("effect_direction")
        if direction:
            directions.add(str(direction))
    incompatible = {"increase_positive", "increase_negative"}
    return incompatible.issubset(directions)


def check_answer_guardrails(answer: str, evidence: list[dict[str, Any]]) -> GuardrailResult:
    violations: list[str] = []
    evidence_text = " ".join(
        f"{item.get('title', '')} {item.get('text_preview', '')} {item.get('matched_text', '')}" for item in evidence
    )
    supported_ranges = {_normalize(value) for value in PARAMETER_RANGE_PATTERN.findall(evidence_text)}
    answer_ranges = list(dict.fromkeys(PARAMETER_RANGE_PATTERN.findall(answer)))
    unsupported_ranges = [value for value in answer_ranges if _normalize(value) not in supported_ranges]
    if unsupported_ranges:
        violations.append("unsupported_process_parameter_range")

    safe_answer = answer.replace("不能作为直接生产指令", "").replace("不得作为直接生产指令", "")
    if any(pattern.search(safe_answer) for pattern in PRODUCTION_DIRECTIVE_PATTERNS):
        violations.append("paper_conclusion_presented_as_production_instruction")

    conflict = evidence_has_conflict(evidence)
    if conflict and not any(marker in answer.lower() for marker in UNCERTAINTY_MARKERS):
        violations.append("single_certain_answer_despite_evidence_conflict")

    return GuardrailResult(
        passed=not violations,
        violations=violations,
        unsupported_parameter_ranges=unsupported_ranges,
        evidence_conflict=conflict,
        need_human_review=bool(violations),
    )

