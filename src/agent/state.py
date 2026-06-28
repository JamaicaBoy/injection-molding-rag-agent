from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentState:
    query: str
    normalized_query: str = ""
    intent: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    confidence: str | float = "low"
    need_human_review: bool = False
    errors: list[str] = field(default_factory=list)
    step_count: int = 0

    query_rewrite: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    node_history: list[str] = field(default_factory=list)
    retrieval_history: list[dict[str, Any]] = field(default_factory=list)
    retrieval_stalled: bool = False
    tool_error_count: int = 0
    generated_evidence: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    guardrail_violations: list[str] = field(default_factory=list)
    knowledge_gap: dict[str, Any] | None = None
    review_ticket: dict[str, Any] | None = None
    final_output: dict[str, Any] | None = None

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

