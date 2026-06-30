from __future__ import annotations

import operator
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Canonical LangGraph data flow:
    # query -> query_info -> retrieved_docs -> reranked_docs -> evidence_list.
    query: str
    conversation_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    query_info: dict[str, Any]
    retrieved_docs: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    draft_answer: str
    final_answer: str
    citations: list[str]
    confidence: str | float
    confidence_reason: str
    need_human_review: bool
    human_review_reason: str
    llm_mode: str
    llm_model: str
    llm_fallback_reason: str
    errors: Annotated[list[str], operator.add]
    step_count: int

    evidence_list: list[dict[str, Any]]
    limitations: list[str]
    citation_passed: bool
    risk_passed: bool
    revision_count: int
    review_ticket: dict[str, Any] | None
    memory_updated: bool
    node_history: Annotated[list[str], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
