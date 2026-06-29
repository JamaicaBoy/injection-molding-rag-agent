from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from src.agent.context_manager import estimate_tokens
from src.agent.memory import sanitize_text

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Trigger a rolling summary once the in-memory turn count exceeds this, OR
# once the estimated token cost of all in-memory turns exceeds the threshold
# below - whichever happens first.
SUMMARY_TRIGGER_TURNS = 8
SUMMARY_TOKEN_THRESHOLD = 1400

# How many of the most recent turns stay in full (verbatim) after a summary
# pass collapses everything older into the rolling ConversationSummary.
DEFAULT_RECENT_TURNS_AFTER_SUMMARY = 4

MAX_OPEN_QUESTIONS = 6
MAX_HUMAN_REVIEW_ITEMS = 6
MAX_USER_GOAL_CHARS = 160

# Maps a ConversationTurn's key_entities category to the ConversationSummary
# field it is allowed to be promoted into - and ONLY promoted into when the
# same turn also carries at least one cited_paper_id (see rule_summarize_turns).
ENTITY_TO_FIELD = {
    "material": "confirmed_materials",
    "defect_type": "confirmed_defects",
    "parameters": "confirmed_parameters",
}


class SummarizerLLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class ConversationSummary:
    """Rolling summary of conversation turns that have been compressed out of
    short-term memory. Deliberately mirrors the fields the user asked for so
    it can be merged turn-batch by turn-batch as a conversation grows.
    """

    user_goal: str = ""
    confirmed_materials: list[str] = field(default_factory=list)
    confirmed_defects: list[str] = field(default_factory=list)
    confirmed_parameters: list[str] = field(default_factory=list)
    cited_papers: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    human_review_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_goal": self.user_goal,
            "confirmed_materials": list(self.confirmed_materials),
            "confirmed_defects": list(self.confirmed_defects),
            "confirmed_parameters": list(self.confirmed_parameters),
            "cited_papers": list(self.cited_papers),
            "open_questions": list(self.open_questions),
            "human_review_items": list(self.human_review_items),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationSummary":
        return cls(
            user_goal=str(payload.get("user_goal", "")),
            confirmed_materials=list(payload.get("confirmed_materials", []) or []),
            confirmed_defects=list(payload.get("confirmed_defects", []) or []),
            confirmed_parameters=list(payload.get("confirmed_parameters", []) or []),
            cited_papers=list(payload.get("cited_papers", []) or []),
            open_questions=list(payload.get("open_questions", []) or []),
            human_review_items=list(payload.get("human_review_items", []) or []),
        )


def _turn_payload(turn: Any) -> dict[str, Any]:
    if isinstance(turn, dict):
        return turn
    if hasattr(turn, "to_dict"):
        return turn.to_dict()
    raise TypeError(f"Expected a conversation turn dict or object with to_dict(), got {type(turn).__name__}")


def _dedupe(items: Iterable[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    if limit is not None and limit >= 0:
        return result[-limit:]
    return result


def should_trigger_summary(
    turns: Iterable[Any],
    max_turns: int = SUMMARY_TRIGGER_TURNS,
    token_threshold: int = SUMMARY_TOKEN_THRESHOLD,
) -> bool:
    """True once `turns` (the in-memory window) is large enough by count or by
    estimated token cost that it should be rolled into a summary instead of
    being kept verbatim."""
    turn_list = [_turn_payload(t) for t in turns]
    if len(turn_list) > max_turns:
        return True
    combined_text = " ".join(
        f"{t.get('user_question', '')} {t.get('system_answer_brief', '')}" for t in turn_list
    )
    return estimate_tokens(combined_text) > token_threshold


def rule_summarize_turns(
    turns: Iterable[Any],
    previous_summary: ConversationSummary | None = None,
) -> ConversationSummary:
    """Deterministic, citation-gated summarizer - the sole source of truth for
    confirmed_*/cited_papers/human_review_items.

    Per requirement, an entity (material/defect/parameter) is only promoted
    into a `confirmed_*` field when the SAME turn also carries at least one
    `cited_paper_id` - i.e. when the answer that mentioned it was actually
    grounded in retrieved evidence. A turn with entities but no citation never
    touches confirmed_*; its question is recorded under `open_questions`
    instead, so unverified model guesses can never leak into a confirmed field.
    """
    turn_list = [_turn_payload(t) for t in turns]

    summary = ConversationSummary(
        user_goal=previous_summary.user_goal if previous_summary else "",
        confirmed_materials=list(previous_summary.confirmed_materials) if previous_summary else [],
        confirmed_defects=list(previous_summary.confirmed_defects) if previous_summary else [],
        confirmed_parameters=list(previous_summary.confirmed_parameters) if previous_summary else [],
        cited_papers=list(previous_summary.cited_papers) if previous_summary else [],
        open_questions=list(previous_summary.open_questions) if previous_summary else [],
        human_review_items=list(previous_summary.human_review_items) if previous_summary else [],
    )

    if not summary.user_goal and turn_list:
        first_question = str(turn_list[0].get("user_question", "")).strip()
        summary.user_goal = sanitize_text(first_question, MAX_USER_GOAL_CHARS)

    for turn in turn_list:
        cited = [str(p) for p in (turn.get("cited_paper_ids") or []) if p]
        entities = turn.get("key_entities") or {}
        question = str(turn.get("user_question", "")).strip()
        has_citation = bool(cited)

        if has_citation:
            summary.cited_papers.extend(cited)
            for entity_name, field_name in ENTITY_TO_FIELD.items():
                values = [str(v) for v in (entities.get(entity_name) or []) if v]
                if values:
                    getattr(summary, field_name).extend(values)
        elif question:
            # No supporting citation in this turn - never promote to
            # confirmed_*, keep it as an open question instead.
            summary.open_questions.append(question)

        if turn.get("need_human_review"):
            reason = str(turn.get("review_reason") or "需要人工复核").strip()
            summary.human_review_items.append(f"第{turn.get('turn_index', '?')}轮：{reason}")

    summary.confirmed_materials = _dedupe(summary.confirmed_materials)
    summary.confirmed_defects = _dedupe(summary.confirmed_defects)
    summary.confirmed_parameters = _dedupe(summary.confirmed_parameters)
    summary.cited_papers = _dedupe(summary.cited_papers)
    summary.human_review_items = _dedupe(summary.human_review_items, limit=MAX_HUMAN_REVIEW_ITEMS)

    # Once an entity is confirmed, drop any open question that was only about
    # naming that same entity, so resolved items stop cluttering the list.
    confirmed_terms = {
        term
        for term in (*summary.confirmed_materials, *summary.confirmed_defects, *summary.confirmed_parameters)
        if term
    }
    filtered_open = [q for q in summary.open_questions if not any(term in q for term in confirmed_terms)]
    summary.open_questions = _dedupe(filtered_open, limit=MAX_OPEN_QUESTIONS)

    return summary


def _llm_user_goal(turn_list: list[dict[str, Any]], llm_client: SummarizerLLMClient | None) -> str | None:
    if llm_client is None or not turn_list:
        return None
    questions = "\n".join(f"- {t.get('user_question', '')}" for t in turn_list if t.get("user_question"))
    if not questions.strip():
        return None
    system_prompt = (
        "你是对话摘要助手。请用一句简短中文概括用户在本段对话中的核心目标，"
        "不要编造未出现的内容，不要包含具体结论、数值或论文引用，只描述用户想解决的问题。"
    )
    try:
        response = llm_client.generate(system_prompt, questions)
    except Exception:
        return None
    cleaned = sanitize_text(str(response or "").strip(), MAX_USER_GOAL_CHARS)
    return cleaned or None


def summarize_turns(
    turns: Iterable[Any],
    previous_summary: ConversationSummary | None = None,
    llm_client: SummarizerLLMClient | None = None,
) -> ConversationSummary:
    """Build/extend a rolling `ConversationSummary` from `turns` (oldest-first).

    The deterministic rule-based extraction always runs first and is the only
    source of confirmed_*/cited_papers/human_review_items - an optional local
    LLM is never given the chance to inject an uncited claim into a confirmed
    field. If `llm_client` is provided, it is used only to phrase `user_goal`
    more fluently the first time a summary is created; on any failure
    (including the LLM being unavailable) the rule-based fallback is kept
    unchanged, satisfying the "local LLM unavailable -> rule fallback"
    requirement without any special-casing here.
    """
    turn_list = [_turn_payload(t) for t in turns]
    summary = rule_summarize_turns(turn_list, previous_summary)

    if llm_client is not None and (previous_summary is None or not previous_summary.user_goal):
        polished = _llm_user_goal(turn_list, llm_client)
        if polished:
            summary.user_goal = polished

    return summary


def maybe_compress_turns(
    turns: list[Any],
    previous_summary: ConversationSummary | None,
    recent_keep: int = DEFAULT_RECENT_TURNS_AFTER_SUMMARY,
    llm_client: SummarizerLLMClient | None = None,
    max_turns: int = SUMMARY_TRIGGER_TURNS,
    token_threshold: int = SUMMARY_TOKEN_THRESHOLD,
) -> tuple[list[Any], ConversationSummary | None, bool]:
    """If `turns` has grown past the trigger (by count or token estimate),
    roll everything except the most recent `recent_keep` turns into the
    summary and return the shrunk turn list. Returns (turns, summary,
    triggered) - `turns`/`summary` are returned unchanged when not triggered.
    """
    if not should_trigger_summary(turns, max_turns=max_turns, token_threshold=token_threshold):
        return list(turns), previous_summary, False

    recent_keep = max(0, recent_keep)
    if len(turns) <= recent_keep:
        return list(turns), previous_summary, False

    split_at = len(turns) - recent_keep
    to_summarize = turns[:split_at]
    kept = turns[split_at:]
    new_summary = summarize_turns(to_summarize, previous_summary, llm_client)
    return list(kept), new_summary, True
