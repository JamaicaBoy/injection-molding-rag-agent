from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agent.conversation_summarizer import (
    ConversationSummary,
    DEFAULT_RECENT_TURNS_AFTER_SUMMARY,
    SUMMARY_TOKEN_THRESHOLD,
    SUMMARY_TRIGGER_TURNS,
    SummarizerLLMClient,
    maybe_compress_turns,
)
from src.agent.memory import sanitize_text
from src.retrieval.query_rewrite import RewrittenQuery, rewrite_query, unique

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERSATION_DIR = PROJECT_ROOT / "data" / "runtime" / "conversations"
# A generous hard cap on in-memory turns. The rolling summarizer (see
# conversation_summarizer.py) compresses everything older than
# `recent_turns_after_summary` into a ConversationSummary well before this
# cap is ever reached under default settings; this is just a backstop.
DEFAULT_MAX_TURNS = 20
MAX_QUESTION_LENGTH = 500
MAX_ANSWER_BRIEF_LENGTH = 200

ENTITY_FIELDS = ("defect_type", "material", "parameters", "quality_metric")

# Markers that suggest the user is referring back to something said earlier
# instead of restating it explicitly (e.g. "那对缩水呢？", "这个会影响翘曲吗？").
FOLLOWUP_MARKERS = (
    "那",
    "这个",
    "这种",
    "它",
    "该",
    "此",
    "上述",
    "上面",
    "刚才",
    "之前",
    "继续",
    "呢",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def has_followup_marker(question: str) -> bool:
    return any(marker in question for marker in FOLLOWUP_MARKERS)


@dataclass
class ConversationTurn:
    """One compact conversation turn kept in short-term memory.

    Deliberately does NOT store full retrieval evidence or paper text -
    only the question, a short answer brief, extracted entities, and the
    cited paper_id list, so the conversation log stays small and safe.
    """

    turn_index: int
    timestamp: str
    user_question: str
    system_answer_brief: str
    key_entities: dict[str, list[str]]
    cited_paper_ids: list[str]
    need_human_review: bool = False
    review_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
            "user_question": self.user_question,
            "system_answer_brief": self.system_answer_brief,
            "key_entities": self.key_entities,
            "cited_paper_ids": self.cited_paper_ids,
            "need_human_review": self.need_human_review,
            "review_reason": self.review_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationTurn":
        raw_entities = payload.get("key_entities") or {}
        return cls(
            turn_index=int(payload.get("turn_index", 0)),
            timestamp=str(payload.get("timestamp", "")),
            user_question=str(payload.get("user_question", "")),
            system_answer_brief=str(payload.get("system_answer_brief", "")),
            key_entities={name: list(raw_entities.get(name, []) or []) for name in ENTITY_FIELDS},
            cited_paper_ids=list(payload.get("cited_paper_ids", []) or []),
            need_human_review=bool(payload.get("need_human_review", False)),
            review_reason=str(payload.get("review_reason", "")),
        )


class ConversationState:
    """Short-term, per-session conversation memory.

    Keeps only the most recent `max_turns` turns in memory and persists every
    turn as one JSONL line under data/runtime/conversations/<conversation_id>.jsonl
    so a session can later be reloaded. The on-disk directory is local-only
    runtime data (see .gitignore: data/runtime/) and is never committed.
    """

    def __init__(
        self,
        conversation_id: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        storage_dir: Path = DEFAULT_CONVERSATION_DIR,
        clock: Callable[[], str] = _now,
        recent_turns_after_summary: int = DEFAULT_RECENT_TURNS_AFTER_SUMMARY,
        summary_trigger_turns: int = SUMMARY_TRIGGER_TURNS,
        summary_token_threshold: int = SUMMARY_TOKEN_THRESHOLD,
        summarizer_llm_client: SummarizerLLMClient | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.conversation_id = conversation_id or new_conversation_id()
        self.max_turns = max_turns
        self.storage_dir = Path(storage_dir)
        self.clock = clock
        self.recent_turns_after_summary = recent_turns_after_summary
        self.summary_trigger_turns = summary_trigger_turns
        self.summary_token_threshold = summary_token_threshold
        self.summarizer_llm_client = summarizer_llm_client
        self._lock = threading.Lock()
        self.turns: list[ConversationTurn] = []
        self.summary: ConversationSummary | None = None

    @property
    def storage_path(self) -> Path:
        return self.storage_dir / f"{self.conversation_id}.jsonl"

    def add_turn(
        self,
        user_question: str,
        system_answer: str,
        evidence_list: list[dict[str, Any]] | None = None,
        rewrite: RewrittenQuery | dict[str, Any] | None = None,
        need_human_review: bool = False,
        review_reason: str | None = None,
    ) -> ConversationTurn:
        sanitized_question = sanitize_text(user_question, MAX_QUESTION_LENGTH)
        if not sanitized_question:
            raise ValueError("user_question must not be empty")
        brief = sanitize_text(system_answer or "", MAX_ANSWER_BRIEF_LENGTH)
        entities = self._extract_entities(sanitized_question, rewrite)
        paper_ids = self._extract_paper_ids(evidence_list)

        turn = ConversationTurn(
            turn_index=len(self.turns) + 1,
            timestamp=self.clock(),
            user_question=sanitized_question,
            system_answer_brief=brief,
            key_entities=entities,
            cited_paper_ids=paper_ids,
            need_human_review=bool(need_human_review),
            review_reason=sanitize_text(review_reason or "", MAX_ANSWER_BRIEF_LENGTH),
        )
        self.turns.append(turn)
        self.turns = self.turns[-self.max_turns :]
        self._append(turn)
        self._maybe_compress()
        return turn

    def _maybe_compress(self) -> bool:
        """Roll older turns into `self.summary` once the in-memory window has
        grown past the configured trigger (turn count or token estimate), so
        long conversations degrade to summary + recent_turns instead of
        silently losing history once `max_turns` is hit."""
        new_turns, new_summary, triggered = maybe_compress_turns(
            self.turns,
            self.summary,
            recent_keep=self.recent_turns_after_summary,
            llm_client=self.summarizer_llm_client,
            max_turns=self.summary_trigger_turns,
            token_threshold=self.summary_token_threshold,
        )
        if triggered:
            self.turns = new_turns
            self.summary = new_summary
        return triggered

    @staticmethod
    def _extract_entities(
        question: str, rewrite: RewrittenQuery | dict[str, Any] | None
    ) -> dict[str, list[str]]:
        if rewrite is None and question:
            rewrite = rewrite_query(question)
        if isinstance(rewrite, RewrittenQuery):
            payload: dict[str, Any] = rewrite.to_dict()
        elif isinstance(rewrite, dict):
            payload = rewrite
        else:
            payload = {}
        return {name: list(payload.get(name, []) or []) for name in ENTITY_FIELDS}

    @staticmethod
    def _extract_paper_ids(evidence_list: list[dict[str, Any]] | None) -> list[str]:
        paper_ids: list[str] = []
        for item in evidence_list or []:
            paper_id = str(item.get("paper_id") or "")
            if paper_id and paper_id not in paper_ids:
                paper_ids.append(paper_id)
        return paper_ids[:5]

    def last_entities(self) -> dict[str, list[str]]:
        """Most recent non-empty value per entity category, newest turn wins."""
        merged: dict[str, list[str]] = {name: [] for name in ENTITY_FIELDS}
        for turn in reversed(self.turns):
            for name in ENTITY_FIELDS:
                if not merged[name] and turn.key_entities.get(name):
                    merged[name] = list(turn.key_entities[name])
            if all(merged.values()):
                break
        return merged

    def recent_turns(self, limit: int | None = None) -> list[ConversationTurn]:
        if limit is None:
            return list(self.turns)
        if limit <= 0:
            return []
        return self.turns[-limit:]

    def clear(self, delete_file: bool = True) -> None:
        with self._lock:
            self.turns = []
            self.summary = None
            if delete_file and self.storage_path.exists():
                try:
                    self.storage_path.unlink()
                except OSError:
                    pass

    def summary_dict(self) -> dict[str, Any] | None:
        """The current rolling summary as a plain dict, or None if the
        conversation has not yet grown long enough to trigger one."""
        return self.summary.to_dict() if self.summary is not None else None

    def _append(self, turn: ConversationTurn) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, self.storage_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(
        cls,
        conversation_id: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        storage_dir: Path = DEFAULT_CONVERSATION_DIR,
        clock: Callable[[], str] = _now,
    ) -> "ConversationState":
        state = cls(
            conversation_id=conversation_id,
            max_turns=max_turns,
            storage_dir=storage_dir,
            clock=clock,
        )
        path = state.storage_path
        if path.exists():
            loaded: list[ConversationTurn] = []
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    try:
                        loaded.append(ConversationTurn.from_dict(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
            state.turns = loaded[-max_turns:]
        return state

    @classmethod
    def from_chat_session(cls, chat: dict[str, Any]) -> "ConversationState":
        """Build bounded reasoning memory from a persisted chat session.

        Chat messages remain the UI history source. This adapter only creates
        compact ConversationTurn objects for follow-up resolution and the
        context manager; it never copies full evidence or chunk text.
        """
        state = cls(conversation_id=str(chat.get("conversation_id") or new_conversation_id()))
        summary = chat.get("summary")
        if isinstance(summary, ConversationSummary):
            state.summary = summary
        elif isinstance(summary, dict):
            state.summary = ConversationSummary.from_dict(summary)

        pending_user: dict[str, Any] | None = None
        turns: list[ConversationTurn] = []
        for message in chat.get("messages", []):
            role = message.get("role")
            if role == "user":
                pending_user = message
                continue
            if role != "assistant" or pending_user is None:
                continue
            rewrite = message.get("query_rewrite") or {}
            evidence = message.get("evidence") or []
            turns.append(
                ConversationTurn(
                    turn_index=len(turns) + 1,
                    timestamp=str(message.get("created_at") or pending_user.get("created_at") or ""),
                    user_question=sanitize_text(
                        str(pending_user.get("content", "")), MAX_QUESTION_LENGTH
                    ),
                    system_answer_brief=sanitize_text(
                        str(message.get("content", "")), MAX_ANSWER_BRIEF_LENGTH
                    ),
                    key_entities={
                        name: list(rewrite.get(name, []) or []) for name in ENTITY_FIELDS
                    },
                    cited_paper_ids=cls._extract_paper_ids(evidence),
                    need_human_review=bool(message.get("need_human_review", False)),
                    review_reason=str(
                        (message.get("agent_trace_summary") or {}).get(
                            "human_review_reason", ""
                        )
                    )[:MAX_ANSWER_BRIEF_LENGTH],
                )
            )
            pending_user = None
        state.turns = turns[-state.max_turns :]
        state._maybe_compress()
        return state

    def summary_text(self) -> str | None:
        if self.summary is None:
            return None
        return json.dumps(self.summary.to_dict(), ensure_ascii=False)


def ensure_current_conversation_state(
    value: Any,
    *,
    conversation_id: str | None = None,
) -> ConversationState:
    """Upgrade a Streamlit session object created by an older class version.

    Streamlit keeps objects in ``session_state`` across source hot reloads. If
    ConversationState gains attributes, those existing instances do not run
    the new ``__init__``. Rebuild with the current class while retaining the
    compact turn history and rolling summary.
    """
    required_attributes = (
        "summary",
        "recent_turns_after_summary",
        "summary_trigger_turns",
        "summary_token_threshold",
        "summarizer_llm_client",
    )
    if isinstance(value, ConversationState) and all(
        hasattr(value, attribute) for attribute in required_attributes
    ):
        return value

    current_id = str(
        conversation_id or getattr(value, "conversation_id", "") or new_conversation_id()
    )
    max_turns = int(getattr(value, "max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS)
    storage_dir = Path(getattr(value, "storage_dir", DEFAULT_CONVERSATION_DIR))
    clock = getattr(value, "clock", _now)
    if not callable(clock):
        clock = _now
    upgraded = ConversationState(
        conversation_id=current_id,
        max_turns=max_turns,
        storage_dir=storage_dir,
        clock=clock,
        recent_turns_after_summary=int(
            getattr(value, "recent_turns_after_summary", DEFAULT_RECENT_TURNS_AFTER_SUMMARY)
        ),
        summary_trigger_turns=int(
            getattr(value, "summary_trigger_turns", SUMMARY_TRIGGER_TURNS)
        ),
        summary_token_threshold=int(
            getattr(value, "summary_token_threshold", SUMMARY_TOKEN_THRESHOLD)
        ),
        summarizer_llm_client=getattr(value, "summarizer_llm_client", None),
    )

    converted_turns: list[ConversationTurn] = []
    for turn in list(getattr(value, "turns", []) or []):
        if isinstance(turn, ConversationTurn):
            converted_turns.append(turn)
        elif isinstance(turn, dict):
            converted_turns.append(ConversationTurn.from_dict(turn))
        elif hasattr(turn, "to_dict"):
            converted_turns.append(ConversationTurn.from_dict(turn.to_dict()))
    upgraded.turns = converted_turns[-max_turns:]

    previous_summary = getattr(value, "summary", None)
    if isinstance(previous_summary, ConversationSummary):
        upgraded.summary = previous_summary
    elif isinstance(previous_summary, dict):
        upgraded.summary = ConversationSummary.from_dict(previous_summary)
    elif hasattr(previous_summary, "to_dict"):
        upgraded.summary = ConversationSummary.from_dict(previous_summary.to_dict())
    return upgraded


def resolve_followup_query(question: str, history: ConversationState) -> RewrittenQuery:
    """Rewrite `question`, filling omitted referents ("那"/"这个"/"呢" ...) from history.

    Only entity categories that the current question left empty are filled in
    from the most recent turns; anything the user states explicitly always
    wins. If the question carries no follow-up marker, or history has nothing
    to offer, the plain `rewrite_query` result is returned unchanged.
    """
    base = rewrite_query(question)
    if not has_followup_marker(question):
        return base

    history_entities = history.last_entities()
    if not any(history_entities.values()):
        return base

    merged: dict[str, list[str]] = {}
    changed = False
    for name in ENTITY_FIELDS:
        current = list(getattr(base, name))
        if not current and history_entities.get(name):
            merged[name] = list(history_entities[name])
            changed = True
        else:
            merged[name] = current

    if not changed:
        return base

    must_have_terms = unique(
        [*merged["defect_type"], *merged["material"], *merged["parameters"], *merged["quality_metric"]]
    )
    appended_terms = [term for term in must_have_terms if term not in base.normalized_query]
    normalized_query = base.normalized_query
    if appended_terms:
        normalized_query = f"{normalized_query} {' '.join(appended_terms)}".strip()

    intent = base.intent
    if intent == "general_qa":
        if merged["parameters"]:
            intent = "parameter_effect"
        elif merged["defect_type"]:
            intent = "defect_diagnosis"

    return replace(
        base,
        normalized_query=normalized_query,
        intent=intent,
        defect_type=merged["defect_type"],
        material=merged["material"],
        parameters=merged["parameters"],
        quality_metric=merged["quality_metric"],
        must_have_terms=must_have_terms,
    )
