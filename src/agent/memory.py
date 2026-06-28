from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.retrieval.query_rewrite import rewrite_query


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "data" / "logs" / "agent_memory.jsonl"
DEFAULT_CLI_MEMORY_PATH = Path("data/logs/agent_trace.jsonl")
MAX_EVIDENCE_ITEMS = 5
MAX_SUMMARY_LENGTH = 160
MAX_QUERY_LENGTH = 1000
MAX_FEEDBACK_LENGTH = 500
FORBIDDEN_MEMORY_KEYS = {"full_text", "paper_full_text", "raw_text", "document", "documents"}

PRIVACY_PATTERNS = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[REDACTED_ID]"),
    (re.compile(r"(?:姓名|联系人)\s*[:：]\s*[\u4e00-\u9fff]{2,4}"), "[REDACTED_NAME]"),
    (re.compile(r"(?:工号|客户编号|订单号)\s*[:：]\s*[A-Za-z0-9_-]{3,}"), "[REDACTED_IDENTIFIER]"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(value: str, limit: int | None = None) -> str:
    sanitized = re.sub(r"\s+", " ", str(value)).strip()
    for pattern, replacement in PRIVACY_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized[:limit] if limit is not None else sanitized


def sanitize_value(value: Any, string_limit: int = MAX_FEEDBACK_LENGTH) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, string_limit)
    if isinstance(value, dict):
        return {
            str(key): sanitize_value(item, string_limit)
            for key, item in value.items()
            if str(key).lower() not in FORBIDDEN_MEMORY_KEYS
        }
    if isinstance(value, list):
        return [sanitize_value(item, string_limit) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item, string_limit) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value), string_limit)


def _evidence_id(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return str(item.get("evidence_id") or metadata.get("evidence_id") or "")


def _evidence_summary(item: dict[str, Any]) -> str:
    return sanitize_text(
        str(item.get("text_preview") or item.get("matched_text") or item.get("summary") or ""),
        MAX_SUMMARY_LENGTH,
    )


def _lexical_tokens(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z0-9_/-]{2,}", normalized))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _is_low_confidence(value: str | float | int) -> bool:
    if isinstance(value, (int, float)):
        return float(value) < 0.65
    return str(value).lower() in {"low", "very_low", "unknown", ""}


class AgentMemory:
    """Small local JSONL memory containing only query-level audit metadata."""

    def __init__(
        self,
        memory_path: Path = DEFAULT_MEMORY_PATH,
        clock: Callable[[], str] = _now,
    ) -> None:
        self.memory_path = Path(memory_path)
        self.clock = clock
        self._lock = threading.Lock()

    def record_query(
        self,
        query: str,
        intent: str,
        evidence: list[dict[str, Any]] | None,
        answer_confidence: str | float,
        need_human_review: bool,
        user_feedback: Any = None,
    ) -> dict[str, Any]:
        sanitized_query = sanitize_text(query, MAX_QUERY_LENGTH)
        if not sanitized_query:
            raise ValueError("query must not be empty")

        top_ids: list[str] = []
        summaries: list[dict[str, str]] = []
        for item in (evidence or [])[:MAX_EVIDENCE_ITEMS]:
            evidence_id = _evidence_id(item)
            if evidence_id and evidence_id not in top_ids:
                top_ids.append(evidence_id)
                summaries.append({"evidence_id": evidence_id, "summary": _evidence_summary(item)})

        record = {
            "timestamp": self.clock(),
            "query": sanitized_query,
            "intent": sanitize_text(intent, 80),
            "top_evidence_ids": top_ids,
            "evidence_summaries": summaries,
            "answer_confidence": answer_confidence,
            "need_human_review": bool(need_human_review),
            "user_feedback": sanitize_value(user_feedback),
        }
        self._append(record)
        return record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.memory_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def read_recent(self, limit: int = 5) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(reversed(self.read_all()[-limit:]))

    def find_recent_similar(
        self,
        query: str,
        intent: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.35,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        rewritten = rewrite_query(sanitize_text(query, MAX_QUERY_LENGTH))
        target_intent = intent or rewritten.intent
        target_entities = set([*rewritten.defect_type, *rewritten.material, *rewritten.parameters, *rewritten.quality_metric])
        target_tokens = _lexical_tokens(query)

        matches: list[dict[str, Any]] = []
        for record in reversed(self.read_all()):
            record_rewrite = rewrite_query(str(record.get("query", "")))
            record_entities = set(
                [
                    *record_rewrite.defect_type,
                    *record_rewrite.material,
                    *record_rewrite.parameters,
                    *record_rewrite.quality_metric,
                ]
            )
            entity_score = _jaccard(target_entities, record_entities)
            lexical_score = _jaccard(target_tokens, _lexical_tokens(str(record.get("query", ""))))
            intent_match = str(record.get("intent", "")) == target_intent
            if target_entities or record_entities:
                similarity = 0.65 * entity_score + 0.25 * lexical_score + (0.10 if intent_match else 0.0)
            else:
                similarity = 0.80 * lexical_score + (0.20 if intent_match else 0.0)
            if intent_match and similarity >= min_similarity:
                matches.append({**record, "similarity_score": round(similarity, 4)})
            if len(matches) >= limit:
                break
        return matches

    def similar_question_hint(self, query: str, intent: str | None = None, limit: int = 3) -> str | None:
        matches = self.find_recent_similar(query, intent=intent, limit=limit)
        if not matches:
            return None
        return f"你之前问过 {len(matches)} 个类似问题，最近一次是：{matches[0]['query']}"

    def export_statistics(self, top_n: int = 10) -> dict[str, Any]:
        records = self.read_all()
        rewrites = [rewrite_query(str(record.get("query", ""))) for record in records if record.get("query")]
        defect_counts = Counter(item for rewritten in rewrites for item in rewritten.defect_type)
        parameter_counts = Counter(item for rewritten in rewrites for item in rewritten.parameters)
        high_frequency_defects = [
            {"defect_type": name, "count": count} for name, count in defect_counts.most_common(top_n)
        ]
        high_frequency_parameters = [
            {"parameter": name, "count": count} for name, count in parameter_counts.most_common(top_n)
        ]
        low_confidence_count = sum(
            1 for record in records if _is_low_confidence(record.get("answer_confidence", "low"))
        )
        return {
            "total_records": len(records),
            "high_frequency_defects": high_frequency_defects,
            "high_frequency_parameters": high_frequency_parameters,
            "low_confidence_count": low_confidence_count,
            "human_review_count": sum(1 for record in records if record.get("need_human_review") is True),
            # Compatibility aliases for existing callers.
            "total_queries": len(records),
            "frequent_defects": high_frequency_defects,
            "frequent_parameters": high_frequency_parameters,
            "low_confidence_question_count": low_confidence_count,
        }

    def _append(self, record: dict[str, Any]) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.memory_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _format_counts(items: list[dict[str, Any]], name_key: str) -> str:
    if not items:
        return "(none)"
    return ", ".join(f"{item[name_key]}={item['count']}" for item in items)


def run_demo(memory: AgentMemory, recent_n: int) -> None:
    demo_records = [
        {
            "query": "缩水咋办，保压压力可能有什么影响？",
            "intent": "defect_diagnosis",
            "evidence": [{"evidence_id": "E1", "text_preview": "Packing pressure is related to shrinkage in the retrieved experiment."}],
            "answer_confidence": "low",
            "need_human_review": True,
            "user_feedback": "希望补充更多论文证据",
        },
        {
            "query": "翘曲与模具温度有什么关系？",
            "intent": "parameter_effect",
            "evidence": [{"evidence_id": "E2", "text_preview": "Mold temperature was evaluated together with warpage."}],
            "answer_confidence": "high",
            "need_human_review": False,
            "user_feedback": None,
        },
        {
            "query": "有哪些机器学习方法用于注塑质量预测？",
            "intent": "method_search",
            "evidence": [{"evidence_id": "E3", "text_preview": "The retrieved evidence discusses machine-learning quality prediction."}],
            "answer_confidence": "medium",
            "need_human_review": False,
            "user_feedback": "结果可用于文献调研",
        },
    ]
    for record in demo_records:
        memory.record_query(**record)

    recent = memory.read_recent(recent_n)
    stats = memory.export_statistics()
    print("Memory demo completed.")
    print(f"memory_path: {memory.memory_path.as_posix()}")
    print(f"records_written: {len(demo_records)}")
    print(f"recent_records: {len(recent)}")
    print(f"total_records: {stats['total_records']}")
    print(f"high_frequency_defects: {_format_counts(stats['high_frequency_defects'], 'defect_type')}")
    print(f"high_frequency_parameters: {_format_counts(stats['high_frequency_parameters'], 'parameter')}")
    print(f"low_confidence_count: {stats['low_confidence_count']}")
    for index, record in enumerate(recent, start=1):
        print(f"recent_{index}: [{record.get('intent', '')}] {record.get('query', '')}")


def print_statistics(memory: AgentMemory) -> None:
    stats = memory.export_statistics()
    print("Memory statistics.")
    print(f"memory_path: {memory.memory_path.as_posix()}")
    print(f"total_records: {stats['total_records']}")
    print(f"high_frequency_defects: {_format_counts(stats['high_frequency_defects'], 'defect_type')}")
    print(f"high_frequency_parameters: {_format_counts(stats['high_frequency_parameters'], 'parameter')}")
    print(f"low_confidence_count: {stats['low_confidence_count']}")
    print(f"human_review_count: {stats['human_review_count']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or demonstrate the local JSONL Agent memory.")
    parser.add_argument("--memory_path", type=Path, default=DEFAULT_CLI_MEMORY_PATH)
    parser.add_argument("--recent_n", type=int, default=5)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.recent_n <= 0:
        raise SystemExit("--recent_n must be positive")
    memory = AgentMemory(args.memory_path)
    if args.demo:
        run_demo(memory, args.recent_n)
    if args.stats:
        print_statistics(memory)
    if not args.demo and not args.stats:
        print("No action selected. Use --demo or --stats.")
    return 0


if __name__ == "__main__":
    main()
