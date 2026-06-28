from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.agent.guardrails import check_answer_guardrails
from src.eval.eval_retrieval import (
    DEFAULT_INPUT,
    LocalRetrievalRunner,
    RetrievalCallable,
    matched_expected_keywords,
    parse_expected_keywords,
    validate_questions,
)
from src.rag.answer_generator import AnswerGenerator
from src.rag.citation_guard import check_citations
from src.retrieval.query_rewrite import rewrite_query


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "generation_eval.csv"
DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "eval" / "generation_review_queue.csv"


def search_results_to_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in results:
        source = item.get("source_location") or {}
        candidates.append(
            {
                "chunk_id": str(item.get("chunk_id", "")),
                "paper_id": str(item.get("paper_id", "")),
                "title": str(item.get("title", "")),
                "section_name": str(source.get("section") or ""),
                "chunk_type": "",
                "score": float(item.get("rerank_score") or item.get("relevance_score") or 0.0),
                "source": "eval_retrieval",
                "text_preview": str(item.get("matched_text", ""))[:600],
                "metadata": {
                    "evidence_id": item.get("evidence_id"),
                    "year": item.get("year"),
                    "page_start": source.get("page"),
                    "citation": item.get("citation"),
                },
            }
        )
    return candidates


def evaluate_generation_dataframe(
    frame: pd.DataFrame,
    retrieval_runner: RetrievalCallable,
    generator: AnswerGenerator,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, source in tqdm(frame.iterrows(), total=len(frame), desc="generation eval", unit="question"):
        started = time.perf_counter()
        base = source.to_dict()
        expected = parse_expected_keywords(source["expected_keywords"])
        try:
            rewritten = rewrite_query(str(source["question"]))
            retrieval_output = retrieval_runner(str(source["question"]), top_k)
            candidates = search_results_to_candidates(retrieval_output.get("results", []))
            generated = generator.generate(str(source["question"]), rewritten, candidates)
            output = generated.to_dict()
            answer = str(output.get("answer", ""))
            evidence_list = list(output.get("evidence_list", []))
            matched = matched_expected_keywords(
                expected,
                [{"title": "", "matched_text": answer, "matched_keywords": []}],
            )
            citation_result = check_citations(answer, evidence_list)
            guardrail_result = check_answer_guardrails(answer, evidence_list)
            row = {
                **base,
                "question_id": int(index) + 1,
                "normalized_query": rewritten.normalized_query,
                "detected_intent": rewritten.intent,
                "detected_risk_level": rewritten.risk_level,
                "llm_mode": generator.active_mode,
                "answer": answer,
                "answer_length": len(answer),
                "answer_matched_keywords": "|".join(matched),
                "answer_keyword_recall": len(matched) / len(expected) if expected else 0.0,
                "evidence_count": len(evidence_list),
                "evidence_ids": "|".join(str(item.get("evidence_id", "")) for item in evidence_list),
                "citation_count": len(citation_result.citations),
                "invalid_citations": "|".join(citation_result.invalid_citations),
                "unsupported_values": "|".join(citation_result.unsupported_values),
                "unsupported_titles": "|".join(citation_result.unsupported_titles),
                "citation_guard_passed": int(citation_result.passed),
                "answer_guardrail_passed": int(guardrail_result.passed),
                "guardrail_issues": "|".join([*citation_result.issues, *guardrail_result.violations]),
                "confidence": output.get("confidence", "low"),
                "need_human_review": bool(output.get("need_human_review", False) or guardrail_result.need_human_review),
                "limitations": "|".join(str(item) for item in output.get("limitations", [])),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            row = {
                **base,
                "question_id": int(index) + 1,
                "normalized_query": "",
                "detected_intent": "",
                "detected_risk_level": "",
                "llm_mode": generator.active_mode,
                "answer": "",
                "answer_length": 0,
                "answer_matched_keywords": "",
                "answer_keyword_recall": 0.0,
                "evidence_count": 0,
                "evidence_ids": "",
                "citation_count": 0,
                "invalid_citations": "",
                "unsupported_values": "",
                "unsupported_titles": "",
                "citation_guard_passed": 0,
                "answer_guardrail_passed": 0,
                "guardrail_issues": "",
                "confidence": "low",
                "need_human_review": True,
                "limitations": "",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        rows.append(row)
    return pd.DataFrame(rows)


def run_generation_evaluation(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    top_k: int = 5,
    llm_mode: str = "ollama",
    limit: int | None = None,
    retrieval_runner: RetrievalCallable | None = None,
    generator: AnswerGenerator | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    validate_questions(frame)
    if limit is not None:
        frame = frame.head(limit)
    runner = retrieval_runner or LocalRetrievalRunner(search_type="hybrid", rerank=True)
    answer_generator = generator or AnswerGenerator(mode=llm_mode, review_queue=DEFAULT_REVIEW_QUEUE)
    result = evaluate_generation_dataframe(frame, runner, answer_generator, top_k=top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print_generation_summary(result, output_path)
    return result


def print_generation_summary(result: pd.DataFrame, output_path: Path) -> None:
    successful = result[result["status"] == "ok"]
    print(f"questions: {len(result)}")
    print(f"successful: {len(successful)}")
    print(f"errors: {len(result) - len(successful)}")
    print(
        f"mean_answer_keyword_recall: {successful['answer_keyword_recall'].mean():.4f}"
        if len(successful)
        else "mean_answer_keyword_recall: 0.0000"
    )
    print(
        f"citation_guard_pass_rate: {successful['citation_guard_passed'].mean():.4f}"
        if len(successful)
        else "citation_guard_pass_rate: 0.0000"
    )
    print(f"human_review_count: {int(successful['need_human_review'].sum()) if len(successful) else 0}")
    print(f"output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local RAG answer generation against eval_questions.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--llm_mode", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_generation_evaluation(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        llm_mode=args.llm_mode,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

