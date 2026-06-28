from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from src.agent.tools import search_papers_tool
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.reranker import Reranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "eval_questions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.csv"
REQUIRED_COLUMNS = {"question", "intent", "expected_keywords", "difficulty", "risk_level", "notes"}

RetrievalCallable = Callable[[str, int], dict[str, Any]]


def parse_expected_keywords(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def matched_expected_keywords(expected: list[str], results: list[dict[str, Any]]) -> list[str]:
    corpus = " ".join(
        " ".join(
            (
                str(item.get("title", "")),
                str(item.get("matched_text", "")),
                " ".join(str(keyword) for keyword in item.get("matched_keywords", [])),
            )
        )
        for item in results
    ).lower().replace("_", " ")
    return [keyword for keyword in expected if keyword.lower().replace("_", " ") in corpus]


class LocalRetrievalRunner:
    def __init__(self, search_type: str = "hybrid", rerank: bool = True) -> None:
        self.search_type = search_type
        self.rerank = rerank
        self.bm25 = BM25Retriever() if search_type in {"keyword", "hybrid"} else None
        self.dense = DenseRetriever() if search_type in {"semantic", "hybrid"} else None
        self.reranker = Reranker(mode="rule")

    def __call__(self, question: str, top_k: int) -> dict[str, Any]:
        return search_papers_tool(
            query=question,
            search_type=self.search_type,
            filters={},
            top_k=top_k,
            rerank=self.rerank,
            return_chunks=True,
            language="auto",
            _bm25=self.bm25,
            _dense=self.dense,
            _reranker=self.reranker,
        )


def validate_questions(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"eval_questions.csv is missing columns: {sorted(missing)}")


def evaluate_retrieval_dataframe(
    frame: pd.DataFrame,
    retrieval_runner: RetrievalCallable,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, source in tqdm(frame.iterrows(), total=len(frame), desc="retrieval eval", unit="question"):
        started = time.perf_counter()
        base = source.to_dict()
        expected = parse_expected_keywords(source["expected_keywords"])
        try:
            rewritten = rewrite_query(str(source["question"]))
            output = retrieval_runner(str(source["question"]), top_k)
            results = output.get("results", [])
            matched = matched_expected_keywords(expected, results)
            scores = [float(item.get("rerank_score") or item.get("relevance_score") or 0.0) for item in results]
            row = {
                **base,
                "question_id": int(index) + 1,
                "normalized_query": rewritten.normalized_query,
                "detected_intent": rewritten.intent,
                "detected_risk_level": rewritten.risk_level,
                "top_k": top_k,
                "result_count": len(results),
                "hit_at_k": int(bool(matched)),
                "matched_keywords": "|".join(matched),
                "matched_keyword_count": len(matched),
                "expected_keyword_count": len(expected),
                "keyword_recall": len(matched) / len(expected) if expected else 0.0,
                "top_score": max(scores, default=0.0),
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "unique_paper_count": len({item.get("paper_id") for item in results if item.get("paper_id")}),
                "top_evidence_ids": "|".join(str(item.get("evidence_id", "")) for item in results),
                "top_paper_ids": "|".join(dict.fromkeys(str(item.get("paper_id", "")) for item in results)),
                "overall_confidence": float(output.get("overall_confidence", 0.0)),
                "warnings": "|".join(str(item) for item in output.get("warnings", [])),
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
                "top_k": top_k,
                "result_count": 0,
                "hit_at_k": 0,
                "matched_keywords": "",
                "matched_keyword_count": 0,
                "expected_keyword_count": len(expected),
                "keyword_recall": 0.0,
                "top_score": 0.0,
                "mean_score": 0.0,
                "unique_paper_count": 0,
                "top_evidence_ids": "",
                "top_paper_ids": "",
                "overall_confidence": 0.0,
                "warnings": "",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        rows.append(row)
    return pd.DataFrame(rows)


def run_retrieval_evaluation(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    top_k: int = 5,
    search_type: str = "hybrid",
    rerank: bool = True,
    limit: int | None = None,
    retrieval_runner: RetrievalCallable | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    validate_questions(frame)
    if limit is not None:
        frame = frame.head(limit)
    runner = retrieval_runner or LocalRetrievalRunner(search_type=search_type, rerank=rerank)
    result = evaluate_retrieval_dataframe(frame, runner, top_k=top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print_retrieval_summary(result, output_path)
    return result


def print_retrieval_summary(result: pd.DataFrame, output_path: Path) -> None:
    successful = result[result["status"] == "ok"]
    print(f"questions: {len(result)}")
    print(f"successful: {len(successful)}")
    print(f"errors: {len(result) - len(successful)}")
    print(f"hit_at_k: {successful['hit_at_k'].mean():.4f}" if len(successful) else "hit_at_k: 0.0000")
    print(
        f"mean_keyword_recall: {successful['keyword_recall'].mean():.4f}"
        if len(successful)
        else "mean_keyword_recall: 0.0000"
    )
    print(f"output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local paper retrieval against eval_questions.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--search_type", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_retrieval_evaluation(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        search_type=args.search_type,
        rerank=not args.no_rerank,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

