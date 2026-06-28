from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_common import (  # noqa: E402
    create_logger,
    display_path,
    print_pipeline_failure,
    resolve_project_path,
    run_step,
)
from src.eval.eval_generation import run_generation_evaluation  # noqa: E402
from src.eval.eval_retrieval import LocalRetrievalRunner, run_retrieval_evaluation  # noqa: E402
from src.rag.answer_generator import AnswerGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval and answer-generation evaluations.")
    parser.add_argument("--input", type=Path, default=Path("data/eval/eval_questions.csv"))
    parser.add_argument("--retrieval_output", type=Path, default=Path("data/eval/retrieval_eval.csv"))
    parser.add_argument("--generation_output", type=Path, default=Path("data/eval/generation_eval.csv"))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm_mode", choices=["mock", "ollama"], default="mock")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger, log_path = create_logger("run_eval")
    try:
        input_path = resolve_project_path(args.input)
        retrieval_output = resolve_project_path(args.retrieval_output)
        generation_output = resolve_project_path(args.generation_output)
        runner = run_step(
            "initialize_retrieval",
            lambda: LocalRetrievalRunner(search_type="hybrid", rerank=True),
            logger,
        )
        retrieval = run_step(
            "retrieval_evaluation",
            lambda: run_retrieval_evaluation(
                input_path=input_path,
                output_path=retrieval_output,
                top_k=args.top_k,
                limit=args.limit,
                retrieval_runner=runner,
            ),
            logger,
        )
        generator = AnswerGenerator(
            mode=args.llm_mode,
            review_queue=PROJECT_ROOT / "data/eval/generation_review_queue.csv",
        )
        generation = run_step(
            "generation_evaluation",
            lambda: run_generation_evaluation(
                input_path=input_path,
                output_path=generation_output,
                top_k=args.top_k,
                llm_mode=args.llm_mode,
                limit=args.limit,
                retrieval_runner=runner,
                generator=generator,
            ),
            logger,
        )
    except Exception as exc:
        logger.exception("pipeline_failed")
        print_pipeline_failure(exc, log_path)
        return 1

    retrieval_ok = retrieval[retrieval["status"] == "ok"]
    generation_ok = generation[generation["status"] == "ok"]
    print("Evaluation completed.")
    print(f"questions: {len(retrieval)}")
    print(f"retrieval_errors: {len(retrieval) - len(retrieval_ok)}")
    print(f"hit_at_k: {retrieval_ok['hit_at_k'].mean():.4f}" if len(retrieval_ok) else "hit_at_k: 0.0000")
    print(
        f"retrieval_keyword_recall: {retrieval_ok['keyword_recall'].mean():.4f}"
        if len(retrieval_ok)
        else "retrieval_keyword_recall: 0.0000"
    )
    print(f"generation_mode: {generator.active_mode}")
    print(f"generation_errors: {len(generation) - len(generation_ok)}")
    print(
        f"generation_keyword_recall: {generation_ok['answer_keyword_recall'].mean():.4f}"
        if len(generation_ok)
        else "generation_keyword_recall: 0.0000"
    )
    print(
        f"citation_guard_pass_rate: {generation_ok['citation_guard_passed'].mean():.4f}"
        if len(generation_ok)
        else "citation_guard_pass_rate: 0.0000"
    )
    print(f"human_review_count: {int(generation_ok['need_human_review'].sum()) if len(generation_ok) else 0}")
    print(f"retrieval_output: {display_path(retrieval_output)}")
    print(f"generation_output: {display_path(generation_output)}")
    print(f"log: {display_path(log_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
