from __future__ import annotations

import contextlib
import io
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PAPERS_DIR = PROJECT_ROOT / "data" / "raw_papers"
T = TypeVar("T")


class PipelineStepError(RuntimeError):
    def __init__(self, step: str, cause: Exception) -> None:
        super().__init__(f"{step}: {type(cause).__name__}: {cause}")
        self.step = step
        self.cause = cause


@dataclass(frozen=True)
class IngestPaths:
    parsed_docs: Path
    parse_errors: Path
    cleaned_sections: Path
    clean_report: Path
    paper_cards: Path
    defect_cards: Path
    method_cards: Path
    parameter_cards: Path
    review_queue: Path
    chunks: Path
    chunk_report: Path


def dev_paths() -> IngestPaths:
    return IngestPaths(
        parsed_docs=PROJECT_ROOT / "data/interim/parsed_docs.jsonl",
        parse_errors=PROJECT_ROOT / "data/interim/parse_errors.csv",
        cleaned_sections=PROJECT_ROOT / "data/processed/cleaned_sections.jsonl",
        clean_report=PROJECT_ROOT / "data/processed/clean_report.md",
        paper_cards=PROJECT_ROOT / "data/processed/paper_cards.jsonl",
        defect_cards=PROJECT_ROOT / "data/processed/defect_cards.jsonl",
        method_cards=PROJECT_ROOT / "data/processed/method_cards.jsonl",
        parameter_cards=PROJECT_ROOT / "data/processed/parameter_cards.jsonl",
        review_queue=PROJECT_ROOT / "data/manual_review/review_queue.csv",
        chunks=PROJECT_ROOT / "data/chunks/dev_chunks.jsonl",
        chunk_report=PROJECT_ROOT / "data/chunks/dev_chunks_report.md",
    )


def selected_paths() -> IngestPaths:
    return IngestPaths(
        parsed_docs=PROJECT_ROOT / "data/interim/selected_parsed_docs.jsonl",
        parse_errors=PROJECT_ROOT / "data/interim/selected_parse_errors.csv",
        cleaned_sections=PROJECT_ROOT / "data/processed/selected_cleaned_sections.jsonl",
        clean_report=PROJECT_ROOT / "data/processed/selected_clean_report.md",
        paper_cards=PROJECT_ROOT / "data/processed/selected_paper_cards.jsonl",
        defect_cards=PROJECT_ROOT / "data/processed/selected_defect_cards.jsonl",
        method_cards=PROJECT_ROOT / "data/processed/selected_method_cards.jsonl",
        parameter_cards=PROJECT_ROOT / "data/processed/selected_parameter_cards.jsonl",
        review_queue=PROJECT_ROOT / "data/manual_review/selected_review_queue.csv",
        chunks=PROJECT_ROOT / "data/chunks/selected_chunks.jsonl",
        chunk_report=PROJECT_ROOT / "data/chunks/selected_chunk_report.md",
    )


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def require_full_run_confirmation(input_dir: Path, confirmation: str | None) -> None:
    if input_dir.resolve() == RAW_PAPERS_DIR.resolve() and confirmation != "yes":
        raise ValueError(
            "Refusing to process data/raw_papers without "
            "--confirm_full_run yes. Use data/dev_papers or data/selected_papers for normal runs."
        )


def create_logger(command_name: str) -> tuple[logging.Logger, Path]:
    log_dir = PROJECT_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{command_name}_{stamp}.log"
    logger = logging.getLogger(f"pipeline.{command_name}.{stamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger, log_path


def run_step(step: str, operation: Callable[[], T], logger: logging.Logger) -> T:
    output = io.StringIO()
    started = time.perf_counter()
    logger.info("step_started | %s", step)
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = operation()
    except Exception as exc:
        _write_captured_output(logger, step, output.getvalue())
        logger.exception("step_failed | %s", step)
        raise PipelineStepError(step, exc) from exc
    _write_captured_output(logger, step, output.getvalue())
    logger.info("step_completed | %s | elapsed_seconds=%.2f", step, time.perf_counter() - started)
    return result


def _write_captured_output(logger: logging.Logger, step: str, captured: str) -> None:
    for line in captured.replace("\r", "\n").splitlines():
        if line.strip():
            logger.info("step_output | %s | %s", step, line.strip())


def run_ingest_pipeline(
    input_dir: Path,
    paths: IngestPaths,
    max_files: int | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    from src.index.build_chunks import build_chunks
    from src.ingest.clean_sections import run_cleaning
    from src.ingest.extract_cards import extract_cards
    from src.ingest.parse_papers import parse_papers

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    started = time.perf_counter()
    parse_stats = run_step(
        "parse_papers",
        lambda: parse_papers(
            input_dir=input_dir,
            output=paths.parsed_docs,
            errors=paths.parse_errors,
            max_files=max_files,
        ),
        logger,
    )
    clean_results = run_step(
        "clean_sections",
        lambda: run_cleaning(paths.parsed_docs, paths.cleaned_sections, paths.clean_report),
        logger,
    )
    card_stats = run_step(
        "extract_cards",
        lambda: extract_cards(
            input_path=paths.cleaned_sections,
            paper_cards_path=paths.paper_cards,
            defect_cards_path=paths.defect_cards,
            method_cards_path=paths.method_cards,
            parameter_cards_path=paths.parameter_cards,
            review_queue_path=paths.review_queue,
        ),
        logger,
    )
    chunks = run_step(
        "build_chunks",
        lambda: build_chunks(
            sections_path=paths.cleaned_sections,
            paper_cards_path=paths.paper_cards,
            defect_cards_path=paths.defect_cards,
            method_cards_path=paths.method_cards,
            parameter_cards_path=paths.parameter_cards,
            output_path=paths.chunks,
            report_path=paths.chunk_report,
        ),
        logger,
    )

    raw_chars = sum(item.raw_char_count for item in clean_results)
    clean_chars = sum(item.clean_char_count for item in clean_results)
    deleted_ratio = 0.0 if not raw_chars else max(raw_chars - clean_chars, 0) / raw_chars
    summary = {
        "input": display_path(input_dir),
        "papers": parse_stats.total_papers,
        "successful_papers": parse_stats.successful_papers,
        "failed_papers": parse_stats.failed_papers,
        "pages": parse_stats.total_pages,
        "sections": sum(len(item.sections) for item in clean_results),
        "section_recognition_rate": (
            sum(item.recognized_section_count > 0 for item in clean_results) / len(clean_results)
            if clean_results
            else 0.0
        ),
        "deleted_text_ratio": deleted_ratio,
        **card_stats,
        "chunks": len(chunks),
        "chunk_types": dict(Counter(str(item.get("chunk_type", "unknown")) for item in chunks)),
        "chunks_output": display_path(paths.chunks),
        "elapsed_seconds": time.perf_counter() - started,
    }
    logger.info("pipeline_summary | %s", summary)
    return summary


def print_ingest_summary(summary: dict[str, Any], log_path: Path) -> None:
    print("Pipeline completed.")
    print(f"input: {summary['input']}")
    print(
        "papers: "
        f"{summary['papers']} (success={summary['successful_papers']}, failed={summary['failed_papers']})"
    )
    print(f"pages: {summary['pages']}")
    print(f"sections: {summary['sections']}")
    print(f"section_recognition_rate: {summary['section_recognition_rate']:.2%}")
    print(f"deleted_text_ratio: {summary['deleted_text_ratio']:.2%}")
    print(
        "cards: "
        f"paper={summary['paper_cards']}, defect={summary['defect_cards']}, "
        f"method={summary['method_cards']}, parameter={summary['parameter_cards']}, "
        f"low_confidence={summary['low_confidence']}"
    )
    print(f"chunks: {summary['chunks']}")
    print(f"chunk_types: {summary['chunk_types']}")
    print(f"chunks_output: {summary['chunks_output']}")
    print(f"elapsed_seconds: {summary['elapsed_seconds']:.2f}")
    print(f"log: {display_path(log_path)}")


def print_pipeline_failure(error: Exception, log_path: Path) -> None:
    if isinstance(error, PipelineStepError):
        print(f"Pipeline failed at step: {error.step}", file=sys.stderr)
        print(f"error: {type(error.cause).__name__}: {error.cause}", file=sys.stderr)
    else:
        print("Pipeline failed before processing steps.", file=sys.stderr)
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
    print(f"log: {display_path(log_path)}", file=sys.stderr)
