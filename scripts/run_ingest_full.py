from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.index.build_chunks import build_chunks  # noqa: E402
from src.ingest.clean_sections import clean_paper, read_parsed_docs, write_report, write_sections  # noqa: E402
from src.ingest.extract_cards import extract_cards  # noqa: E402
from src.ingest.parse_papers import parse_pdf_pages  # noqa: E402
from src.ingest.scan_papers import scan_papers, write_inventory  # noqa: E402


@dataclass(frozen=True)
class FullIngestPaths:
    inventory: Path = PROJECT_ROOT / "data/metadata/full_paper_inventory.csv"
    parsed_docs: Path = PROJECT_ROOT / "data/interim/full_parsed_docs.jsonl"
    parsed_parts: Path = PROJECT_ROOT / "data/interim/full_parsed_parts"
    parse_errors: Path = PROJECT_ROOT / "data/interim/full_parse_errors.csv"
    cleaned_sections: Path = PROJECT_ROOT / "data/processed/full_cleaned_sections.jsonl"
    clean_report: Path = PROJECT_ROOT / "data/processed/full_clean_report.md"
    paper_cards: Path = PROJECT_ROOT / "data/processed/full_paper_cards.jsonl"
    defect_cards: Path = PROJECT_ROOT / "data/processed/full_defect_cards.jsonl"
    method_cards: Path = PROJECT_ROOT / "data/processed/full_method_cards.jsonl"
    parameter_cards: Path = PROJECT_ROOT / "data/processed/full_parameter_cards.jsonl"
    review_queue: Path = PROJECT_ROOT / "data/manual_review/full_review_queue.csv"
    chunks: Path = PROJECT_ROOT / "data/chunks/full_chunks.jsonl"
    chunk_report: Path = PROJECT_ROOT / "data/chunks/full_chunk_report.md"
    ingest_report: Path = PROJECT_ROOT / "data/logs/full_ingest_report.md"


ERROR_FIELDS = ["file_name", "error_type", "error_message"]


def atomic_write_part(part_path: Path, records: list[dict[str, Any]]) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = part_path.with_name(f".{part_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, part_path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_pdf_to_part(pdf_path: str, paper_id: str, parts_dir: str) -> dict[str, Any]:
    source = Path(pdf_path)
    part_path = Path(parts_dir) / f"{paper_id}.jsonl"
    try:
        records = parse_pdf_pages(source, paper_id)
        if not records:
            raise ValueError("PDF produced no page records.")
        atomic_write_part(part_path, records)
        return {
            "paper_id": paper_id,
            "file_name": source.name,
            "success": True,
            "pages": len(records),
            "text_chars": sum(int(record.get("text_length", 0)) for record in records),
        }
    except Exception as exc:
        return {
            "paper_id": paper_id,
            "file_name": source.name,
            "success": False,
            "pages": 0,
            "text_chars": 0,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
        }


def pdf_path_map(input_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in input_dir.rglob("*.pdf")
        if path.is_file()
    }


def is_complete_record_group(records: list[dict[str, Any]], pdf_path: Path) -> bool:
    page_numbers = sorted(int(record.get("page_num", 0)) for record in records)
    if not page_numbers or page_numbers != list(range(1, len(page_numbers) + 1)):
        return False
    try:
        with fitz.open(pdf_path) as document:
            return document.page_count == len(records)
    except Exception:
        return False


def migrate_existing_parsed_output(
    parsed_docs: Path,
    parts_dir: Path,
    input_dir: Path,
) -> int:
    if not parsed_docs.is_file() or parsed_docs.stat().st_size == 0:
        return 0
    paths_by_name = pdf_path_map(input_dir)
    migrated = 0
    current_paper_id: str | None = None
    current_records: list[dict[str, Any]] = []
    closed_paper_ids: set[str] = set()

    def flush_group() -> None:
        nonlocal migrated, current_records, current_paper_id
        if not current_paper_id or not current_records:
            return
        if current_paper_id in closed_paper_ids:
            raise ValueError(f"Existing parsed JSONL is not grouped by paper_id: {current_paper_id}")
        closed_paper_ids.add(current_paper_id)
        file_name = str(current_records[0].get("file_name", ""))
        pdf_path = paths_by_name.get(file_name)
        part_path = parts_dir / f"{current_paper_id}.jsonl"
        if part_path.exists() or pdf_path is None:
            return
        if is_complete_record_group(current_records, pdf_path):
            atomic_write_part(part_path, current_records)
            migrated += 1

    with parsed_docs.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            paper_id = str(record.get("paper_id", ""))
            if current_paper_id is None:
                current_paper_id = paper_id
            if paper_id != current_paper_id:
                flush_group()
                current_paper_id = paper_id
                current_records = []
            current_records.append(record)
    flush_group()
    return migrated


def write_parse_errors(path: Path, failures: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "file_name": failure["file_name"],
                    "error_type": failure.get("error_type", "UnknownError"),
                    "error_message": failure.get("error_message", ""),
                }
            )


def materialize_parsed_docs(
    records: list[dict[str, str]],
    parts_dir: Path,
    output_path: Path,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    papers = 0
    pages = 0
    try:
        with temporary.open("wb") as output:
            for record in records:
                part_path = parts_dir / f"{record['paper_id']}.jsonl"
                if not part_path.is_file():
                    continue
                with part_path.open("rb") as source:
                    for line in source:
                        if line.strip():
                            pages += 1
                        output.write(line)
                papers += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return papers, pages


def clear_full_outputs(paths: FullIngestPaths) -> None:
    safe_root = (PROJECT_ROOT / "data").resolve()
    for path in paths.__dict__.values():
        resolved = Path(path).resolve()
        if path == paths.parsed_parts:
            if path.exists():
                if resolved.parent != (PROJECT_ROOT / "data/interim").resolve() or path.name != "full_parsed_parts":
                    raise RuntimeError(f"Refusing to remove unexpected parts directory: {path}")
                shutil.rmtree(path)
            continue
        try:
            resolved.relative_to(safe_root)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to remove output outside data/: {path}") from exc
        path.unlink(missing_ok=True)


def suppress_module_output(operation):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return operation()


def run_parse_stage(
    records: list[dict[str, str]],
    input_dir: Path,
    paths: FullIngestPaths,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    paths.parsed_parts.mkdir(parents=True, exist_ok=True)
    migrated = migrate_existing_parsed_output(paths.parsed_docs, paths.parsed_parts, input_dir) if resume else 0
    paths_by_name = pdf_path_map(input_dir)
    completed_ids = {part.stem for part in paths.parsed_parts.glob("*.jsonl") if part.is_file()}
    pending = [record for record in records if record["paper_id"] not in completed_ids]
    failures: list[dict[str, Any]] = []
    parsed_now = 0
    pages_now = 0
    with tqdm(total=len(records), initial=len(records) - len(pending), desc="parse papers", unit="paper") as progress:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    parse_pdf_to_part,
                    str(paths_by_name[record["file_name"]]),
                    record["paper_id"],
                    str(paths.parsed_parts),
                ): record
                for record in pending
            }
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    parsed_now += 1
                    pages_now += int(result["pages"])
                else:
                    failures.append(result)
                progress.update(1)
    write_parse_errors(paths.parse_errors, failures)
    materialized_papers, materialized_pages = materialize_parsed_docs(records, paths.parsed_parts, paths.parsed_docs)
    return {
        "migrated_existing": migrated,
        "resumed_papers": len(records) - len(pending),
        "parsed_now": parsed_now,
        "pages_now": pages_now,
        "failed_papers": len(failures),
        "materialized_papers": materialized_papers,
        "materialized_pages": materialized_pages,
    }


def run_clean_stage(paths: FullIngestPaths) -> tuple[list[Any], int]:
    papers = read_parsed_docs(paths.parsed_docs)
    results = [
        clean_paper(paper_id, pages)
        for paper_id, pages in tqdm(papers.items(), desc="clean papers", unit="paper")
    ]
    write_sections(results, paths.cleaned_sections)
    write_report(results, paths.clean_report)
    return results, sum(len(result.sections) for result in results)


def write_ingest_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Full Corpus Ingest Report",
        "",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Failed step: {summary.get('failed_step', '') or 'none'}",
        f"- Raw inventory papers: {summary.get('inventory_papers', 0)}",
        f"- Target papers for this run: {summary.get('target_papers', 0)}",
        f"- Parsed/materialized papers: {summary.get('processed_papers', 0)}",
        f"- Parse failures: {summary.get('failed_papers', 0)}",
        f"- Parsed pages: {summary.get('parsed_pages', 0)}",
        f"- Sections: {summary.get('sections', 0)}",
        f"- Paper cards: {summary.get('paper_cards', 0)}",
        f"- Defect cards: {summary.get('defect_cards', 0)}",
        f"- Method cards: {summary.get('method_cards', 0)}",
        f"- Parameter cards: {summary.get('parameter_cards', 0)}",
        f"- Chunks: {summary.get('chunks', 0)}",
        f"- Resume enabled: {summary.get('resume', False)}",
        f"- Workers: {summary.get('workers', 0)}",
        f"- Elapsed seconds: {summary.get('elapsed_seconds', 0):.2f}",
        "",
        "## Outputs",
        "",
        "- Parsed docs: `data/interim/full_parsed_docs.jsonl`",
        "- Parse errors: `data/interim/full_parse_errors.csv`",
        "- Cleaned sections: `data/processed/full_cleaned_sections.jsonl`",
        "- Paper cards: `data/processed/full_paper_cards.jsonl`",
        "- Chunks: `data/chunks/full_chunks.jsonl`",
        "- Chunk report: `data/chunks/full_chunk_report.md`",
    ]
    if summary.get("error"):
        lines.extend(["", "## Error", "", f"- {summary['error']}"])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_full_ingest(args: argparse.Namespace, paths: FullIngestPaths | None = None) -> dict[str, Any]:
    paths = paths or FullIngestPaths()
    input_dir = args.input.resolve()
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "status": "running",
        "failed_step": "",
        "resume": args.resume,
        "workers": args.workers,
    }
    current_step = "preflight"
    try:
        if args.force:
            clear_full_outputs(paths)
        elif not args.resume and (paths.parsed_docs.exists() or paths.parsed_parts.exists()):
            raise RuntimeError("Full ingest outputs already exist. Use --resume or --force.")

        current_step = "scan"
        print("[scan] inventory")
        inventory = scan_papers(input_dir)
        write_inventory(inventory, paths.inventory)
        target_records = inventory[: args.limit] if args.limit is not None else inventory
        summary["inventory_papers"] = len(inventory)
        summary["target_papers"] = len(target_records)

        current_step = "parse"
        parse_stats = run_parse_stage(target_records, input_dir, paths, args.workers, args.resume)
        summary.update(parse_stats)
        summary["processed_papers"] = parse_stats["materialized_papers"]
        summary["parsed_pages"] = parse_stats["materialized_pages"]

        current_step = "clean"
        clean_results, section_count = run_clean_stage(paths)
        summary["sections"] = section_count
        del clean_results

        current_step = "extract_paper_cards"
        print("[extract_paper_cards] rules")
        card_stats = suppress_module_output(
            lambda: extract_cards(
                input_path=paths.cleaned_sections,
                paper_cards_path=paths.paper_cards,
                defect_cards_path=paths.defect_cards,
                method_cards_path=paths.method_cards,
                parameter_cards_path=paths.parameter_cards,
                review_queue_path=paths.review_queue,
            )
        )
        summary.update(card_stats)

        current_step = "build_chunks"
        print("[build_chunks] section-aware")
        chunks = suppress_module_output(
            lambda: build_chunks(
                sections_path=paths.cleaned_sections,
                paper_cards_path=paths.paper_cards,
                defect_cards_path=paths.defect_cards,
                method_cards_path=paths.method_cards,
                parameter_cards_path=paths.parameter_cards,
                output_path=paths.chunks,
                report_path=paths.chunk_report,
            )
        )
        summary["chunks"] = len(chunks)
        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failed_step"] = current_step
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_ingest_report(paths.ingest_report, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable full-corpus ingest from raw PDFs through chunks.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw_papers")
    parser.add_argument("--limit", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero.")
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8.")
    return args


def print_summary(summary: dict[str, Any]) -> None:
    print(f"processed_papers: {summary.get('processed_papers', 0)}")
    print(f"failed_papers: {summary.get('failed_papers', 0)}")
    print(f"sections: {summary.get('sections', 0)}")
    print(f"chunks: {summary.get('chunks', 0)}")
    print(f"elapsed_seconds: {summary.get('elapsed_seconds', 0):.2f}")


def main() -> int:
    args = parse_args()
    try:
        summary = run_full_ingest(args)
    except Exception as exc:
        print(f"full_ingest_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
