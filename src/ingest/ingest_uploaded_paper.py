from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.index.build_chunks import (
    build_text_chunks,
    split_oversized_chunks,
    write_jsonl,
    write_report,
)
from src.ingest.clean_sections import clean_paper, write_sections
from src.ingest.parse_papers import parse_pdf_pages, stable_paper_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"
SAFE_FILE_PATTERN = re.compile(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+")


@dataclass(frozen=True)
class UploadIngestStats:
    session_id: str
    uploaded_count: int
    parsed_count: int
    failed_count: int
    page_count: int
    section_count: int
    chunk_count: int
    chunks_path: str
    errors_path: str
    failures: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_session_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    if not cleaned:
        raise ValueError("session_id must contain letters or numbers")
    return cleaned[:80]


def safe_pdf_filename(file_name: str) -> str:
    base_name = Path(str(file_name).replace("\\", "/")).name
    stem = SAFE_FILE_PATTERN.sub("_", Path(base_name).stem).strip(" ._") or "uploaded_paper"
    if Path(base_name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")
    return f"{stem[:120]}.pdf"


def upload_paths(session_id: str, upload_root: Path = DEFAULT_UPLOAD_ROOT) -> dict[str, Path]:
    session_dir = Path(upload_root) / safe_session_id(session_id)
    return {
        "session_dir": session_dir,
        "raw_dir": session_dir / "raw",
        "parsed": session_dir / "interim" / "parsed_docs.jsonl",
        "errors": session_dir / "interim" / "parse_errors.csv",
        "sections": session_dir / "processed" / "cleaned_sections.jsonl",
        "clean_report": session_dir / "processed" / "clean_report.md",
        "chunks": session_dir / "chunks" / "chunks.jsonl",
        "chunk_report": session_dir / "chunks" / "chunk_report.md",
        "status": session_dir / "status.json",
    }


def save_uploaded_pdf(
    file_name: str,
    content: bytes,
    *,
    session_id: str,
    upload_root: Path = DEFAULT_UPLOAD_ROOT,
) -> Path:
    if not content.startswith(b"%PDF"):
        raise ValueError("Uploaded file does not have a valid PDF header")
    paths = upload_paths(session_id, upload_root)
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    safe_name = safe_pdf_filename(file_name)
    destination = paths["raw_dir"] / safe_name
    counter = 2
    while destination.exists() and destination.read_bytes() != content:
        destination = paths["raw_dir"] / f"{Path(safe_name).stem}_{counter}.pdf"
        counter += 1
    destination.write_bytes(content)
    return destination


def _write_parsed(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_errors(errors_path: Path, failures: list[dict[str, str]]) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["file_name", "error_type", "error_message"],
        )
        writer.writeheader()
        writer.writerows(failures)


def ingest_uploaded_papers(
    session_id: str,
    *,
    pdf_paths: list[Path] | None = None,
    upload_root: Path = DEFAULT_UPLOAD_ROOT,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> UploadIngestStats:
    paths = upload_paths(session_id, upload_root)
    candidates = pdf_paths or sorted(paths["raw_dir"].glob("*.pdf"))
    candidates = [Path(path) for path in candidates]
    if not candidates:
        raise ValueError("No uploaded PDF files found")

    parsed_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    parsed_count = 0
    if progress:
        progress("parsing", {"uploaded_count": len(candidates)})
    for pdf_path in candidates:
        try:
            safe_name = safe_pdf_filename(pdf_path.name)
            paper_id = stable_paper_id(f"upload:{safe_session_id(session_id)}:{safe_name}")
            pages = parse_pdf_pages(pdf_path, paper_id)
            if not pages:
                raise ValueError("PDF contains no pages")
            parsed_records.extend(pages)
            parsed_count += 1
        except Exception as exc:
            failures.append(
                {
                    "file_name": pdf_path.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                }
            )
    _write_parsed(parsed_records, paths["parsed"])
    _write_errors(paths["errors"], failures)

    if not parsed_records:
        raise RuntimeError("All uploaded PDFs failed to parse")
    if progress:
        progress("cleaning", {"page_count": len(parsed_records)})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in parsed_records:
        grouped.setdefault(str(record["paper_id"]), []).append(record)
    clean_results = [clean_paper(paper_id, pages) for paper_id, pages in grouped.items()]
    write_sections(clean_results, paths["sections"])
    paths["clean_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["clean_report"].write_text(
        "# Uploaded Paper Clean Summary\n\n"
        f"- papers: {len(clean_results)}\n"
        f"- sections: {sum(len(item.sections) for item in clean_results)}\n"
        f"- failures: {len(failures)}\n",
        encoding="utf-8",
    )

    sections = [section for result in clean_results for section in result.sections]
    chunks = split_oversized_chunks(build_text_chunks(sections, {}))
    write_jsonl(paths["chunks"], chunks)
    write_report(paths["chunk_report"], chunks)
    stats = UploadIngestStats(
        session_id=safe_session_id(session_id),
        uploaded_count=len(candidates),
        parsed_count=parsed_count,
        failed_count=len(failures),
        page_count=len(parsed_records),
        section_count=len(sections),
        chunk_count=len(chunks),
        chunks_path=str(paths["chunks"]),
        errors_path=str(paths["errors"]),
        failures=failures,
    )
    paths["status"].write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if progress:
        progress("chunked", {"chunk_count": len(chunks), "failed_count": len(failures)})
    return stats
