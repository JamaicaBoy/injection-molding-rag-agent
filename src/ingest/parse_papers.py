from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "dev_papers"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "parsed_docs.jsonl"
DEFAULT_ERRORS = PROJECT_ROOT / "data" / "interim" / "parse_errors.csv"
DEFAULT_METADATA_FILES = (
    PROJECT_ROOT / "data" / "metadata" / "selected_papers.csv",
    PROJECT_ROOT / "data" / "metadata" / "paper_inventory.csv",
)

ERROR_FIELDS = ["file_name", "error_type", "error_message"]


@dataclass
class ParseStats:
    total_papers: int = 0
    successful_papers: int = 0
    failed_papers: int = 0
    total_pages: int = 0
    text_lengths: list[int] | None = None
    first_paper_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.text_lengths is None:
            self.text_lengths = []
        if self.first_paper_ids is None:
            self.first_paper_ids = []

    @property
    def average_chars_per_page(self) -> float:
        if not self.text_lengths:
            return 0.0
        return mean(self.text_lengths)


def stable_paper_id(file_name: str) -> str:
    digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()
    return f"paper_{digest[:12]}"


def load_paper_id_map(metadata_files: tuple[Path, ...] = DEFAULT_METADATA_FILES) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for metadata_file in metadata_files:
        if not metadata_file.exists():
            continue
        with metadata_file.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                file_name = row.get("file_name", "")
                paper_id = row.get("paper_id", "")
                if file_name and paper_id:
                    mapping.setdefault(file_name, paper_id)
    return mapping


def iter_pdf_paths(input_dir: Path, max_files: int | None = None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    pdf_paths = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )
    if max_files is not None:
        return pdf_paths[:max_files]
    return pdf_paths


def parse_pdf_pages(pdf_path: Path, paper_id: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            records.append(
                {
                    "paper_id": paper_id,
                    "file_name": pdf_path.name,
                    "page_num": page_index,
                    "raw_text": raw_text,
                    "text_length": len(raw_text),
                }
            )
    return records


def write_error_header(errors_csv: Path) -> None:
    errors_csv.parent.mkdir(parents=True, exist_ok=True)
    with errors_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=ERROR_FIELDS)
        writer.writeheader()


def append_error(errors_csv: Path, file_name: str, error: Exception) -> None:
    with errors_csv.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=ERROR_FIELDS)
        writer.writerow(
            {
                "file_name": file_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )


def parse_papers(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output: Path = DEFAULT_OUTPUT,
    errors: Path = DEFAULT_ERRORS,
    max_files: int | None = None,
    metadata_files: tuple[Path, ...] = DEFAULT_METADATA_FILES,
) -> ParseStats:
    pdf_paths = iter_pdf_paths(input_dir, max_files=max_files)
    paper_id_map = load_paper_id_map(metadata_files)
    stats = ParseStats(total_papers=len(pdf_paths))

    output.parent.mkdir(parents=True, exist_ok=True)
    write_error_header(errors)

    with output.open("w", encoding="utf-8") as output_file:
        for pdf_path in pdf_paths:
            paper_id = paper_id_map.get(pdf_path.name, stable_paper_id(pdf_path.name))
            try:
                page_records = parse_pdf_pages(pdf_path, paper_id)
            except Exception as exc:
                stats.failed_papers += 1
                append_error(errors, pdf_path.name, exc)
                continue

            stats.successful_papers += 1
            stats.total_pages += len(page_records)
            if len(stats.first_paper_ids) < 2:
                stats.first_paper_ids.append(paper_id)

            for record in page_records:
                stats.text_lengths.append(int(record["text_length"]))
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return stats


def print_stats(stats: ParseStats) -> None:
    print(f"总论文数: {stats.total_papers}")
    print(f"成功数: {stats.successful_papers}")
    print(f"失败数: {stats.failed_papers}")
    print(f"总页数: {stats.total_pages}")
    print(f"平均每页字数: {stats.average_chars_per_page:.2f}")
    print("前 2 个 paper_id:")
    for paper_id in stats.first_paper_ids[:2]:
        print(f"  {paper_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDF pages from the dev paper set only.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--max_files", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = parse_papers(
        input_dir=args.input_dir,
        output=args.output,
        errors=args.errors,
        max_files=args.max_files,
    )
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
