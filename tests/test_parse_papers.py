import json
from pathlib import Path

import fitz

from src.ingest.parse_papers import parse_papers


def create_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_parse_papers_smoke_first_three(tmp_path: Path) -> None:
    input_dir = tmp_path / "dev_papers"
    input_dir.mkdir()
    for index in range(5):
        create_pdf(input_dir / f"paper_{index}.pdf", f"Smoke test page {index}")

    output = tmp_path / "parsed_docs.jsonl"
    errors = tmp_path / "parse_errors.csv"
    stats = parse_papers(
        input_dir=input_dir,
        output=output,
        errors=errors,
        max_files=3,
        metadata_files=(),
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    first_record = json.loads(lines[0])

    assert stats.total_papers == 3
    assert stats.successful_papers == 3
    assert stats.failed_papers == 0
    assert lines
    assert {"paper_id", "file_name", "page_num", "raw_text", "text_length"}.issubset(first_record)
    assert first_record["text_length"] > 0
