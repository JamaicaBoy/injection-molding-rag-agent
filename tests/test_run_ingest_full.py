import json
from pathlib import Path

import fitz

from scripts.run_ingest_full import materialize_parsed_docs, migrate_existing_parsed_output
from src.ingest.parse_papers import parse_pdf_pages


def test_existing_complete_parse_migrates_to_resumable_part(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    pdf_path = input_dir / "paper.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "page one")
    document.new_page().insert_text((72, 72), "page two")
    document.save(pdf_path)
    document.close()

    paper_id = "paper_test"
    parsed_docs = tmp_path / "full_parsed_docs.jsonl"
    records = parse_pdf_pages(pdf_path, paper_id)
    with parsed_docs.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    parts_dir = tmp_path / "parts"
    migrated = migrate_existing_parsed_output(parsed_docs, parts_dir, input_dir)
    output = tmp_path / "materialized.jsonl"
    papers, pages = materialize_parsed_docs(
        [{"paper_id": paper_id, "file_name": pdf_path.name}],
        parts_dir,
        output,
    )

    assert migrated == 1
    assert (parts_dir / f"{paper_id}.jsonl").is_file()
    assert papers == 1
    assert pages == 2
    assert sum(1 for line in output.open(encoding="utf-8") if line.strip()) == 2
