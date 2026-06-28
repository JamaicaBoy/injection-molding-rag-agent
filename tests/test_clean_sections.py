import json
from pathlib import Path

from src.ingest.clean_sections import detect_section_heading, run_cleaning


def write_parsed_jsonl(path: Path) -> None:
    records = [
        {
            "paper_id": "paper_a",
            "file_name": "sample.pdf",
            "page_num": 1,
            "raw_text": "\n".join(
                [
                    "Journal Header",
                    "1",
                    "Abstract",
                    "This paper studies injection-",
                    "molding defects.",
                    "Figure 1. Defect sample",
                    "Introduction",
                    "Injection molding quality matters.",
                ]
            ),
            "text_length": 180,
        },
        {
            "paper_id": "paper_a",
            "file_name": "sample.pdf",
            "page_num": 2,
            "raw_text": "\n".join(
                [
                    "Journal Header",
                    "2",
                    "Method",
                    "We use sensor data.",
                    "References",
                    "[1] Reference item.",
                ]
            ),
            "text_length": 120,
        },
    ]
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_detect_section_heading_english_and_chinese() -> None:
    assert detect_section_heading("1. Introduction") == "Introduction"
    assert detect_section_heading("参考文献") == "References"
    assert detect_section_heading("Results and Discussion") == "Results"


def test_run_cleaning_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "parsed_docs.jsonl"
    output_path = tmp_path / "cleaned_sections.jsonl"
    report_path = tmp_path / "clean_report.md"
    write_parsed_jsonl(input_path)

    results = run_cleaning(input_path=input_path, output_path=output_path, report_path=report_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    sections = [json.loads(line) for line in lines]

    assert results
    assert lines
    assert any(section["section_name"] == "Abstract" for section in sections)
    assert any(section["is_reference_section"] for section in sections)
    assert any("injection molding" in section["clean_text"] for section in sections)
    assert any("Figure 1. Defect sample" in section["clean_text"] for section in sections)
    assert "Section recognition success rate" in report_path.read_text(encoding="utf-8")
