from pathlib import Path

from src.ingest.scan_papers import (
    CSV_FIELDS,
    guess_keyword_tags,
    guess_language,
    guess_year,
    scan_papers,
    write_inventory,
)


def test_scan_papers_uses_filename_metadata_only(tmp_path: Path) -> None:
    project_root = tmp_path
    raw_dir = project_root / "data" / "raw_papers"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "Chen 等 - 2025 - Quality Prediction of ABS Injection Molding Defects.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake test fixture")

    records = scan_papers(raw_dir, project_root=project_root)

    assert len(records) == 1
    record = records[0]
    assert set(CSV_FIELDS) == set(record)
    assert record["year_guess"] == "2025"
    assert record["language_guess"] == "zh"
    assert "quality prediction" in record["keyword_tags_guess"]
    assert "ABS" in record["keyword_tags_guess"]
    assert "injection molding" in record["keyword_tags_guess"]
    assert record["selected_stage"] == "raw"


def test_write_inventory_creates_csv(tmp_path: Path) -> None:
    records = [
        {
            field: ""
            for field in CSV_FIELDS
        }
    ]
    output_csv = tmp_path / "metadata" / "paper_inventory.csv"

    write_inventory(records, output_csv)

    csv_text = output_csv.read_text(encoding="utf-8-sig")
    assert csv_text.startswith(",".join(CSV_FIELDS))


def test_guess_helpers() -> None:
    file_name = "Wang - 2024 - Warpage and shrinkage optimization for PP.pdf"

    assert guess_year(file_name) == "2024"
    assert guess_language(file_name) == "en"
    assert {"warpage", "shrinkage", "optimization", "PP"}.issubset(set(guess_keyword_tags(file_name)))


def test_short_rag_keyword_does_not_match_name_fragments() -> None:
    assert "RAG" not in guess_keyword_tags("Brag 等 - 2023 - Development and Production.pdf")
    assert "RAG" in guess_keyword_tags("Li - 2025 - RAG for Injection Molding.pdf")
