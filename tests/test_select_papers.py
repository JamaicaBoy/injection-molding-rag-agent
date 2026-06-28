import csv
from pathlib import Path

from src.ingest.select_papers import match_categories, run_selection


def write_inventory(project_root: Path, rows: list[dict[str, str]]) -> Path:
    inventory_csv = project_root / "data" / "metadata" / "paper_inventory.csv"
    inventory_csv.parent.mkdir(parents=True)
    with inventory_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return inventory_csv


def make_row(project_root: Path, paper_id: str, file_name: str, tags: str = "") -> dict[str, str]:
    raw_dir = project_root / "data" / "raw_papers"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / file_name
    file_path.write_bytes(b"%PDF-1.4 fake fixture")
    return {
        "paper_id": paper_id,
        "file_name": file_name,
        "file_path": file_path.relative_to(project_root).as_posix(),
        "file_size_mb": "0.01",
        "modified_time": "2026-01-01T00:00:00",
        "title_guess": Path(file_name).stem,
        "year_guess": "2025",
        "language_guess": "en",
        "keyword_tags_guess": tags,
        "selected_stage": "raw",
    }


def test_match_categories_from_filename_and_metadata() -> None:
    record = {
        "file_name": "Quality prediction with sensor data for ABS injection molding.pdf",
        "title_guess": "",
        "keyword_tags_guess": "quality prediction; ABS; injection molding",
    }

    scores = match_categories(record)

    assert "质量预测" in scores
    assert "材料和场景" in scores


def test_run_selection_writes_csv_and_copies_files(tmp_path: Path) -> None:
    category_files = [
        "Warpage shrinkage defect in injection molding 2025.pdf",
        "Melt temperature and cooling time process parameter 2025.pdf",
        "Quality prediction process monitoring sensor 2025.pdf",
        "GA PSO DOE response surface optimization 2025.pdf",
        "Machine learning digital twin RAG LLM 2025.pdf",
        "ABS PP PC PMMA micro injection molding Moldflow CAE 2025.pdf",
    ]
    rows = []
    for index in range(100):
        file_name = f"{index:03d} - {category_files[index % len(category_files)]}"
        rows.append(make_row(tmp_path, f"paper_{index:03d}", file_name))
    inventory_csv = write_inventory(tmp_path, rows)

    selected_records, dev_records = run_selection(
        inventory_csv=inventory_csv,
        output_csv=tmp_path / "data" / "metadata" / "selected_papers.csv",
        dev_dir=tmp_path / "data" / "dev_papers",
        selected_dir=tmp_path / "data" / "selected_papers",
        selected_count=60,
        dev_count=24,
        project_root=tmp_path,
    )

    assert len(selected_records) == 60
    assert len(dev_records) == 24
    assert len(list((tmp_path / "data" / "selected_papers").glob("*.pdf"))) == 60
    assert len(list((tmp_path / "data" / "dev_papers").glob("*.pdf"))) == 24

    with (tmp_path / "data" / "metadata" / "selected_papers.csv").open("r", encoding="utf-8-sig") as file:
        output_rows = list(csv.DictReader(file))
    assert len(output_rows) == 60
    assert {"primary_category", "matched_categories", "in_dev_set"}.issubset(output_rows[0])
