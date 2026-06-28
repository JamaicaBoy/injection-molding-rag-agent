import csv
import json
from pathlib import Path

from src.ingest.extract_cards import extract_cards


def write_sections(path: Path) -> None:
    records = [
        {
            "paper_id": "paper_1",
            "file_name": "Smith - 2025 - Quality prediction of ABS injection molding.pdf",
            "section_name": "Abstract",
            "section_order": 1,
            "clean_text": (
                "Abstract. This study uses BP neural network and response surface optimization "
                "for ABS injection molding. Melt temperature and cooling time affect warpage "
                "and shrinkage. The experiment uses Moldflow simulation data."
            ),
            "char_count": 220,
            "page_start": 1,
            "page_end": 1,
            "is_reference_section": False,
        },
        {
            "paper_id": "paper_1",
            "file_name": "Smith - 2025 - Quality prediction of ABS injection molding.pdf",
            "section_name": "Conclusion",
            "section_order": 2,
            "clean_text": (
                "Conclusion. Optimized melt temperature can reduce warpage. "
                "The method improves quality prediction accuracy."
            ),
            "char_count": 120,
            "page_start": 5,
            "page_end": 5,
            "is_reference_section": False,
        },
        {
            "paper_id": "paper_2",
            "file_name": "Unknown - 2024 - General plastic note.pdf",
            "section_name": "Unknown",
            "section_order": 1,
            "clean_text": "A short note with weak evidence.",
            "char_count": 32,
            "page_start": 1,
            "page_end": 1,
            "is_reference_section": False,
        },
    ]
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_extract_cards_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "cleaned_sections.jsonl"
    write_sections(input_path)

    stats = extract_cards(
        input_path=input_path,
        paper_cards_path=tmp_path / "paper_cards.jsonl",
        defect_cards_path=tmp_path / "defect_cards.jsonl",
        method_cards_path=tmp_path / "method_cards.jsonl",
        parameter_cards_path=tmp_path / "parameter_cards.jsonl",
        review_queue_path=tmp_path / "review_queue.csv",
    )

    paper_cards = read_jsonl(tmp_path / "paper_cards.jsonl")
    defect_cards = read_jsonl(tmp_path / "defect_cards.jsonl")
    method_cards = read_jsonl(tmp_path / "method_cards.jsonl")
    parameter_cards = read_jsonl(tmp_path / "parameter_cards.jsonl")

    assert stats["paper_cards"] == 2
    assert any(card["defect_type"] == "warpage" for card in defect_cards)
    assert any(card["method_name"] == "BP neural network" for card in method_cards)
    assert any(card["parameter_name"] == "melt temperature" for card in parameter_cards)
    assert all(len(card["evidence_text"]) < 1000 for card in defect_cards)
    assert paper_cards[0]["title"] == "Quality prediction of ABS injection molding"

    with (tmp_path / "review_queue.csv").open("r", encoding="utf-8-sig", newline="") as file:
        review_rows = list(csv.DictReader(file))
    assert review_rows
