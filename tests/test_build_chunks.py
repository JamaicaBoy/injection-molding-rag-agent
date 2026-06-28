import json
from pathlib import Path

from src.index.build_chunks import build_chunks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_chunks_section_aware_smoke(tmp_path: Path) -> None:
    sections = [
        {
            "paper_id": "paper_1",
            "file_name": "Smith - 2025 - ABS injection molding.pdf",
            "section_name": "Abstract",
            "section_order": 1,
            "clean_text": "Abstract. This paper studies ABS injection molding warpage. " * 20,
            "char_count": 1200,
            "page_start": 1,
            "page_end": 1,
            "is_reference_section": False,
        },
        {
            "paper_id": "paper_1",
            "file_name": "Smith - 2025 - ABS injection molding.pdf",
            "section_name": "Results",
            "section_order": 2,
            "clean_text": (
                "Results show reduced warpage.\n"
                "Figure 1. Warpage contour from Moldflow.\n"
                "The optimized melt temperature improves quality.\n"
                "Table 1. Process parameter setting.\n"
                "Cooling time and packing pressure are important."
            ),
            "char_count": 220,
            "page_start": 2,
            "page_end": 3,
            "is_reference_section": False,
        },
        {
            "paper_id": "paper_1",
            "file_name": "Smith - 2025 - ABS injection molding.pdf",
            "section_name": "References",
            "section_order": 3,
            "clean_text": "References should not be chunked.",
            "char_count": 34,
            "page_start": 4,
            "page_end": 4,
            "is_reference_section": True,
        },
    ]
    paper_cards = [
        {
            "paper_id": "paper_1",
            "title": "ABS injection molding",
            "year": "2025",
            "research_problem": "warpage",
            "material": ["ABS"],
            "process": ["injection molding"],
            "method": ["Moldflow"],
            "dataset_or_experiment": "simulation",
            "quality_metrics": ["warpage"],
            "main_findings": "reduced warpage",
            "limitations": "",
            "evidence_sections": ["Abstract", "Results"],
            "confidence": "high",
        }
    ]
    defect_cards = [
        {
            "defect_type": "warpage",
            "possible_causes": "melt temperature",
            "related_parameters": ["melt temperature"],
            "suggested_actions": "optimize",
            "evidence_paper_id": "paper_1",
            "evidence_text": "Optimized melt temperature reduces warpage.",
            "confidence": "high",
        }
    ]

    sections_path = tmp_path / "cleaned_sections.jsonl"
    paper_cards_path = tmp_path / "paper_cards.jsonl"
    defect_cards_path = tmp_path / "defect_cards.jsonl"
    method_cards_path = tmp_path / "missing_method_cards.jsonl"
    parameter_cards_path = tmp_path / "missing_parameter_cards.jsonl"
    output_path = tmp_path / "chunks.jsonl"
    report_path = tmp_path / "chunk_report.md"
    write_jsonl(sections_path, sections)
    write_jsonl(paper_cards_path, paper_cards)
    write_jsonl(defect_cards_path, defect_cards)

    chunks = build_chunks(
        sections_path=sections_path,
        paper_cards_path=paper_cards_path,
        defect_cards_path=defect_cards_path,
        method_cards_path=method_cards_path,
        parameter_cards_path=parameter_cards_path,
        output_path=output_path,
        report_path=report_path,
    )

    assert output_path.exists()
    assert report_path.exists()
    assert all("References should not be chunked" not in chunk["text"] for chunk in chunks)
    assert any(chunk["section_name"] == "Abstract" for chunk in chunks)
    assert any(chunk["chunk_type"] == "table_or_figure_context" for chunk in chunks)
    assert any(chunk["chunk_type"] == "knowledge_card" for chunk in chunks)
    assert all({"title", "year", "section", "paper_id", "chunk_type"}.issubset(chunk["metadata"]) for chunk in chunks)
