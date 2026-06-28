from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "parsed_docs.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cleaned_sections.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "clean_report.md"

SECTION_NAMES = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiment",
    "Results",
    "Discussion",
    "Conclusion",
    "References",
]

SECTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Abstract", ("abstract", "摘要")),
    ("Introduction", ("introduction", "引言", "绪论")),
    ("Related Work", ("related work", "literature review", "background", "相关工作", "文献综述", "研究现状")),
    ("Method", ("method", "methods", "methodology", "materials and methods", "方法", "模型", "算法")),
    ("Experiment", ("experiment", "experiments", "experimental", "case study", "实验", "试验")),
    ("Results", ("results", "result", "结果")),
    ("Discussion", ("discussion", "讨论", "分析")),
    ("Conclusion", ("conclusion", "conclusions", "结论", "总结")),
    ("References", ("references", "reference", "bibliography", "参考文献")),
]

COPYRIGHT_PATTERNS = (
    "copyright",
    "all rights reserved",
    "creative commons",
    "http://creativecommons.org",
    "©",
    "版权所有",
)

FIGURE_TABLE_PATTERN = re.compile(r"^\s*(figure|fig\.?|table|图|表)\s*[\d一二三四五六七八九十ivxlc]*", re.IGNORECASE)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class PaperCleanResult:
    paper_id: str
    file_name: str
    sections: list[dict[str, Any]]
    raw_char_count: int
    clean_char_count: int
    reference_char_count: int
    recognized_section_count: int
    removed_noise_count: int
    anomalies: list[str] = field(default_factory=list)


def read_parsed_docs(input_path: Path) -> dict[str, list[dict[str, Any]]]:
    papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            papers[record["paper_id"]].append(record)
    for records in papers.values():
        records.sort(key=lambda record: int(record["page_num"]))
    return papers


def normalize_repeated_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip().lower()
    line = re.sub(r"\d+", "#", line)
    return line


def is_isolated_page_number(line: str) -> bool:
    text = line.strip()
    return bool(re.fullmatch(r"(?:page\s*)?\d{1,4}", text, flags=re.IGNORECASE))


def is_copyright_line(line: str) -> bool:
    lowered = line.lower()
    return any(pattern in lowered for pattern in COPYRIGHT_PATTERNS)


def is_figure_or_table_line(line: str) -> bool:
    return bool(FIGURE_TABLE_PATTERN.search(line.strip()))


def find_repeated_noise_lines(pages: list[dict[str, Any]]) -> set[str]:
    page_count = len(pages)
    if page_count < 3:
        return set()

    line_pages: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        page_num = int(page["page_num"])
        raw_lines = str(page.get("raw_text", "")).splitlines()
        candidate_lines = raw_lines[:5] + raw_lines[-5:]
        for line in candidate_lines:
            normalized = normalize_repeated_line(line)
            if not normalized or len(normalized) < 6 or is_figure_or_table_line(line):
                continue
            line_pages[normalized].add(page_num)

    threshold = max(3, int(page_count * 0.3))
    return {line for line, page_nums in line_pages.items() if len(page_nums) >= threshold}


def remove_page_noise(pages: list[dict[str, Any]]) -> tuple[list[tuple[str, int]], int]:
    repeated_noise = find_repeated_noise_lines(pages)
    cleaned_lines: list[tuple[str, int]] = []
    removed_count = 0

    for page in pages:
        page_num = int(page["page_num"])
        for line in str(page.get("raw_text", "")).splitlines():
            stripped = line.strip()
            normalized = normalize_repeated_line(stripped)
            if not stripped:
                cleaned_lines.append(("", page_num))
                continue
            if is_figure_or_table_line(stripped):
                cleaned_lines.append((stripped, page_num))
                continue
            if is_isolated_page_number(stripped) or is_copyright_line(stripped) or normalized in repeated_noise:
                removed_count += 1
                continue
            cleaned_lines.append((stripped, page_num))

    return cleaned_lines, removed_count


def fix_hyphenation(lines: list[tuple[str, int]]) -> list[tuple[str, int]]:
    fixed: list[tuple[str, int]] = []
    index = 0
    while index < len(lines):
        line, page_num = lines[index]
        if (
            line.endswith("-")
            and index + 1 < len(lines)
            and re.search(r"[A-Za-z]-$", line)
            and re.match(r"^[A-Za-z]", lines[index + 1][0].lstrip())
        ):
            next_line, _ = lines[index + 1]
            fixed.append((line[:-1].rstrip() + " " + next_line.lstrip(), page_num))
            index += 2
            continue
        fixed.append((line, page_num))
        index += 1
    return fixed


def strip_heading_number(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^(?:chapter\s+)?\d+(?:\.\d+)*\s*[\.\)]?\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[IVXLC]+\s*[\.\)]?\s+", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t:：.-")
    return text


def detect_section_heading(line: str) -> str | None:
    text = strip_heading_number(line)
    if not text or len(text) > 90:
        return None
    normalized = normalize_repeated_line(text).strip(" .:：")

    for section_name, aliases in SECTION_PATTERNS:
        for alias in aliases:
            alias_normalized = normalize_repeated_line(alias).strip()
            if CHINESE_PATTERN.search(alias):
                if normalized == alias_normalized or normalized.startswith(alias_normalized + " "):
                    return section_name
            elif normalized == alias_normalized or normalized.startswith(alias_normalized + " "):
                return section_name
    return None


def normalize_clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def lines_to_text(lines: list[tuple[str, int]]) -> str:
    text = "\n".join(line for line, _ in lines)
    return normalize_clean_text(text)


def build_sections(paper_id: str, file_name: str, lines: list[tuple[str, int]]) -> tuple[list[dict[str, Any]], int]:
    heading_positions: list[tuple[int, str]] = []
    for index, (line, _) in enumerate(lines):
        section_name = detect_section_heading(line)
        if section_name:
            heading_positions.append((index, section_name))

    recognized_count = len(heading_positions)
    if not heading_positions:
        text = lines_to_text(lines)
        page_nums = [page_num for line, page_num in lines if line.strip()]
        return (
            [
                {
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "section_name": "Unknown",
                    "section_order": 1,
                    "clean_text": text,
                    "char_count": len(text),
                    "page_start": min(page_nums) if page_nums else "",
                    "page_end": max(page_nums) if page_nums else "",
                    "is_reference_section": False,
                }
            ],
            recognized_count,
        )

    if heading_positions[0][0] > 0:
        heading_positions.insert(0, (0, "Front Matter"))

    sections: list[dict[str, Any]] = []
    reference_seen = False
    for order, (start_index, section_name) in enumerate(heading_positions, start=1):
        end_index = heading_positions[order][0] if order < len(heading_positions) else len(lines)
        section_lines = lines[start_index:end_index]
        clean_text = lines_to_text(section_lines)
        page_nums = [page_num for line, page_num in section_lines if line.strip()]
        if not clean_text:
            continue

        if section_name == "References":
            reference_seen = True
        sections.append(
            {
                "paper_id": paper_id,
                "file_name": file_name,
                "section_name": section_name,
                "section_order": len(sections) + 1,
                "clean_text": clean_text,
                "char_count": len(clean_text),
                "page_start": min(page_nums) if page_nums else "",
                "page_end": max(page_nums) if page_nums else "",
                "is_reference_section": reference_seen,
            }
        )

    return sections, recognized_count


def clean_paper(paper_id: str, pages: list[dict[str, Any]]) -> PaperCleanResult:
    file_name = str(pages[0].get("file_name", "")) if pages else ""
    raw_char_count = sum(len(str(page.get("raw_text", ""))) for page in pages)
    cleaned_lines, removed_noise_count = remove_page_noise(pages)
    cleaned_lines = fix_hyphenation(cleaned_lines)
    sections, recognized_count = build_sections(paper_id, file_name, cleaned_lines)

    clean_char_count = sum(int(section["char_count"]) for section in sections)
    reference_char_count = sum(
        int(section["char_count"]) for section in sections if bool(section["is_reference_section"])
    )
    anomalies: list[str] = []
    if recognized_count == 0:
        anomalies.append("no_section_heading_detected")
    if len(sections) <= 1:
        anomalies.append("single_section_only")
    if raw_char_count and clean_char_count / raw_char_count < 0.2:
        anomalies.append("low_clean_text_ratio")
    if not any(not section["is_reference_section"] for section in sections):
        anomalies.append("no_non_reference_section")

    return PaperCleanResult(
        paper_id=paper_id,
        file_name=file_name,
        sections=sections,
        raw_char_count=raw_char_count,
        clean_char_count=clean_char_count,
        reference_char_count=reference_char_count,
        recognized_section_count=recognized_count,
        removed_noise_count=removed_noise_count,
        anomalies=anomalies,
    )


def write_sections(results: list[PaperCleanResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            for section in result.sections:
                file.write(json.dumps(section, ensure_ascii=False) + "\n")


def build_report(results: list[PaperCleanResult]) -> str:
    total_papers = len(results)
    recognized_papers = sum(1 for result in results if result.recognized_section_count > 0)
    raw_chars = sum(result.raw_char_count for result in results)
    clean_chars = sum(result.clean_char_count for result in results)
    reference_chars = sum(result.reference_char_count for result in results)
    deleted_ratio = 0.0 if raw_chars == 0 else max(raw_chars - clean_chars, 0) / raw_chars
    reference_ratio = 0.0 if clean_chars == 0 else reference_chars / clean_chars
    section_distribution = Counter(len(result.sections) for result in results)
    anomaly_results = [result for result in results if result.anomalies]

    lines = [
        "# Clean Report",
        "",
        "## Summary",
        "",
        f"- Total papers: {total_papers}",
        f"- Section recognition success rate: {recognized_papers}/{total_papers} ({recognized_papers / total_papers:.2%})" if total_papers else "- Section recognition success rate: 0/0 (0.00%)",
        f"- Deleted text ratio after cleaning: {deleted_ratio:.2%}",
        f"- Reference-section ratio excluded from RAG chunks by default: {reference_ratio:.2%}",
        f"- Total cleaned sections: {sum(len(result.sections) for result in results)}",
        "",
        "## Section Count Distribution",
        "",
    ]
    for section_count, paper_count in sorted(section_distribution.items()):
        lines.append(f"- {section_count} sections per paper: {paper_count} papers")

    lines.extend(["", "## Anomalous Papers", ""])
    if anomaly_results:
        for result in anomaly_results:
            lines.append(f"- `{result.paper_id}` | {result.file_name} | {', '.join(result.anomalies)}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_report(results: list[PaperCleanResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(results), encoding="utf-8")


def print_stats(results: list[PaperCleanResult]) -> None:
    total_papers = len(results)
    recognized_papers = sum(1 for result in results if result.recognized_section_count > 0)
    total_sections = sum(len(result.sections) for result in results)
    raw_chars = sum(result.raw_char_count for result in results)
    clean_chars = sum(result.clean_char_count for result in results)
    deleted_ratio = 0.0 if raw_chars == 0 else max(raw_chars - clean_chars, 0) / raw_chars
    anomalies = sum(1 for result in results if result.anomalies)

    print(f"总论文数: {total_papers}")
    print(f"章节识别成功论文数: {recognized_papers}")
    print(f"总 section 数: {total_sections}")
    print(f"被删除文本比例: {deleted_ratio:.2%}")
    print(f"异常论文数: {anomalies}")
    print("前 2 条 section 摘要:")
    shown = 0
    for result in results:
        for section in result.sections:
            snippet = re.sub(r"\s+", " ", str(section["clean_text"]))[:120]
            print(
                f"  {section['paper_id']} | {section['section_name']} | "
                f"chars={section['char_count']} | pages={section['page_start']}-{section['page_end']} | {snippet}"
            )
            shown += 1
            if shown >= 2:
                return


def run_cleaning(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> list[PaperCleanResult]:
    papers = read_parsed_docs(input_path)
    results = [clean_paper(paper_id, pages) for paper_id, pages in papers.items()]
    write_sections(results, output_path)
    write_report(results, report_path)
    print_stats(results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean parsed PDF text and identify paper sections.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_cleaning(input_path=args.input, output_path=args.output, report_path=args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
