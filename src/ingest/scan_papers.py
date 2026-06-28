from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw_papers"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "paper_inventory.csv"

CSV_FIELDS = [
    "paper_id",
    "file_name",
    "file_path",
    "file_size_mb",
    "modified_time",
    "title_guess",
    "year_guess",
    "language_guess",
    "keyword_tags_guess",
    "selected_stage",
]

YEAR_PATTERN = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")

KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("injection molding", ("injection molding", "injection moulding", "micro-injection", "molding", "moulding", "注塑", "注射成型")),
    ("defect", ("defect", "defection", "defects", "缺陷", "翘曲", "收缩", "开裂")),
    ("warpage", ("warpage", "warp", "翘曲")),
    ("shrinkage", ("shrinkage", "shrink", "收缩")),
    ("process parameter", ("process parameter", "processing parameter", "工艺参数", "processing condition", "工艺优化")),
    ("quality prediction", ("quality prediction", "quality analysis", "质量预测", "质量分析", "品质")),
    ("optimization", ("optimization", "optimisation", "optimiz", "optimis", "优化", "最优")),
    ("machine learning", ("machine learning", "deep learning", "neural network", "ai-driven", "artificial intelligence", "机器学习", "神经网络", "智能")),
    ("knowledge graph", ("knowledge graph", "知识图谱")),
    ("RAG", (" rag ", "retrieval augmented", "检索增强")),
    ("sensor", ("sensor", "sensing", "传感")),
    ("CAE", ("cae", "moldflow", "moldex3d", "simulation", "仿真", "模拟", "数值模拟")),
    ("simulation", ("simulation", "simul", "numerical", "模拟", "仿真")),
    ("PC", (" pc ", "polycarbonate", "聚碳酸酯")),
    ("PMMA", ("pmma", "polymethyl methacrylate", "聚甲基丙烯酸甲酯")),
    ("ABS", (" abs ", "acrylonitrile butadiene styrene", "丙烯腈-苯乙烯")),
    ("PP", (" pp ", "polypropylene", "聚丙烯")),
]


def normalize_for_keyword_match(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    normalized = re.sub(r"[_\-–—().,\[\]{}+]+", " ", stem)
    return f" {re.sub(r'\s+', ' ', normalized).strip()} "


def guess_title(file_name: str) -> str:
    title = Path(file_name).stem
    title = re.sub(r"[_]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"\s+-\s+", " - ", title)
    title = re.sub(r"\s+\(\d+\)$", "", title)
    title = re.sub(r"\s+_\d+$", "", title)
    return title.strip()


def guess_year(file_name: str) -> str:
    match = YEAR_PATTERN.search(file_name)
    return match.group(1) if match else ""


def guess_language(file_name: str) -> str:
    return "zh" if CHINESE_PATTERN.search(file_name) else "en"


def guess_keyword_tags(file_name: str) -> list[str]:
    text = normalize_for_keyword_match(file_name)
    tags: list[str] = []
    for tag, keywords in KEYWORD_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            tags.append(tag)
    return tags


def build_paper_id(relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()
    return f"paper_{digest[:12]}"


def scan_papers(input_dir: Path, project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    records: list[dict[str, str]] = []
    pdf_paths = sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )

    for pdf_path in pdf_paths:
        stat = pdf_path.stat()
        try:
            relative_path = pdf_path.relative_to(project_root).as_posix()
        except ValueError:
            relative_path = pdf_path.as_posix()

        tags = guess_keyword_tags(pdf_path.name)
        records.append(
            {
                "paper_id": build_paper_id(relative_path),
                "file_name": pdf_path.name,
                "file_path": relative_path,
                "file_size_mb": f"{stat.st_size / (1024 * 1024):.2f}",
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "title_guess": guess_title(pdf_path.name),
                "year_guess": guess_year(pdf_path.name),
                "language_guess": guess_language(pdf_path.name),
                "keyword_tags_guess": "; ".join(tags),
                "selected_stage": "raw",
            }
        )

    return records


def write_inventory(records: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def print_stats(records: list[dict[str, str]]) -> None:
    year_counts = Counter(record["year_guess"] or "unknown" for record in records)
    keyword_counts: Counter[str] = Counter()
    for record in records:
        for tag in filter(None, (tag.strip() for tag in record["keyword_tags_guess"].split(";"))):
            keyword_counts[tag] += 1

    print(f"总 PDF 数: {len(records)}")
    print("年份分布:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")
    print("关键词分布:")
    for keyword, count in keyword_counts.most_common():
        print(f"  {keyword}: {count}")
    print("前 10 条文件名:")
    for record in records[:10]:
        print(f"  {record['file_name']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan PDF inventory without reading PDF content.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = scan_papers(args.input_dir)
    write_inventory(records, args.output_csv)
    print_stats(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
