from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY_CSV = PROJECT_ROOT / "data" / "metadata" / "paper_inventory.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "metadata" / "selected_papers.csv"
DEFAULT_DEV_DIR = PROJECT_ROOT / "data" / "dev_papers"
DEFAULT_SELECTED_DIR = PROJECT_ROOT / "data" / "selected_papers"
DEFAULT_SELECTED_COUNT = 84
DEFAULT_DEV_COUNT = 24


@dataclass(frozen=True)
class CategoryRule:
    name: str
    keywords: tuple[str, ...]


CATEGORY_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        "注塑缺陷",
        (
            "warpage",
            "shrinkage",
            "sink mark",
            "short shot",
            "weld line",
            "flash",
            "void",
            "burn mark",
            "defect",
            "defection",
            "翘曲",
            "收缩",
            "缩痕",
            "短射",
            "熔接线",
            "飞边",
            "空洞",
            "气孔",
            "烧焦",
            "缺陷",
            "开裂",
        ),
    ),
    CategoryRule(
        "工艺参数",
        (
            "melt temperature",
            "mold temperature",
            "mould temperature",
            "injection speed",
            "packing pressure",
            "holding pressure",
            "cooling time",
            "holding time",
            "process parameter",
            "processing parameter",
            "processing condition",
            "熔体温度",
            "模具温度",
            "注射速度",
            "保压压力",
            "冷却时间",
            "保压时间",
            "工艺参数",
            "工艺条件",
        ),
    ),
    CategoryRule(
        "质量预测",
        (
            "quality prediction",
            "defect prediction",
            "process monitoring",
            "sensor",
            "sensing",
            "quality analysis",
            "quality control",
            "质量预测",
            "缺陷预测",
            "过程监控",
            "工艺监控",
            "传感",
            "品质",
            "质量",
        ),
    ),
    CategoryRule(
        "工艺优化",
        (
            "optimization",
            "optimisation",
            "optimiz",
            "optimis",
            " ga ",
            "genetic algorithm",
            " pso ",
            "particle swarm",
            "bayesian optimization",
            " doe ",
            "design of experiment",
            "response surface",
            "响应面",
            "正交试验",
            "田口",
            "遗传算法",
            "粒子群",
            "贝叶斯优化",
            "多目标优化",
            "优化",
        ),
    ),
    CategoryRule(
        "算法方法",
        (
            "machine learning",
            "deep learning",
            "knowledge graph",
            " rag ",
            "retrieval augmented",
            " llm ",
            "large language model",
            "digital twin",
            "neural network",
            "transfer learning",
            "artificial intelligence",
            "机器学习",
            "深度学习",
            "知识图谱",
            "检索增强",
            "大语言模型",
            "数字孪生",
            "神经网络",
            "智能",
        ),
    ),
    CategoryRule(
        "材料和场景",
        (
            " pc ",
            "pmma",
            " abs ",
            " pp ",
            "polycarbonate",
            "polymethyl methacrylate",
            "polypropylene",
            "micro injection molding",
            "micro-injection",
            "microinjection",
            " cae ",
            "moldflow",
            "moldex3d",
            "mouldflow",
            "simulation",
            "聚碳酸酯",
            "聚甲基丙烯酸甲酯",
            "丙烯腈",
            "聚丙烯",
            "微注塑",
            "微注射",
            "模流",
            "仿真",
            "模拟",
        ),
    ),
)

OUTPUT_FIELDS_EXTRA = [
    "primary_category",
    "matched_categories",
    "category_score",
    "selection_rank",
    "in_dev_set",
]


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[_\-–—().,\[\]{}+/:;]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return f" {value} "


def keyword_matches(text: str, keyword: str) -> bool:
    keyword = keyword.lower()
    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword in text
    return normalize_text(keyword) in text


def match_categories(record: dict[str, str]) -> dict[str, int]:
    text = normalize_text(
        " ".join(
            [
                record.get("file_name", ""),
                record.get("title_guess", ""),
                record.get("keyword_tags_guess", ""),
            ]
        )
    )
    scores: dict[str, int] = {}
    for rule in CATEGORY_RULES:
        score = sum(1 for keyword in rule.keywords if keyword_matches(text, keyword))
        if score:
            scores[rule.name] = score
    return scores


def read_inventory(inventory_csv: Path) -> list[dict[str, str]]:
    with inventory_csv.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def resolve_file_path(record: dict[str, str], project_root: Path) -> Path:
    file_path = Path(record["file_path"])
    if not file_path.is_absolute():
        file_path = project_root / file_path
    return file_path


def year_sort_value(record: dict[str, str]) -> int:
    year = record.get("year_guess", "")
    return int(year) if year.isdigit() else 0


def sort_candidates(records: list[dict[str, str]], category: str) -> list[dict[str, str]]:
    return sorted(
        records,
        key=lambda record: (
            record["_category_scores"].get(category, 0),
            len(record["_matched_categories"]),
            year_sort_value(record),
            -float(record.get("file_size_mb") or 0),
            record.get("file_name", ""),
        ),
        reverse=True,
    )


def select_balanced(records: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    category_names = [rule.name for rule in CATEGORY_RULES]
    target_per_category = max(1, target_count // len(category_names))
    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    for category in category_names:
        candidates = [
            record
            for record in records
            if category in record["_category_scores"] and record["paper_id"] not in selected_ids
        ]
        for record in sort_candidates(candidates, category)[:target_per_category]:
            record["_primary_category"] = category
            selected.append(record)
            selected_ids.add(record["paper_id"])

    if len(selected) < target_count:
        remaining = [record for record in records if record["paper_id"] not in selected_ids]
        remaining = sorted(
            remaining,
            key=lambda record: (
                sum(record["_category_scores"].values()),
                len(record["_matched_categories"]),
                year_sort_value(record),
                record.get("file_name", ""),
            ),
            reverse=True,
        )
        for record in remaining[: target_count - len(selected)]:
            if record["_matched_categories"]:
                record["_primary_category"] = record["_matched_categories"][0]
            else:
                record["_primary_category"] = "补充"
            selected.append(record)
            selected_ids.add(record["paper_id"])

    return selected[:target_count]


def prepare_records(records: list[dict[str, str]], project_root: Path) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for record in records:
        file_path = resolve_file_path(record, project_root)
        if not file_path.exists():
            continue
        enriched = dict(record)
        scores = match_categories(enriched)
        enriched["_category_scores"] = scores
        enriched["_matched_categories"] = sorted(scores, key=lambda category: scores[category], reverse=True)
        prepared.append(enriched)
    return prepared


def write_selected_csv(records: list[dict[str, str]], output_csv: Path, dev_ids: set[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    original_fields = [field for field in records[0] if not field.startswith("_")] if records else []
    fieldnames = original_fields + [field for field in OUTPUT_FIELDS_EXTRA if field not in original_fields]

    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            row = {field: record.get(field, "") for field in original_fields}
            row["selected_stage"] = "dev" if record["paper_id"] in dev_ids else "selected"
            row["primary_category"] = record.get("_primary_category", "")
            row["matched_categories"] = "; ".join(record.get("_matched_categories", []))
            row["category_score"] = str(sum(record.get("_category_scores", {}).values()))
            row["selection_rank"] = str(index)
            row["in_dev_set"] = "yes" if record["paper_id"] in dev_ids else "no"
            writer.writerow(row)


def copy_papers(records: list[dict[str, str]], target_dir: Path, project_root: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = resolve_file_path(record, project_root)
        shutil.copy2(source, target_dir / source.name)


def print_selection_stats(selected_records: list[dict[str, str]], dev_records: list[dict[str, str]]) -> None:
    selected_counts = Counter(record.get("_primary_category", "补充") for record in selected_records)
    dev_counts = Counter(record.get("_primary_category", "补充") for record in dev_records)

    print(f"selected_papers 总数: {len(selected_records)}")
    print("selected_papers 类别分布:")
    for rule in CATEGORY_RULES:
        print(f"  {rule.name}: {selected_counts.get(rule.name, 0)}")
    if selected_counts.get("补充", 0):
        print(f"  补充: {selected_counts['补充']}")

    print(f"dev_papers 总数: {len(dev_records)}")
    print("dev_papers 类别分布:")
    for rule in CATEGORY_RULES:
        print(f"  {rule.name}: {dev_counts.get(rule.name, 0)}")
    if dev_counts.get("补充", 0):
        print(f"  补充: {dev_counts['补充']}")


def run_selection(
    inventory_csv: Path = DEFAULT_INVENTORY_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    dev_dir: Path = DEFAULT_DEV_DIR,
    selected_dir: Path = DEFAULT_SELECTED_DIR,
    selected_count: int = DEFAULT_SELECTED_COUNT,
    dev_count: int = DEFAULT_DEV_COUNT,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 60 <= selected_count <= 100:
        raise ValueError("selected_count must be between 60 and 100.")
    if not 20 <= dev_count <= 30:
        raise ValueError("dev_count must be between 20 and 30.")
    if dev_count > selected_count:
        raise ValueError("dev_count cannot be greater than selected_count.")

    inventory_records = read_inventory(inventory_csv)
    prepared_records = prepare_records(inventory_records, project_root)
    if len(prepared_records) < selected_count:
        raise ValueError(f"Only {len(prepared_records)} source PDFs are available; need {selected_count}.")

    selected_records = select_balanced(prepared_records, selected_count)
    dev_records = select_balanced(selected_records, dev_count)
    dev_ids = {record["paper_id"] for record in dev_records}

    write_selected_csv(selected_records, output_csv, dev_ids)
    copy_papers(selected_records, selected_dir, project_root)
    copy_papers(dev_records, dev_dir, project_root)
    print_selection_stats(selected_records, dev_records)
    return selected_records, dev_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select balanced paper subsets from metadata only.")
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--dev-dir", type=Path, default=DEFAULT_DEV_DIR)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--selected-count", type=int, default=DEFAULT_SELECTED_COUNT)
    parser.add_argument("--dev-count", type=int, default=DEFAULT_DEV_COUNT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_selection(
        inventory_csv=args.inventory_csv,
        output_csv=args.output_csv,
        dev_dir=args.dev_dir,
        selected_dir=args.selected_dir,
        selected_count=args.selected_count,
        dev_count=args.dev_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
