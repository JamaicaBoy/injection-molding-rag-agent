from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "cleaned_sections.jsonl"
DEFAULT_PAPER_CARDS = PROJECT_ROOT / "data" / "processed" / "paper_cards.jsonl"
DEFAULT_DEFECT_CARDS = PROJECT_ROOT / "data" / "processed" / "defect_cards.jsonl"
DEFAULT_METHOD_CARDS = PROJECT_ROOT / "data" / "processed" / "method_cards.jsonl"
DEFAULT_PARAMETER_CARDS = PROJECT_ROOT / "data" / "processed" / "parameter_cards.jsonl"
DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "data" / "manual_review" / "review_queue.csv"

PRIORITY_SECTIONS = ("Abstract", "Conclusion", "Results", "Discussion")
YEAR_PATTERN = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+|\n+")

PAPER_FIELDS = [
    "paper_id",
    "title",
    "year",
    "research_problem",
    "material",
    "process",
    "method",
    "dataset_or_experiment",
    "quality_metrics",
    "main_findings",
    "limitations",
    "evidence_sections",
    "confidence",
]
DEFECT_FIELDS = [
    "defect_type",
    "possible_causes",
    "related_parameters",
    "suggested_actions",
    "evidence_paper_id",
    "evidence_text",
    "confidence",
]
METHOD_FIELDS = [
    "method_name",
    "task",
    "input_data",
    "output_target",
    "advantages",
    "limitations",
    "evidence_paper_id",
    "confidence",
]
PARAMETER_FIELDS = [
    "parameter_name",
    "effect_on_quality",
    "related_defects",
    "increase_effect",
    "decrease_effect",
    "evidence_paper_id",
    "confidence",
]
REVIEW_FIELDS = ["card_type", "evidence_paper_id", "item_name", "confidence", "reason"]


@dataclass(frozen=True)
class Rule:
    name: str
    keywords: tuple[str, ...]


DEFECT_RULES = (
    Rule("warpage", ("warpage", "warp", "翘曲", "变形")),
    Rule("shrinkage", ("shrinkage", "shrink", "收缩")),
    Rule("sink mark", ("sink mark", "sink marks", "缩痕")),
    Rule("short shot", ("short shot", "short-shot", "短射", "欠注")),
    Rule("weld line", ("weld line", "weld lines", "熔接线", "焊接线")),
    Rule("flash", ("flash", "flashing", "飞边", "毛边")),
    Rule("void", ("void", "voids", "气孔", "空洞")),
    Rule("burn mark", ("burn mark", "burn marks", "烧焦", "焦痕")),
)

PARAMETER_RULES = (
    Rule("melt temperature", ("melt temperature", "melting temperature", "熔体温度", "料温")),
    Rule("mold temperature", ("mold temperature", "mould temperature", "模具温度", "模温")),
    Rule("injection speed", ("injection speed", "注射速度", "注塑速度")),
    Rule("injection pressure", ("injection pressure", "注射压力", "注塑压力")),
    Rule("packing pressure", ("packing pressure", "holding pressure", "保压压力")),
    Rule("cooling time", ("cooling time", "冷却时间")),
    Rule("holding time", ("holding time", "packing time", "保压时间")),
    Rule("filling time", ("filling time", "fill time", "充填时间", "填充时间")),
)

METHOD_RULES = (
    Rule("machine learning", ("machine learning", "机器学习")),
    Rule("deep learning", ("deep learning", "深度学习")),
    Rule("BP neural network", ("bp neural network", "bp network", "bp 神经网络", "神经网络")),
    Rule("genetic algorithm", ("genetic algorithm", " ga ", "遗传算法")),
    Rule("particle swarm optimization", ("particle swarm", " pso ", "粒子群")),
    Rule("Bayesian optimization", ("bayesian optimization", "贝叶斯优化")),
    Rule("DOE", ("design of experiment", " doe ", "正交试验", "试验设计")),
    Rule("response surface", ("response surface", " rsm ", "响应面")),
    Rule("CAE simulation", (" cae ", "moldflow", "moldex3d", "simulation", "仿真", "模拟")),
    Rule("knowledge graph", ("knowledge graph", "知识图谱")),
    Rule("RAG", (" rag ", "retrieval augmented", "检索增强")),
    Rule("LLM", (" llm ", "large language model", "大语言模型")),
    Rule("digital twin", ("digital twin", "数字孪生")),
)

MATERIAL_RULES = (
    Rule("PC", (" pc ", "polycarbonate", "聚碳酸酯")),
    Rule("PMMA", ("pmma", "polymethyl methacrylate", "聚甲基丙烯酸甲酯")),
    Rule("ABS", (" abs ", "acrylonitrile butadiene styrene", "丙烯腈")),
    Rule("PP", (" pp ", "polypropylene", "聚丙烯")),
    Rule("POM", (" pom ", "polyoxymethylene", "聚甲醛")),
    Rule("PA", (" pa ", "polyamide", "nylon", "聚酰胺", "尼龙")),
    Rule("PEEK", ("peek", "聚醚醚酮")),
)

PROCESS_RULES = (
    Rule("injection molding", ("injection molding", "injection moulding", "注塑", "注射成型")),
    Rule("micro injection molding", ("micro injection", "micro-injection", "微注塑", "微注射")),
    Rule("CAE/Moldflow", ("moldflow", "moldex3d", " cae ", "模流")),
)

QUALITY_RULES = (
    Rule("quality", ("quality", "品质", "质量")),
    Rule("warpage", ("warpage", "翘曲")),
    Rule("shrinkage", ("shrinkage", "收缩")),
    Rule("defect", ("defect", "缺陷")),
    Rule("strength", ("strength", "强度")),
    Rule("deformation", ("deformation", "变形")),
)

LIMITATION_KEYWORDS = ("limitation", "future work", "future research", "not considered", "不足", "局限", "展望")
ACTION_KEYWORDS = ("optimiz", "reduce", "improve", "control", "adjust", "优化", "降低", "改善", "控制", "调整")
CAUSE_KEYWORDS = ("caused by", "due to", "because", "influence", "effect", "影响", "导致", "由于", "原因")
EXPERIMENT_KEYWORDS = ("experiment", "case study", "simulation", "moldflow", "doe", "orthogonal", "实验", "试验", "仿真", "模拟")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-–—().,\[\]{}+/:;]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} "


def matches_keyword(text: str, keyword: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword in text
    return normalize_text(keyword) in normalize_text(text)


def find_matches(text: str, rules: tuple[Rule, ...]) -> list[str]:
    matched: list[str] = []
    for rule in rules:
        if any(matches_keyword(text, keyword) for keyword in rule.keywords):
            matched.append(rule.name)
    return matched


def read_sections(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def group_sections(sections: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in sections:
        grouped[section["paper_id"]].append(section)
    for paper_sections in grouped.values():
        paper_sections.sort(key=lambda item: int(item.get("section_order", 0)))
    return grouped


def title_from_file_name(file_name: str) -> str:
    title = Path(file_name).stem
    title = re.sub(r"^\s*.+?\s+-\s+(19\d{2}|20\d{2})\s+-\s+", "", title)
    title = re.sub(r"[_]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def year_from_file_name(file_name: str) -> str:
    match = YEAR_PATTERN.search(file_name)
    return match.group(1) if match else ""


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if len(sentences) <= 1 and len(text) > 300:
        sentences = [chunk.strip() for chunk in re.split(r"[;；]", text) if chunk.strip()]
    return sentences


def pick_evidence_sentences(text: str, keywords: tuple[str, ...] = (), max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if keywords:
        matched = [
            sentence
            for sentence in sentences
            if any(matches_keyword(sentence, keyword) for keyword in keywords)
        ]
    else:
        matched = sentences
    selected = (matched or sentences)[:max_sentences]
    return " ".join(sentence[:350] for sentence in selected).strip()


def prioritized_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_reference = [section for section in sections if not section.get("is_reference_section")]
    priority = [section for section in non_reference if section.get("section_name") in PRIORITY_SECTIONS]
    other = [section for section in non_reference if section.get("section_name") not in PRIORITY_SECTIONS]
    return priority + other


def section_text(sections: list[dict[str, Any]]) -> str:
    return "\n".join(str(section.get("clean_text", "")) for section in sections)


def confidence_from_evidence(
    matched_items: list[str],
    evidence_sections: list[str],
    evidence_mentions: int,
    has_priority_evidence: bool,
) -> str:
    if len(set(evidence_sections)) >= 2 and evidence_mentions >= 3 and matched_items and has_priority_evidence:
        return "high"
    if matched_items and evidence_mentions >= 1:
        return "medium"
    return "low"


def summarize_field(sections: list[dict[str, Any]], keywords: tuple[str, ...]) -> str:
    text = section_text(sections)
    return pick_evidence_sentences(text, keywords=keywords, max_sentences=2)


def build_paper_card(paper_id: str, sections: list[dict[str, Any]]) -> dict[str, Any]:
    selected_sections = prioritized_sections(sections)
    file_name = str(sections[0].get("file_name", ""))
    combined_text = section_text(selected_sections)
    evidence_sections = [str(section.get("section_name", "")) for section in selected_sections[:5]]

    materials = find_matches(combined_text + " " + file_name, MATERIAL_RULES)
    processes = find_matches(combined_text + " " + file_name, PROCESS_RULES)
    methods = find_matches(combined_text + " " + file_name, METHOD_RULES)
    metrics = find_matches(combined_text + " " + file_name, QUALITY_RULES)
    problem_keywords = tuple(keyword for rule in DEFECT_RULES + QUALITY_RULES for keyword in rule.keywords)

    matched_items = materials + processes + methods + metrics
    confidence = confidence_from_evidence(
        matched_items=matched_items,
        evidence_sections=evidence_sections,
        evidence_mentions=sum(combined_text.lower().count(item.lower()) for item in matched_items if item.isascii()),
        has_priority_evidence=any(section.get("section_name") in PRIORITY_SECTIONS for section in selected_sections),
    )

    return {
        "paper_id": paper_id,
        "title": title_from_file_name(file_name),
        "year": year_from_file_name(file_name),
        "research_problem": summarize_field(selected_sections, problem_keywords),
        "material": materials,
        "process": processes,
        "method": methods,
        "dataset_or_experiment": summarize_field(selected_sections, EXPERIMENT_KEYWORDS),
        "quality_metrics": metrics,
        "main_findings": summarize_field(selected_sections, ("result", "conclusion", "finding", "结果", "结论", "表明")),
        "limitations": summarize_field(selected_sections, LIMITATION_KEYWORDS),
        "evidence_sections": evidence_sections,
        "confidence": confidence,
    }


def evidence_for_item(sections: list[dict[str, Any]], rule: Rule, extra_keywords: tuple[str, ...] = ()) -> tuple[str, list[str], int, bool]:
    selected_sections = prioritized_sections(sections)
    evidence_keywords = rule.keywords + extra_keywords
    matched_sections: list[str] = []
    evidence_parts: list[str] = []
    mentions = 0
    has_priority = False
    for section in selected_sections:
        text = str(section.get("clean_text", ""))
        if any(matches_keyword(text, keyword) for keyword in rule.keywords):
            matched_sections.append(str(section.get("section_name", "")))
            evidence_parts.append(pick_evidence_sentences(text, keywords=evidence_keywords, max_sentences=2))
            mentions += sum(1 for keyword in rule.keywords if matches_keyword(text, keyword))
            has_priority = has_priority or section.get("section_name") in PRIORITY_SECTIONS
    return " ".join(part for part in evidence_parts if part)[:900], matched_sections, mentions, has_priority


def build_defect_cards(paper_id: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined_text = section_text(prioritized_sections(sections))
    parameter_matches = find_matches(combined_text, PARAMETER_RULES)
    cards: list[dict[str, Any]] = []
    for rule in DEFECT_RULES:
        evidence_text, evidence_sections, mentions, has_priority = evidence_for_item(
            sections, rule, CAUSE_KEYWORDS + ACTION_KEYWORDS
        )
        if not evidence_text:
            continue
        confidence = confidence_from_evidence([rule.name], evidence_sections, mentions, has_priority)
        cards.append(
            {
                "defect_type": rule.name,
                "possible_causes": pick_evidence_sentences(evidence_text, CAUSE_KEYWORDS, max_sentences=1),
                "related_parameters": parameter_matches,
                "suggested_actions": pick_evidence_sentences(evidence_text, ACTION_KEYWORDS, max_sentences=1),
                "evidence_paper_id": paper_id,
                "evidence_text": evidence_text,
                "confidence": confidence,
            }
        )
    return cards


def build_method_cards(paper_id: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined_text = section_text(prioritized_sections(sections))
    quality_targets = find_matches(combined_text, QUALITY_RULES)
    cards: list[dict[str, Any]] = []
    for rule in METHOD_RULES:
        evidence_text, evidence_sections, mentions, has_priority = evidence_for_item(
            sections, rule, EXPERIMENT_KEYWORDS + ACTION_KEYWORDS
        )
        if not evidence_text:
            continue
        confidence = confidence_from_evidence([rule.name], evidence_sections, mentions, has_priority)
        cards.append(
            {
                "method_name": rule.name,
                "task": pick_evidence_sentences(evidence_text, ("prediction", "optimization", "classification", "预测", "优化", "分类"), max_sentences=1),
                "input_data": pick_evidence_sentences(evidence_text, ("parameter", "sensor", "data", "参数", "传感", "数据"), max_sentences=1),
                "output_target": quality_targets,
                "advantages": pick_evidence_sentences(evidence_text, ("improve", "accur", "reduce", "提高", "降低", "准确"), max_sentences=1),
                "limitations": pick_evidence_sentences(evidence_text, LIMITATION_KEYWORDS, max_sentences=1),
                "evidence_paper_id": paper_id,
                "confidence": confidence,
            }
        )
    return cards


def build_parameter_cards(paper_id: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined_text = section_text(prioritized_sections(sections))
    defect_matches = find_matches(combined_text, DEFECT_RULES)
    cards: list[dict[str, Any]] = []
    for rule in PARAMETER_RULES:
        evidence_text, evidence_sections, mentions, has_priority = evidence_for_item(
            sections, rule, tuple(keyword for defect in DEFECT_RULES for keyword in defect.keywords)
        )
        if not evidence_text:
            continue
        confidence = confidence_from_evidence([rule.name], evidence_sections, mentions, has_priority)
        cards.append(
            {
                "parameter_name": rule.name,
                "effect_on_quality": pick_evidence_sentences(evidence_text, tuple(keyword for quality in QUALITY_RULES for keyword in quality.keywords), max_sentences=2),
                "related_defects": defect_matches,
                "increase_effect": pick_evidence_sentences(evidence_text, ("increase", "higher", "rise", "增大", "增加", "升高"), max_sentences=1),
                "decrease_effect": pick_evidence_sentences(evidence_text, ("decrease", "lower", "reduce", "降低", "减少", "减小"), max_sentences=1),
                "evidence_paper_id": paper_id,
                "confidence": confidence,
            }
        )
    return cards


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_queue(path: Path, records_by_type: list[tuple[str, list[dict[str, Any]]]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    low_count = 0
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for card_type, records in records_by_type:
            for record in records:
                if record.get("confidence") != "low":
                    continue
                low_count += 1
                writer.writerow(
                    {
                        "card_type": card_type,
                        "evidence_paper_id": record.get("paper_id") or record.get("evidence_paper_id", ""),
                        "item_name": record.get("title")
                        or record.get("defect_type")
                        or record.get("method_name")
                        or record.get("parameter_name")
                        or "",
                        "confidence": "low",
                        "reason": "Rule-based extraction found limited evidence; manual review recommended.",
                    }
                )
    return low_count


def extract_cards(
    input_path: Path = DEFAULT_INPUT,
    paper_cards_path: Path = DEFAULT_PAPER_CARDS,
    defect_cards_path: Path = DEFAULT_DEFECT_CARDS,
    method_cards_path: Path = DEFAULT_METHOD_CARDS,
    parameter_cards_path: Path = DEFAULT_PARAMETER_CARDS,
    review_queue_path: Path = DEFAULT_REVIEW_QUEUE,
    use_local_llm: bool = False,
) -> dict[str, int]:
    if use_local_llm:
        # Placeholder for future Ollama/local model enrichment. No paid API is called.
        pass

    grouped = group_sections(read_sections(input_path))
    paper_cards: list[dict[str, Any]] = []
    defect_cards: list[dict[str, Any]] = []
    method_cards: list[dict[str, Any]] = []
    parameter_cards: list[dict[str, Any]] = []

    for paper_id, sections in grouped.items():
        paper_cards.append(build_paper_card(paper_id, sections))
        defect_cards.extend(build_defect_cards(paper_id, sections))
        method_cards.extend(build_method_cards(paper_id, sections))
        parameter_cards.extend(build_parameter_cards(paper_id, sections))

    write_jsonl(paper_cards_path, paper_cards)
    write_jsonl(defect_cards_path, defect_cards)
    write_jsonl(method_cards_path, method_cards)
    write_jsonl(parameter_cards_path, parameter_cards)
    low_count = write_review_queue(
        review_queue_path,
        [
            ("paper", paper_cards),
            ("defect", defect_cards),
            ("method", method_cards),
            ("parameter", parameter_cards),
        ],
    )

    stats = {
        "paper_cards": len(paper_cards),
        "defect_cards": len(defect_cards),
        "method_cards": len(method_cards),
        "parameter_cards": len(parameter_cards),
        "low_confidence": low_count,
    }
    print(f"paper_cards: {stats['paper_cards']}")
    print(f"defect_cards: {stats['defect_cards']}")
    print(f"method_cards: {stats['method_cards']}")
    print(f"parameter_cards: {stats['parameter_cards']}")
    print(f"低置信度卡片数: {stats['low_confidence']}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract rule-based paper knowledge cards from cleaned sections.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paper_cards", type=Path, default=DEFAULT_PAPER_CARDS)
    parser.add_argument("--defect_cards", type=Path, default=DEFAULT_DEFECT_CARDS)
    parser.add_argument("--method_cards", type=Path, default=DEFAULT_METHOD_CARDS)
    parser.add_argument("--parameter_cards", type=Path, default=DEFAULT_PARAMETER_CARDS)
    parser.add_argument("--review_queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--use_local_llm", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extract_cards(
        input_path=args.input,
        paper_cards_path=args.paper_cards,
        defect_cards_path=args.defect_cards,
        method_cards_path=args.method_cards,
        parameter_cards_path=args.parameter_cards,
        review_queue_path=args.review_queue,
        use_local_llm=args.use_local_llm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
