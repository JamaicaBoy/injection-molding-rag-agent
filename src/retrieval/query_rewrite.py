from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Iterable


DEFECT_TERMS = {
    "warpage": ("翘曲", "翘曲变形", "变形", "warpage", "warping"),
    "sink_mark/shrinkage": ("缩水", "缩痕", "收缩", "sink mark", "sink marks", "shrinkage"),
    "weld_line": ("熔接痕", "熔接线", "焊接线", "weld line", "weld lines", "knit line"),
    "short_shot": ("短射", "缺胶", "填充不足", "short shot", "short-shot", "incomplete filling"),
    "flash": ("飞边", "毛边", "披锋", "flash", "flashing"),
    "void": ("气泡", "空洞", "真空孔", "void", "voids", "bubble"),
    "burn_mark": ("烧焦", "焦痕", "烧伤", "burn mark", "burn marks"),
}

MATERIAL_TERMS = {
    "PMMA": ("pmma", "亚克力", "有机玻璃", "聚甲基丙烯酸甲酯"),
    "PC": ("pc", "聚碳酸酯"),
    "ABS": ("abs", "丙烯腈丁二烯苯乙烯"),
    "PP": ("pp", "聚丙烯"),
    "POM": ("pom", "聚甲醛"),
}

PARAMETER_TERMS = {
    "packing_pressure": ("保压压力", "保压", "packing pressure", "holding pressure"),
    "melt_temperature": ("熔体温度", "料温", "熔胶温度", "melt temperature"),
    "mold_temperature": ("模具温度", "模温", "mold temperature", "mould temperature"),
    "injection_speed": ("注射速度", "注塑速度", "射速", "injection speed"),
    "injection_pressure": ("注射压力", "注塑压力", "injection pressure"),
    "cooling_time": ("冷却时间", "cooling time"),
    "holding_time": ("保压时间", "holding time", "packing time"),
}

QUALITY_METRIC_TERMS = {
    "transmittance/haze": ("透过率", "透光率", "雾度", "发雾", "不透明", "transmittance", "haze"),
    "dimensional_accuracy": ("尺寸精度", "尺寸偏差", "dimensional accuracy"),
    "surface_gloss": ("光泽", "高光", "surface gloss", "gloss"),
    "volume_shrinkage": ("体积收缩率", "收缩率", "volume shrinkage", "shrinkage rate"),
    "warpage": ("翘曲量", "翘曲变形量", "warpage value"),
}

COLLOQUIAL_REPLACEMENTS = (
    (r"咋办|怎么办|怎么搞|咋处理", "如何解决"),
    (r"是不是越大越好", "增大是否持续改善"),
    (r"有啥", "有哪些"),
    (r"为啥", "为什么"),
)

DIRECT_PARAMETER_PATTERNS = (
    r"(?:直接|马上)?给(?:我|出)?(?:一套|具体)?(?:生产|注塑|工艺)?参数",
    r"(?:温度|压力|速度|时间|参数)(?:应该|要|需)?(?:设|设置|调|调整)?(?:到|为)?多少",
    r"(?:具体|直接)(?:数值|参数值|设定值)",
    r"(?:多少|几)(?:mpa|℃|度|秒|s)合适",
)


@dataclass(frozen=True)
class RewrittenQuery:
    original_query: str
    normalized_query: str
    intent: str
    defect_type: list[str]
    material: list[str]
    parameters: list[str]
    quality_metric: list[str]
    must_have_terms: list[str]
    expanded_terms: list[str]
    risk_level: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_text(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    for pattern, replacement in COLLOQUIAL_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized.strip(" ，,。.!！?？")


def find_terms(text: str, dictionary: dict[str, tuple[str, ...]]) -> list[str]:
    matches: list[str] = []
    for canonical, aliases in dictionary.items():
        if any(alias_matches(text, alias) for alias in aliases):
            matches.append(canonical)
    return matches


def alias_matches(text: str, alias: str) -> bool:
    normalized_alias = alias.lower()
    if re.fullmatch(r"[a-z0-9+-]+", normalized_alias):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])", text))
    return normalized_alias in text


def unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def infer_intent(query: str, defects: list[str], parameters: list[str], high_risk: bool) -> str:
    if high_risk:
        return "parameter_recommendation"
    if parameters and re.search(r"影响|关系|增大|减小|提高|降低|越大|越小|effect|increase|decrease", query):
        return "parameter_effect"
    if defects and re.search(r"如何解决|原因|为什么|导致|改善|消除|避免|cause|solve|fix|prevent", query):
        return "defect_diagnosis"
    if re.search(r"论文|文献|研究|参考|paper|literature|study", query):
        return "literature_search"
    if re.search(r"预测|机器学习|深度学习|模型|prediction|machine learning|deep learning", query):
        return "method_search"
    if defects:
        return "defect_diagnosis"
    if parameters:
        return "parameter_effect"
    return "general_qa"


def query_risk(intent: str, high_risk: bool) -> str:
    if high_risk:
        return "high"
    if intent in {"defect_diagnosis", "parameter_effect"}:
        return "medium"
    return "low"


class RuleBasedQueryRewriter:
    def rewrite(self, query: str) -> RewrittenQuery:
        original_query = query.strip()
        if not original_query:
            raise ValueError("Query must not be empty.")

        normalized_text = normalize_text(original_query)
        defects = find_terms(normalized_text, DEFECT_TERMS)
        materials = find_terms(normalized_text, MATERIAL_TERMS)
        parameters = find_terms(normalized_text, PARAMETER_TERMS)
        quality_metrics = find_terms(normalized_text, QUALITY_METRIC_TERMS)

        transparent_part = bool(re.search(r"透明件|透明塑件|透明制品|transparent part", normalized_text))
        fogging = "transmittance/haze" in quality_metrics
        if transparent_part and fogging and not materials:
            materials = ["PMMA", "PC"]

        high_risk = any(re.search(pattern, normalized_text, flags=re.IGNORECASE) for pattern in DIRECT_PARAMETER_PATTERNS)
        intent = infer_intent(normalized_text, defects, parameters, high_risk)
        risk_level = query_risk(intent, high_risk)

        must_have_terms = unique([*defects, *materials, *parameters, *quality_metrics])
        expanded_terms = self._expanded_terms(defects, materials, parameters, quality_metrics)
        normalized_query = normalized_text
        if must_have_terms:
            normalized_query = f"{normalized_text} {' '.join(must_have_terms)}"

        return RewrittenQuery(
            original_query=original_query,
            normalized_query=normalized_query,
            intent=intent,
            defect_type=defects,
            material=materials,
            parameters=parameters,
            quality_metric=quality_metrics,
            must_have_terms=must_have_terms,
            expanded_terms=expanded_terms,
            risk_level=risk_level,
        )

    @staticmethod
    def _expanded_terms(
        defects: list[str],
        materials: list[str],
        parameters: list[str],
        quality_metrics: list[str],
    ) -> list[str]:
        expanded: list[str] = []
        for canonical in defects:
            expanded.extend(DEFECT_TERMS[canonical])
        for canonical in materials:
            expanded.extend(MATERIAL_TERMS[canonical])
        for canonical in parameters:
            expanded.extend(PARAMETER_TERMS[canonical])
        for canonical in quality_metrics:
            expanded.extend(QUALITY_METRIC_TERMS[canonical])
        return unique(term for term in expanded if re.search(r"[a-z]", term.lower()))


def rewrite_query(query: str) -> RewrittenQuery:
    return RuleBasedQueryRewriter().rewrite(query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite a colloquial injection-molding query with local rules.")
    parser.add_argument("question", nargs="+", help="Original user question.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = rewrite_query(" ".join(args.question))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
