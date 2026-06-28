import pytest

from src.retrieval.query_rewrite import rewrite_query


def test_rewrites_colloquial_shrinkage_question() -> None:
    result = rewrite_query("缩水咋办")

    assert result.intent == "defect_diagnosis"
    assert result.defect_type == ["sink_mark/shrinkage"]
    assert "如何解决" in result.normalized_query
    assert "shrinkage" in result.expanded_terms
    assert result.risk_level == "medium"


def test_infers_transparent_material_candidates_and_metrics() -> None:
    result = rewrite_query("透明件发雾")

    assert result.quality_metric == ["transmittance/haze"]
    assert result.material == ["PMMA", "PC"]
    assert "transmittance" in result.expanded_terms


def test_identifies_packing_pressure_effect_intent() -> None:
    result = rewrite_query("保压是不是越大越好")

    assert result.intent == "parameter_effect"
    assert result.parameters == ["packing_pressure"]
    assert result.risk_level == "medium"


def test_direct_production_parameter_request_is_high_risk() -> None:
    result = rewrite_query("直接给我一套 PMMA 注塑生产参数，保压压力设置多少？")

    assert result.intent == "parameter_recommendation"
    assert result.risk_level == "high"
    assert "PMMA" in result.material
    assert "packing_pressure" in result.parameters


def test_maps_common_chinese_and_english_defects() -> None:
    result = rewrite_query("翘曲、熔接痕、短射和飞边分别怎么改善？")

    assert result.defect_type == ["warpage", "weld_line", "short_shot", "flash"]
    assert {"warpage", "weld line", "short shot", "flash"}.issubset(result.expanded_terms)


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        rewrite_query("   ")


def test_short_material_names_require_word_boundaries() -> None:
    result = rewrite_query("How should I apply packing pressure?")

    assert "PP" not in result.material
    assert result.parameters == ["packing_pressure"]
