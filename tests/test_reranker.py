from pathlib import Path
from typing import Any

from src.retrieval.reranker import Reranker


def candidate(
    chunk_id: str,
    section_name: str,
    preview: str,
    score: float = 0.5,
    chunk_type: str = "text",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "paper_id": "paper_1",
        "title": "Injection molding study",
        "section_name": section_name,
        "chunk_type": chunk_type,
        "score": score,
        "source": "bm25+dense",
        "text_preview": preview,
        "metadata": {},
    }


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        return [0.1, 0.9]


def test_rule_reranker_boosts_parameter_card_for_cause_query() -> None:
    candidates = [
        candidate("reference", "References", "unrelated citation list", score=0.8),
        candidate(
            "parameter",
            "Results",
            "card_type: parameter_card packing pressure reduces shrinkage defect",
            score=0.5,
            chunk_type="knowledge_card",
        ),
    ]

    results = Reranker(mode="rule").rerank("保压压力导致缩水的原因和参数建议", candidates, top_n=2)

    assert results[0]["chunk_id"] == "parameter"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]
    assert results[0]["original_score"] == 0.5
    assert results[0]["metadata"]["rerank_mode"] == "rule"


def test_model_reranker_uses_cross_encoder_scores() -> None:
    candidates = [
        candidate("first", "Abstract", "first candidate"),
        candidate("second", "Method", "second candidate"),
    ]

    results = Reranker(mode="model", model=FakeCrossEncoder()).rerank("query", candidates, top_n=1)

    assert results[0]["chunk_id"] == "second"
    assert results[0]["rerank_score"] == 0.9
    assert results[0]["original_score"] == 0.5


def test_missing_local_model_falls_back_to_rule(tmp_path: Path) -> None:
    reranker = Reranker(mode="model", model_name=tmp_path / "missing-model")

    assert reranker.requested_mode == "model"
    assert reranker.active_mode == "rule"
    assert "does not exist" in str(reranker.fallback_reason)

