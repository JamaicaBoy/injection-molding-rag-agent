from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.eval_generation import run_generation_evaluation
from src.eval.eval_retrieval import run_retrieval_evaluation
from src.rag.answer_generator import AnswerGenerator


def questions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "保压压力对缩水有什么影响？",
                "intent": "工艺参数影响",
                "expected_keywords": "保压压力|缩水",
                "difficulty": "中",
                "risk_level": "低",
                "notes": "test",
            }
        ]
    )


def fake_retrieval(question: str, top_k: int) -> dict[str, Any]:
    return {
        "query": question,
        "search_type": "hybrid",
        "results": [
            {
                "evidence_id": "E1",
                "paper_id": "paper_1",
                "title": "缩水研究",
                "year": 2024,
                "chunk_id": "chunk_1",
                "source_location": {"page": 2, "section": "Results", "table": None, "figure": None},
                "matched_text": "保压压力会影响缩水。",
                "matched_keywords": ["保压压力", "缩水"],
                "relevance_score": 0.8,
                "rerank_score": 0.8,
                "evidence_quality": "high",
                "citation": "缩水研究 (2024), Results, paper_1",
            }
        ],
        "overall_confidence": 0.8,
        "warnings": [],
    }


def test_retrieval_and_generation_evaluators_write_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "eval_questions.csv"
    retrieval_output = tmp_path / "retrieval_eval.csv"
    generation_output = tmp_path / "generation_eval.csv"
    questions_frame().to_csv(input_path, index=False, encoding="utf-8")

    retrieval = run_retrieval_evaluation(
        input_path=input_path,
        output_path=retrieval_output,
        top_k=5,
        retrieval_runner=fake_retrieval,
    )
    generation = run_generation_evaluation(
        input_path=input_path,
        output_path=generation_output,
        top_k=5,
        retrieval_runner=fake_retrieval,
        generator=AnswerGenerator(mode="mock", review_queue=tmp_path / "review.csv"),
    )

    assert retrieval_output.exists()
    assert retrieval.loc[0, "hit_at_k"] == 1
    assert retrieval.loc[0, "keyword_recall"] == 1.0
    assert generation_output.exists()
    assert generation.loc[0, "status"] == "ok"
    assert generation.loc[0, "evidence_count"] == 1
    assert generation.loc[0, "citation_guard_passed"] == 1

