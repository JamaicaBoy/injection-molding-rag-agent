import csv
from pathlib import Path

from src.rag.answer_generator import AnswerGenerator
from src.rag.citation_guard import check_citations
from src.rag.prompts import SYSTEM_PROMPT
from src.retrieval.query_rewrite import rewrite_query


def evidence() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "chunk_1",
            "paper_id": "paper_1",
            "title": "Packing pressure and shrinkage",
            "section_name": "Results",
            "chunk_type": "knowledge_card",
            "score": 0.8,
            "text_preview": "Packing pressure reduced shrinkage at 120 MPa in this experiment.",
            "metadata": {},
        }
    ]


class UnsupportedClaimClient:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "建议采用 300 MPa，并参考《不存在的论文》。 [E9]"


def test_prompt_contains_grounding_and_safety_rules() -> None:
    assert "只能基于给定 evidence" in SYSTEM_PROMPT
    assert "不得编造论文名、参数范围、具体数值或实验结论" in SYSTEM_PROMPT
    assert "当前论文库证据不足" in SYSTEM_PROMPT
    assert "不能作为直接生产指令" in SYSTEM_PROMPT
    assert "[E1]" in SYSTEM_PROMPT


def test_citation_guard_accepts_supported_citation_and_value() -> None:
    prepared = [{
        "evidence_id": "E1",
        "title": "Packing pressure and shrinkage",
        "text_preview": "The experiment used 120 MPa.",
    }]
    result = check_citations("该实验使用了 120 MPa。[E1]", prepared)

    assert result.passed
    assert result.citations == ["E1"]


def test_citation_guard_rejects_invalid_citation_value_and_title() -> None:
    prepared = [{"evidence_id": "E1", "title": "Known paper", "text_preview": "No numeric setting."}]
    result = check_citations("建议 300 MPa，参见《Unknown paper》。 [E9]", prepared)

    assert result.high_risk
    assert result.invalid_citations == ["E9"]
    assert "300 MPa" in result.unsupported_values
    assert result.unsupported_titles == ["Unknown paper"]

    supported = check_citations("参见《Known paper》。 [E1]", prepared)
    assert supported.passed


def test_mock_generator_returns_cited_structured_answer(tmp_path: Path) -> None:
    generator = AnswerGenerator(mode="mock", review_queue=tmp_path / "review.csv")
    result = generator.generate("保压压力对缩水有什么影响？", rewrite_query("保压压力对缩水有什么影响？"), evidence())

    assert "[E1]" in result.answer
    assert result.evidence_list[0]["evidence_id"] == "E1"
    assert result.confidence == "low"
    assert not result.need_human_review


def test_guard_failure_sets_review_and_appends_queue(tmp_path: Path) -> None:
    review_queue = tmp_path / "review.csv"
    generator = AnswerGenerator(
        mode="ollama",
        llm_client=UnsupportedClaimClient(),
        review_queue=review_queue,
    )
    result = generator.generate("请给具体生产参数", rewrite_query("请给具体生产参数"), evidence())

    assert result.need_human_review
    assert result.confidence == "low"
    with review_queue.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["card_type"] == "rag_answer"
    assert "citation_id_not_in_evidence" in rows[0]["reason"]
