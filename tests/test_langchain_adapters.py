from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.langchain_adapters.documents import retrieval_result_to_document
from src.langchain_adapters.llm import MOCK_RESPONSE, OllamaLLMAdapter
from src.langchain_adapters.prompts import RAG_CHAT_PROMPT, build_prompt_inputs
from src.langchain_adapters.retriever import HybridRetrieverAdapter


class FakeHybridRetriever:
    def search(self, query: str, top_k: int, candidate_k: int | None = None) -> list[dict[str, Any]]:
        del query, candidate_k
        return [
            {
                "chunk_id": "chunk_1",
                "paper_id": "paper_1",
                "title": "Packing pressure and shrinkage",
                "section_name": "Results",
                "chunk_type": "text",
                "score": 0.82,
                "text_preview": "Packing pressure affects shrinkage compensation.",
                "metadata": {"year": 2024},
            }
        ][:top_k]


def test_retrieval_result_to_langchain_document() -> None:
    document = retrieval_result_to_document(FakeHybridRetriever().search("query", 1)[0])

    assert isinstance(document, Document)
    assert document.page_content.startswith("Packing pressure")
    assert document.metadata["paper_id"] == "paper_1"
    assert document.metadata["title"] == "Packing pressure and shrinkage"
    assert document.metadata["section"] == "Results"
    assert document.metadata["score"] == 0.82
    assert document.metadata["chunk_id"] == "chunk_1"


def test_query_documents_prompt_and_mock_llm_pipeline() -> None:
    retriever = HybridRetrieverAdapter(hybrid_retriever=FakeHybridRetriever(), top_k=1)
    documents = retriever.invoke("保压压力对缩水有什么影响？")
    prompt_inputs = build_prompt_inputs(
        "保压压力对缩水有什么影响？",
        {"intent": "parameter_effect", "risk_level": "low"},
        documents,
    )
    prompt_value = RAG_CHAT_PROMPT.invoke(prompt_inputs)
    answer = OllamaLLMAdapter(mode="mock").invoke(prompt_value)

    assert len(documents) == 1
    assert "[E1]" in prompt_inputs["user_prompt"]
    assert "只能基于给定 evidence 回答" in prompt_value.to_messages()[0].content
    assert answer == MOCK_RESPONSE


def test_get_relevant_documents_compatibility_method() -> None:
    retriever = HybridRetrieverAdapter(hybrid_retriever=FakeHybridRetriever(), top_k=1)

    documents = retriever.get_relevant_documents("缩水咋办")

    assert [document.metadata["chunk_id"] for document in documents] == ["chunk_1"]
