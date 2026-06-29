from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_adapters.documents import documents_to_evidence
from src.rag.prompts import SYSTEM_PROMPT, build_answer_prompt


RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{user_prompt}"),
    ]
)


def build_prompt_inputs(
    question: str,
    query_rewrite: dict[str, Any],
    documents: list[Document],
) -> dict[str, str]:
    evidence = documents_to_evidence(documents)
    return {
        "user_prompt": build_answer_prompt(question, query_rewrite, evidence),
    }


def format_rag_prompt(
    question: str,
    query_rewrite: dict[str, Any],
    documents: list[Document],
):
    return RAG_CHAT_PROMPT.invoke(build_prompt_inputs(question, query_rewrite, documents))
