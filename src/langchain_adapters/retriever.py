from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from src.langchain_adapters.documents import retrieval_results_to_documents
from src.retrieval.hybrid_retriever import HybridRetriever


class HybridRetrieverAdapter(BaseRetriever):
    """Expose the existing hybrid retriever through LangChain's retriever API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: Any = Field(default_factory=HybridRetriever, exclude=True)
    top_k: int = 5
    candidate_k: int | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.hybrid_retriever.search(
            query,
            top_k=self.top_k,
            candidate_k=self.candidate_k,
        )
        return retrieval_results_to_documents(results)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)
