from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from src.retrieval.bm25_retriever import tokenize


SECTION_WEIGHTS = {
    "abstract": 0.12,
    "conclusion": 0.15,
    "results": 0.12,
    "discussion": 0.12,
    "method": 0.05,
    "experiment": 0.05,
    "introduction": 0.02,
    "front matter": -0.03,
    "references": -0.25,
}

CHUNK_TYPE_WEIGHTS = {
    "knowledge_card": 0.08,
    "text": 0.03,
    "table_or_figure_context": 0.02,
}

ADVICE_OR_CAUSE_TERMS = (
    "工艺参数",
    "参数建议",
    "建议",
    "缺陷原因",
    "原因",
    "如何改善",
    "parameter",
    "recommend",
    "cause",
    "defect",
)


class PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any:
        ...


def meaningful_tokens(text: str, expand_query: bool = False) -> set[str]:
    return {token for token in tokenize(text, expand_query=expand_query) if len(token) >= 2}


def candidate_text(result: dict[str, Any]) -> str:
    return " ".join(
        (
            str(result.get("title", "")),
            str(result.get("section_name", "")),
            str(result.get("chunk_type", "")),
            str(result.get("text_preview", "")),
        )
    )


def rule_score(query: str, result: dict[str, Any]) -> float:
    query_tokens = meaningful_tokens(query, expand_query=True)
    document_tokens = meaningful_tokens(candidate_text(result))
    coverage = len(query_tokens & document_tokens) / len(query_tokens) if query_tokens else 0.0

    section = str(result.get("section_name", "")).strip().lower()
    chunk_type = str(result.get("chunk_type", "")).strip().lower()
    section_weight = SECTION_WEIGHTS.get(section, 0.0)
    chunk_type_weight = CHUNK_TYPE_WEIGHTS.get(chunk_type, 0.0)

    query_lower = query.lower()
    metadata = result.get("metadata") or {}
    card_text = " ".join(
        (
            chunk_type,
            str(metadata.get("card_type", "")),
            str(result.get("text_preview", ""))[:80].lower(),
        )
    )
    intent_bonus = 0.0
    if any(term in query_lower for term in ADVICE_OR_CAUSE_TERMS):
        if "defect_card" in card_text or "parameter_card" in card_text:
            intent_bonus = 0.15

    original_score = max(float(result.get("score", 0.0)), 0.0)
    return 0.55 * original_score + 0.35 * coverage + section_weight + chunk_type_weight + intent_bonus


class Reranker:
    def __init__(
        self,
        mode: str = "rule",
        model_name: str | Path | None = None,
        fallback_to_rule: bool = True,
        model: PairScorer | None = None,
    ) -> None:
        if mode not in {"model", "rule"}:
            raise ValueError("Reranker mode must be `model` or `rule`.")

        self.requested_mode = mode
        self.active_mode = mode
        self.model_name = str(model_name) if model_name is not None else None
        self.fallback_reason: str | None = None
        self.model = model

        if mode == "model" and self.model is None:
            try:
                self.model = self._load_local_model(model_name)
            except Exception as exc:
                if not fallback_to_rule:
                    raise
                self.active_mode = "rule"
                self.fallback_reason = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _load_local_model(model_name: str | Path | None) -> PairScorer:
        if model_name is None:
            raise ValueError("Model mode requires a local reranker model path.")
        model_path = Path(model_name).expanduser()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Local reranker model directory does not exist: {model_path}")

        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError("sentence-transformers CrossEncoder is unavailable.") from exc

        try:
            return CrossEncoder(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load local cross-encoder reranker: {model_path}") from exc

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if top_n <= 0 or not candidates:
            return []

        if self.active_mode == "model":
            scores = self._model_scores(query, candidates)
        else:
            scores = [rule_score(query, result) for result in candidates]

        reranked: list[dict[str, Any]] = []
        for result, rerank_score in zip(candidates, scores):
            output = dict(result)
            output["metadata"] = dict(result.get("metadata") or {})
            output["original_score"] = float(result.get("score", 0.0))
            output["rerank_score"] = float(rerank_score)
            output["score"] = float(rerank_score)
            output["metadata"]["rerank_mode"] = self.active_mode
            reranked.append(output)

        return sorted(
            reranked,
            key=lambda result: (-result["rerank_score"], -result["original_score"], result["chunk_id"]),
        )[:top_n]

    def _model_scores(self, query: str, candidates: list[dict[str, Any]]) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model reranker is not initialized.")
        pairs = [(query, candidate_text(result)) for result in candidates]
        predictions = self.model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in predictions]
