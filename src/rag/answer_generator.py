from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from src.agent.context_manager import DEFAULT_TOKEN_BUDGET, DEFAULT_TOP_N_EVIDENCE, build_llm_context, render_prompt
from src.index.build_vector_index import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR
from src.rag.citation_guard import DEFAULT_REVIEW_QUEUE, append_review_queue, check_citations
from src.retrieval.bm25_retriever import BM25Retriever, DEFAULT_CHUNKS
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import merge_results
from src.retrieval.query_rewrite import RewrittenQuery, rewrite_query
from src.retrieval.reranker import Reranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_CONFIG = PROJECT_ROOT / "configs" / "app_config.yaml"


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 120.0,
        num_ctx: int = 2048,
        keep_alive: int | str = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {"temperature": 0.1, "num_ctx": self.num_ctx},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"Ollama request failed: HTTP {exc.code}: {detail}") from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        answer = str(result.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        return answer


def unload_ollama_model(
    model: str,
    base_url: str,
    timeout: float = 15.0,
) -> bool:
    payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    evidence_list: list[dict[str, Any]]
    confidence: str
    limitations: list[str]
    need_human_review: bool
    context_debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_llm_config(config_path: Path = DEFAULT_APP_CONFIG) -> tuple[str, str]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    llm = config.get("llm", {})
    return str(llm.get("model", "qwen2.5")), str(llm.get("base_url", "http://localhost:11434"))


def rewrite_to_dict(rewrite: RewrittenQuery | dict[str, Any]) -> dict[str, Any]:
    if isinstance(rewrite, dict):
        return dict(rewrite)
    if is_dataclass(rewrite):
        return asdict(rewrite)
    raise TypeError("query_rewrite must be a RewrittenQuery or dictionary.")


def prepare_evidence(evidence_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for index, result in enumerate(evidence_results, start=1):
        prepared.append(
            {
                "evidence_id": f"E{index}",
                "chunk_id": str(result.get("chunk_id", "")),
                "paper_id": str(result.get("paper_id", "")),
                "title": str(result.get("title", "")),
                "section_name": str(result.get("section_name", "")),
                "chunk_type": str(result.get("chunk_type", "")),
                "score": float(result.get("rerank_score", result.get("score", 0.0))),
                "text_preview": str(result.get("text_preview", ""))[:600],
                "metadata": dict(result.get("metadata") or {}),
            }
        )
    return prepared


class AnswerGenerator:
    def __init__(
        self,
        mode: str = "ollama",
        llm_client: LLMClient | None = None,
        fallback_to_mock: bool = True,
        app_config: Path = DEFAULT_APP_CONFIG,
        review_queue: Path = DEFAULT_REVIEW_QUEUE,
    ) -> None:
        if mode not in {"ollama", "mock"}:
            raise ValueError("Answer generator mode must be `ollama` or `mock`.")
        self.requested_mode = mode
        self.active_mode = mode
        self.fallback_to_mock = fallback_to_mock
        self.review_queue = Path(review_queue)
        self.fallback_reason: str | None = None
        if mode == "ollama" and llm_client is None:
            model, base_url = load_llm_config(app_config)
            llm_client = OllamaClient(model=model, base_url=base_url)
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        query_rewrite: RewrittenQuery | dict[str, Any],
        evidence_results: list[dict[str, Any]],
        conversation_history: list[Any] | None = None,
        conversation_summary: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        top_n_evidence: int = DEFAULT_TOP_N_EVIDENCE,
        conversation_id: str | None = None,
        recent_turns: list[Any] | None = None,
    ) -> GeneratedAnswer:
        self.active_mode = self.requested_mode
        self.fallback_reason = None
        rewrite = rewrite_to_dict(query_rewrite)
        evidence_list = prepare_evidence(evidence_results)
        limitations: list[str] = []
        context_debug: dict[str, Any] | None = None

        if not evidence_list:
            answer = "当前论文库证据不足。"
            limitations.append("no_retrieval_evidence")
        else:
            # All prompt construction goes through context_manager so that
            # conversation history, evidence, and risk rules are assembled
            # under one priority-ordered, token-budgeted policy instead of
            # being formatted ad hoc per call site.
            managed_context = build_llm_context(
                current_query=question,
                query_info=rewrite,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
                reranked_evidence=evidence_list,
                token_budget=token_budget,
                top_n_evidence=top_n_evidence,
                conversation_id=conversation_id,
                recent_turns=recent_turns,
            )
            context_debug = managed_context.context_debug

            if self.active_mode == "mock":
                answer = self._mock_answer(rewrite, evidence_list)
                limitations.append("mock_mode_does_not_generate_new_synthesis")
            else:
                try:
                    if self.llm_client is None:
                        raise RuntimeError("Ollama client is not initialized.")
                    prompt = render_prompt(managed_context.llm_context, query_info=rewrite)
                    answer = self.llm_client.generate(managed_context.llm_context["system_instruction"], prompt)
                except Exception as exc:
                    if not self.fallback_to_mock:
                        raise
                    self.active_mode = "mock"
                    self.fallback_reason = f"{type(exc).__name__}: {exc}"
                    answer = self._mock_answer(rewrite, evidence_list)
                    limitations.append(f"ollama_unavailable_fallback_to_mock: {self.fallback_reason}")

        guard = check_citations(answer, evidence_list)
        limitations.extend(guard.issues)
        query_high_risk = str(rewrite.get("risk_level", "")).lower() == "high"
        if query_high_risk:
            limitations.append("direct_production_parameter_request_requires_engineer_review")
        limitations = list(dict.fromkeys(limitations))

        need_human_review = guard.high_risk or query_high_risk
        confidence = self._confidence(evidence_list, guard.high_risk, self.active_mode)
        if need_human_review:
            reasons = [*guard.issues]
            if query_high_risk:
                reasons.append("high_risk_production_parameter_request")
            append_review_queue(question, evidence_list, confidence, list(dict.fromkeys(reasons)), self.review_queue)

        return GeneratedAnswer(
            answer=answer,
            evidence_list=evidence_list,
            confidence=confidence,
            limitations=limitations,
            need_human_review=need_human_review,
            context_debug=context_debug,
        )

    @staticmethod
    def _mock_answer(rewrite: dict[str, Any], evidence_list: list[dict[str, Any]]) -> str:
        lines = ["Mock 模式仅整理检索证据，不生成超出证据的新结论。"]
        for evidence in evidence_list[:3]:
            preview = str(evidence.get("text_preview", "")).strip()
            if preview:
                lines.append(f"- 候选证据：{preview} [{evidence['evidence_id']}]")
        if str(rewrite.get("risk_level", "")).lower() == "high":
            lines.append("上述内容只能作为候选方向，不能作为直接生产指令，需要人工审核。")
        return "\n".join(lines) if len(lines) > 1 else "当前论文库证据不足。"

    @staticmethod
    def _confidence(evidence_list: list[dict[str, Any]], guard_high_risk: bool, mode: str) -> str:
        if not evidence_list or guard_high_risk:
            return "low"
        if mode == "mock":
            return "low"
        return "high" if len(evidence_list) >= 3 else "medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded injection-molding RAG answer.")
    parser.add_argument("query", nargs="+", help="User question.")
    parser.add_argument("--llm_mode", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--persist_dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    question = " ".join(args.query)
    rewrite = rewrite_query(question)
    candidate_k = max(args.top_k * 2, args.top_k)
    bm25 = BM25Retriever(args.chunks)
    dense = DenseRetriever(args.persist_dir, args.collection)
    hybrid = merge_results(
        bm25.search(rewrite.normalized_query, top_k=candidate_k),
        dense.search(rewrite.normalized_query, top_k=candidate_k),
        top_k=candidate_k,
    )
    evidence = Reranker(mode="rule").rerank(question, hybrid, top_n=args.top_k)
    generator = AnswerGenerator(mode=args.llm_mode)
    result = generator.generate(question, rewrite, evidence)
    output = result.to_dict()
    output["llm_mode"] = generator.active_mode
    if generator.fallback_reason:
        output["fallback_reason"] = generator.fallback_reason
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
