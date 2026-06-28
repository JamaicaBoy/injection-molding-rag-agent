from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import chromadb
import yaml

from src.index.build_vector_index import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, runtime_persist_dir
from src.retrieval.bm25_retriever import BM25Retriever, DEFAULT_CHUNKS
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import merge_results
from src.retrieval.reranker import Reranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_CONFIG = PROJECT_ROOT / "configs" / "retrieval_config.yaml"


EXAMPLE_QUESTIONS = [
    "翘曲可能和哪些工艺参数有关？",
    "保压压力对缩水有什么影响？",
    "有哪些机器学习方法用于注塑质量预测？",
    "PMMA 注塑透过率预测可以参考哪些论文？",
]


def load_vector_store_settings(config_path: Path = DEFAULT_RETRIEVAL_CONFIG) -> tuple[Path, str]:
    persist_dir = DEFAULT_PERSIST_DIR
    collection_name = DEFAULT_COLLECTION
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        vector_store = config.get("vector_store", {})
        configured_dir = vector_store.get("persist_dir")
        if configured_dir:
            candidate = Path(str(configured_dir))
            persist_dir = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        collection_name = str(vector_store.get("collection_name") or collection_name)
    return persist_dir, collection_name


def inspect_retrieval_state(
    chunks_path: Path = DEFAULT_CHUNKS,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict[str, Any]:
    chunks_path = Path(chunks_path)
    persist_dir = Path(persist_dir)
    chunks_count = 0
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as file:
            chunks_count = sum(1 for line in file if line.strip())

    state: dict[str, Any] = {
        "chunks_count": chunks_count,
        "chroma_persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "collection_count": 0,
        "collection_id": "",
        "vector_store_ready": False,
    }
    if not persist_dir.exists():
        return state

    client = chromadb.PersistentClient(path=str(runtime_persist_dir(persist_dir)))
    collection = client.get_collection(collection_name)
    state["collection_count"] = collection.count()
    state["collection_id"] = str(collection.id)
    state["vector_store_ready"] = state["collection_count"] > 0
    return state


def retrieve_debug_results(
    question: str,
    chunks_path: Path = DEFAULT_CHUNKS,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    reranker: Reranker | None = None,
) -> dict[str, Any]:
    state = inspect_retrieval_state(chunks_path, persist_dir, collection_name)
    if not state["vector_store_ready"]:
        raise RuntimeError(f"Chroma collection is empty: {collection_name}")

    bm25 = BM25Retriever(chunks_path=chunks_path)
    dense = DenseRetriever(persist_dir=persist_dir, collection_name=collection_name)
    candidate_k = max(top_k * 2, top_k)
    bm25_candidates = bm25.search(question, top_k=candidate_k)
    dense_candidates = dense.search(question, top_k=candidate_k)
    hybrid_candidates = merge_results(
        bm25_candidates,
        dense_candidates,
        top_k=candidate_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    hybrid_results = hybrid_candidates[:top_k]
    reranked_results = reranker.rerank(question, hybrid_candidates, top_n=top_k) if reranker else []
    return {
        "bm25_results": bm25_candidates[:top_k],
        "dense_results": dense_candidates[:top_k],
        "hybrid_results": hybrid_results,
        "reranked_results": reranked_results,
        "debug_stats": {
            "chunks_count": state["chunks_count"],
            "chroma_persist_dir": state["chroma_persist_dir"],
            "collection_name": state["collection_name"],
            "collection_count": state["collection_count"],
            "dense_results_count": len(dense_candidates[:top_k]),
            "bm25_results_count": len(bm25_candidates[:top_k]),
            "hybrid_results_count": len(hybrid_results),
        },
    }


def print_results(label: str, results: list[dict[str, Any]]) -> None:
    print(f"\n[{label} top{len(results)}]")
    for rank, result in enumerate(results, start=1):
        score_text = f"score={result['score']:.4f}"
        if "rerank_score" in result:
            score_text = (
                f"rerank_score={result['rerank_score']:.4f} "
                f"original_score={result['original_score']:.4f}"
            )
        print(
            f"{rank}. {score_text} source={result['source']} "
            f"paper_id={result['paper_id']} section={result['section_name']} "
            f"chunk_id={result['chunk_id']}"
        )
        print(f"   title: {result['title']}")
        print(f"   preview: {result['text_preview'][:200]}")


def debug_question(
    question: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    reranker: Reranker | None = None,
) -> None:
    candidate_k = max(top_k * 2, top_k)
    bm25_candidates = bm25.search(question, top_k=candidate_k)
    dense_candidates = dense.search(question, top_k=candidate_k)
    hybrid_candidates = merge_results(
        bm25_candidates,
        dense_candidates,
        top_k=candidate_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    print(f"\n{'=' * 80}\n问题: {question}")
    print_results("BM25", bm25_candidates[:top_k])
    print_results("Dense", dense_candidates[:top_k])
    print_results("Hybrid before rerank", hybrid_candidates[:top_k])
    if reranker is not None:
        reranked_results = reranker.rerank(question, hybrid_candidates, top_n=top_k)
        print_results(f"Rerank ({reranker.active_mode})", reranked_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BM25, dense, and hybrid retrieval results.")
    parser.add_argument("question", nargs="?", help="Question to retrieve. Runs built-in examples when omitted.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--persist_dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--dense_weight", type=float, default=0.6)
    parser.add_argument("--bm25_weight", type=float, default=0.4)
    parser.add_argument("--use_rerank", action="store_true")
    parser.add_argument("--rerank_mode", choices=["model", "rule"], default="rule")
    parser.add_argument("--reranker_model", type=Path, default=None, help="Local CrossEncoder/BGE reranker directory.")
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    bm25 = BM25Retriever(chunks_path=args.chunks)
    dense = DenseRetriever(persist_dir=args.persist_dir, collection_name=args.collection)
    reranker = None
    if args.use_rerank:
        reranker = Reranker(mode=args.rerank_mode, model_name=args.reranker_model)
        print(f"rerank requested mode: {reranker.requested_mode}")
        print(f"rerank active mode: {reranker.active_mode}")
        if reranker.fallback_reason:
            print(f"rerank fallback reason: {reranker.fallback_reason}")
    questions = [args.question] if args.question else EXAMPLE_QUESTIONS
    for question in questions:
        debug_question(
            question,
            bm25,
            dense,
            top_k=args.top_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            reranker=reranker,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
