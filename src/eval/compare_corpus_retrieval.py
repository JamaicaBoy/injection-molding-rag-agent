from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from src.config import load_corpus_config
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever, QueryEncoder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "full_corpus_smoke_questions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "dev_vs_full_retrieval_compare.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "eval" / "full_corpus_validation_report.md"
MODES = ("dev", "full")

SearchCallable = Callable[[str, int], list[dict[str, Any]]]


class CorpusRetrievalRunner:
    def __init__(self, mode: str, encoder: QueryEncoder | None = None) -> None:
        corpus = load_corpus_config(mode=mode)
        self.mode = mode
        self.bm25 = BM25Retriever(corpus.chunks_path)
        self.dense = DenseRetriever(
            persist_dir=corpus.vector_persist_dir,
            collection_name=corpus.collection_name,
            encoder=encoder,
        )
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = Reranker(mode="rule")
        self.stats = {
            "mode": mode,
            "paper_count": len(
                {str(chunk.get("paper_id")) for chunk in self.bm25.chunks if chunk.get("paper_id")}
            ),
            "chunk_count": len(self.bm25.chunks),
            "vector_count": self.dense.collection.count(),
            "chunks_path": corpus.chunks_path_label,
            "collection_name": corpus.collection_name,
        }

    @property
    def encoder(self) -> QueryEncoder:
        return self.dense.encoder

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 2, top_k)
        candidates = self.hybrid.search(query, top_k=candidate_k, candidate_k=candidate_k)
        return self.reranker.rerank(query, candidates, top_n=top_k)


def validate_questions(frame: pd.DataFrame) -> None:
    missing = {"query", "category"} - set(frame.columns)
    if missing:
        raise ValueError(f"Smoke question CSV is missing columns: {sorted(missing)}")
    if frame["query"].fillna("").str.strip().eq("").any():
        raise ValueError("Smoke question CSV contains an empty query.")


def result_row(
    query: str,
    category: str,
    mode: str,
    results: list[dict[str, Any]],
    latency_ms: float,
) -> dict[str, Any]:
    paper_ids = [str(item.get("paper_id", "")) for item in results]
    scores = [round(float(item.get("rerank_score", item.get("score", 0.0))), 6) for item in results]
    titles = [str(item.get("title", "")) for item in results]
    sections = [str(item.get("section_name", "")) for item in results]
    return {
        "query": query,
        "category": category,
        "mode": mode,
        "top_paper_ids": json.dumps(paper_ids, ensure_ascii=False),
        "unique_paper_count": len({paper_id for paper_id in paper_ids if paper_id}),
        "top_scores": json.dumps(scores, ensure_ascii=False),
        "top_titles": json.dumps(titles, ensure_ascii=False),
        "top_sections": json.dumps(sections, ensure_ascii=False),
        "result_count": len(results),
        "latency_ms": round(latency_ms, 2),
        "status": "ok",
        "error": "",
    }


def error_row(query: str, category: str, mode: str, exc: Exception, latency_ms: float) -> dict[str, Any]:
    return {
        "query": query,
        "category": category,
        "mode": mode,
        "top_paper_ids": "[]",
        "unique_paper_count": 0,
        "top_scores": "[]",
        "top_titles": "[]",
        "top_sections": "[]",
        "result_count": 0,
        "latency_ms": round(latency_ms, 2),
        "status": "error",
        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
    }


def compare_questions(
    questions: pd.DataFrame,
    runners: dict[str, SearchCallable],
    top_k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in tqdm(questions.to_dict("records"), desc="dev vs full retrieval", unit="question"):
        query = str(source["query"])
        category = str(source["category"])
        for mode in MODES:
            started = time.perf_counter()
            try:
                results = runners[mode](query, top_k)
                rows.append(
                    result_row(
                        query,
                        category,
                        mode,
                        results,
                        (time.perf_counter() - started) * 1000,
                    )
                )
            except Exception as exc:
                rows.append(
                    error_row(
                        query,
                        category,
                        mode,
                        exc,
                        (time.perf_counter() - started) * 1000,
                    )
                )
    return pd.DataFrame(rows)


def decode_list(value: Any) -> list[Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    return decoded if isinstance(decoded, list) else []


def comparison_metrics(result: pd.DataFrame) -> dict[str, Any]:
    successful = result[result["status"] == "ok"]
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rows = successful[successful["mode"] == mode]
        all_papers = {
            str(paper_id)
            for value in rows["top_paper_ids"]
            for paper_id in decode_list(value)
            if paper_id
        }
        by_mode[mode] = {
            "queries": len(rows),
            "aggregate_unique_papers": len(all_papers),
            "mean_unique_papers_per_query": float(rows["unique_paper_count"].mean()) if len(rows) else 0.0,
            "mean_latency_ms": float(rows["latency_ms"].mean()) if len(rows) else 0.0,
            "all_papers": all_papers,
        }

    paired_rows: list[dict[str, Any]] = []
    for query, group in successful.groupby("query", sort=False):
        mode_rows = {str(row["mode"]): row for _, row in group.iterrows()}
        if set(mode_rows) != set(MODES):
            continue
        dev_row = mode_rows["dev"]
        full_row = mode_rows["full"]
        dev_papers = set(decode_list(dev_row["top_paper_ids"]))
        full_papers = set(decode_list(full_row["top_paper_ids"]))
        union = dev_papers | full_papers
        dev_titles = decode_list(dev_row["top_titles"])
        full_titles = decode_list(full_row["top_titles"])
        paired_rows.append(
            {
                "query": query,
                "category": str(dev_row["category"]),
                "dev_unique": int(dev_row["unique_paper_count"]),
                "full_unique": int(full_row["unique_paper_count"]),
                "overlap": len(dev_papers & full_papers),
                "full_only": len(full_papers - dev_papers),
                "jaccard": len(dev_papers & full_papers) / len(union) if union else 0.0,
                "dev_top_title": str(dev_titles[0]) if dev_titles else "",
                "full_top_title": str(full_titles[0]) if full_titles else "",
            }
        )

    full_more = sum(row["full_unique"] > row["dev_unique"] for row in paired_rows)
    tied = sum(row["full_unique"] == row["dev_unique"] for row in paired_rows)
    full_less = sum(row["full_unique"] < row["dev_unique"] for row in paired_rows)
    return {
        "by_mode": by_mode,
        "paired_rows": paired_rows,
        "full_more_queries": full_more,
        "tied_queries": tied,
        "full_less_queries": full_less,
        "mean_jaccard": (
            sum(row["jaccard"] for row in paired_rows) / len(paired_rows) if paired_rows else 0.0
        ),
        "full_retrieves_more": (
            by_mode["full"]["aggregate_unique_papers"] > by_mode["dev"]["aggregate_unique_papers"]
        ),
    }


def write_report(
    report_path: Path,
    result: pd.DataFrame,
    corpus_stats: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    metrics = comparison_metrics(result)
    dev = metrics["by_mode"]["dev"]
    full = metrics["by_mode"]["full"]
    verdict = "是" if metrics["full_retrieves_more"] else "否"
    lines = [
        "# Full Corpus Retrieval Validation Report",
        "",
        f"- 问题数: {result['query'].nunique()}",
        f"- 每种模式 top_k: {top_k}",
        f"- 成功行: {(result['status'] == 'ok').sum()} / {len(result)}",
        f"- Full 是否在本轮问题累计召回中覆盖更多论文: **{verdict}**",
        "",
        "## Corpus Baseline",
        "",
        "| Mode | Corpus papers | Chunks | Vectors | Collection |",
        "|---|---:|---:|---:|---|",
    ]
    for mode in MODES:
        stats = corpus_stats[mode]
        lines.append(
            f"| {mode} | {stats['paper_count']} | {stats['chunk_count']} | "
            f"{stats['vector_count']} | `{stats['collection_name']}` |"
        )
    lines.extend(
        [
            "",
            "## Retrieval Coverage",
            "",
            "| Mode | Aggregate unique papers | Mean unique papers/query | Mean latency (ms) |",
            "|---|---:|---:|---:|",
            f"| dev | {dev['aggregate_unique_papers']} | {dev['mean_unique_papers_per_query']:.2f} | {dev['mean_latency_ms']:.2f} |",
            f"| full | {full['aggregate_unique_papers']} | {full['mean_unique_papers_per_query']:.2f} | {full['mean_latency_ms']:.2f} |",
            "",
            "## Paired Comparison",
            "",
            f"- Full 每题唯一论文数更多: {metrics['full_more_queries']} 题",
            f"- 两者持平: {metrics['tied_queries']} 题",
            f"- Full 更少: {metrics['full_less_queries']} 题",
            f"- Dev/Full top-10 paper ID 平均 Jaccard: {metrics['mean_jaccard']:.3f}",
            "",
            "top_k=10 限制了单题最多可观察到的论文数，因此单题持平并不代表语料规模相同。"
            "累计唯一论文数和 full-only 论文数更适合判断扩容是否真实生效。",
            "",
            "## Query Differences",
            "",
            "| Category | Query | Dev unique | Full unique | Overlap | Full-only | Dev top title | Full top title |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in metrics["paired_rows"]:
        safe_query = row["query"].replace("|", "\\|")
        dev_title = row["dev_top_title"].replace("|", "\\|")
        full_title = row["full_top_title"].replace("|", "\\|")
        lines.append(
            f"| {row['category']} | {safe_query} | {row['dev_unique']} | {row['full_unique']} | "
            f"{row['overlap']} | {row['full_only']} | {dev_title} | {full_title} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Full 不要求每题都优于 dev：更大的语料库会引入更多候选，也可能改变排名或产生同论文多 chunk 竞争。",
            "- 若单题唯一论文数持平但 Full-only 大于 0，说明 full 确实切换了召回来源，只是受 top-10 上限约束。",
            "- 本轮 Dev/Full paper_id 重合度很低；这可能同时来自排名变化和不同 ingest 批次的 ID 命名空间，不能单独据此断言原始文件完全不重合。",
            "- 分数来自各模式内部的 Hybrid + rule rerank，适合比较排序结构；不应把跨语料分数差直接解释为答案质量差。",
            "- 累计覆盖提升证明 full 切换生效，但不等同于相关性必然提升；知识图谱、透明件发雾等问题仍应结合人工 relevance 标注复核。",
            "- 本报告只使用 paper_id、标题、分数和 section 元数据，不包含论文或 chunk 全文。",
            "",
        ]
    )
    errors = result[result["status"] != "ok"]
    if len(errors):
        lines.extend(["## Errors", ""])
        for row in errors.to_dict("records"):
            lines.append(f"- {row['mode']} / {row['query']}: {row['error']}")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return metrics


def run_comparison(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    top_k: int = 10,
    runners: dict[str, SearchCallable] | None = None,
    corpus_stats: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    questions = pd.read_csv(input_path)
    validate_questions(questions)
    if runners is None:
        dev_runner = CorpusRetrievalRunner("dev")
        full_runner = CorpusRetrievalRunner("full", encoder=dev_runner.encoder)
        runner_objects = {"dev": dev_runner, "full": full_runner}
        runners = {mode: runner_objects[mode].search for mode in MODES}
        corpus_stats = {mode: runner_objects[mode].stats for mode in MODES}
    if corpus_stats is None:
        raise ValueError("corpus_stats is required when custom runners are supplied")

    result = compare_questions(questions, runners, top_k=top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    metrics = write_report(report_path, result, corpus_stats, top_k)
    print(f"questions: {questions['query'].nunique()}")
    print(f"successful_rows: {(result['status'] == 'ok').sum()} / {len(result)}")
    print(f"dev_aggregate_unique_papers: {metrics['by_mode']['dev']['aggregate_unique_papers']}")
    print(f"full_aggregate_unique_papers: {metrics['by_mode']['full']['aggregate_unique_papers']}")
    print(f"full_retrieves_more: {metrics['full_retrieves_more']}")
    print(f"output: {output_path}")
    print(f"report: {report_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dev and full local corpus retrieval.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_comparison(args.input, args.output, args.report, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
