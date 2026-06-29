from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.config import DEFAULT_CORPUS_CONFIG, SUPPORTED_CORPUS_MODES, load_corpus_config
from src.index.build_vector_index import runtime_persist_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_CONFIG = DEFAULT_CORPUS_CONFIG
DEFAULT_REPORT = PROJECT_ROOT / "data" / "logs" / "corpus_audit_report.md"
DEFAULT_STATS = PROJECT_ROOT / "data" / "logs" / "corpus_audit_stats.csv"


@dataclass(frozen=True)
class CorpusSettings:
    corpus_mode: str
    chunks_path: Path
    persist_dir: Path
    collection_name: str
    configured_chunks_path: str | None
    configured_persist_dir: str | None
    configured_collection_name: str | None
    configured_corpus_mode: str | None


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_display_path(path: Path) -> str:
    absolute = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(absolute.resolve())


def load_corpus_settings(
    retrieval_config_path: Path = DEFAULT_RETRIEVAL_CONFIG,
    mode: str | None = None,
    prefer_configured: bool = False,
    chunks_override: Path | None = None,
    persist_override: Path | None = None,
    collection_override: str | None = None,
) -> CorpusSettings:
    corpus = load_corpus_config(
        mode=mode,
        config_path=retrieval_config_path,
        prefer_configured=prefer_configured,
    )
    chunks_path = chunks_override or corpus.chunks_path
    persist_dir = persist_override or corpus.vector_persist_dir
    collection_name = collection_override or corpus.collection_name
    return CorpusSettings(
        corpus_mode=corpus.corpus_mode,
        chunks_path=Path(chunks_path),
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        configured_chunks_path=corpus.display_path(corpus.configured_chunks_path),
        configured_persist_dir=corpus.display_path(corpus.configured_vector_persist_dir),
        configured_collection_name=corpus.configured_collection_name,
        configured_corpus_mode=corpus.corpus_mode,
    )


def read_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid chunks JSONL at line {line_number}: {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Chunk at line {line_number} is not a JSON object: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Chunks file is empty: {path}")
    return rows


def value_counter(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key) or "<missing>").strip() or "<missing>" for row in rows)


def pdf_file_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.name for item in path.rglob("*.pdf") if item.is_file()}


def infer_corpus_source(chunk_file_names: set[str], project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    source_sets = {
        "dev_papers": pdf_file_names(project_root / "data" / "dev_papers"),
        "selected_papers": pdf_file_names(project_root / "data" / "selected_papers"),
        "raw_papers": pdf_file_names(project_root / "data" / "raw_papers"),
    }
    matches = {
        name: len(chunk_file_names & file_names)
        for name, file_names in source_sets.items()
    }
    if chunk_file_names and chunk_file_names == source_sets["dev_papers"]:
        inferred_mode = "dev_papers"
        reason = "Chunk file names exactly match the development-paper directory."
    elif chunk_file_names and chunk_file_names.issubset(source_sets["dev_papers"]):
        inferred_mode = "dev_papers"
        reason = "All chunk file names are contained in the development-paper directory."
    elif chunk_file_names and chunk_file_names.issubset(source_sets["selected_papers"]):
        inferred_mode = "selected_papers"
        reason = "All chunk file names are contained in the selected-paper directory, but not the development set."
    elif chunk_file_names and chunk_file_names == source_sets["raw_papers"]:
        inferred_mode = "raw_papers"
        reason = "Chunk file names exactly match the full raw-paper directory."
    elif chunk_file_names and chunk_file_names.issubset(source_sets["raw_papers"]):
        inferred_mode = "raw_papers_subset"
        reason = "Chunk file names are a subset of the raw-paper directory."
    else:
        inferred_mode = "unknown"
        reason = "Chunk file names do not map cleanly to a known corpus directory."
    return {
        "inferred_mode": inferred_mode,
        "reason": reason,
        "source_counts": {name: len(files) for name, files in source_sets.items()},
        "source_matches": matches,
    }


def read_chroma_metadata(persist_dir: Path, collection_name: str) -> tuple[Any, list[str], list[dict[str, Any]]]:
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist_dir does not exist: {persist_dir}")
    client = chromadb.PersistentClient(path=str(runtime_persist_dir(persist_dir)))
    collection = client.get_collection(collection_name)
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    page_size = 500
    for offset in range(0, collection.count(), page_size):
        response = collection.get(limit=page_size, offset=offset, include=["metadatas"])
        page_ids = [str(value) for value in response.get("ids", [])]
        page_metadatas = [dict(value or {}) for value in response.get("metadatas", [])]
        ids.extend(page_ids)
        metadatas.extend(page_metadatas)
    return collection, ids, metadatas


def audit_corpus(
    settings: CorpusSettings,
    report_path: Path = DEFAULT_REPORT,
    stats_path: Path = DEFAULT_STATS,
) -> dict[str, Any]:
    errors: list[str] = []
    chunks: list[dict[str, Any]] = []
    chunks_status = "missing"
    if settings.chunks_path.is_file():
        try:
            chunks = read_chunks(settings.chunks_path)
            chunks_status = "ready"
        except Exception as exc:
            chunks_status = "error"
            errors.append(f"chunks: {type(exc).__name__}: {exc}")

    collection = None
    chroma_ids: list[str] = []
    chroma_metadatas: list[dict[str, Any]] = []
    chroma_status = "missing"
    if settings.persist_dir.is_dir():
        try:
            collection, chroma_ids, chroma_metadatas = read_chroma_metadata(
                settings.persist_dir,
                settings.collection_name,
            )
            chroma_status = "ready"
        except Exception as exc:
            chroma_status = "error"
            errors.append(f"chroma: {type(exc).__name__}: {exc}")

    chunk_paper_ids = {str(row.get("paper_id")) for row in chunks if row.get("paper_id")}
    chroma_paper_ids = {str(row.get("paper_id")) for row in chroma_metadatas if row.get("paper_id")}
    chunk_ids = Counter(str(row.get("chunk_id") or "") for row in chunks)
    chroma_chunk_ids = Counter(
        str(metadata.get("chunk_id") or chroma_id)
        for chroma_id, metadata in zip(chroma_ids, chroma_metadatas)
    )
    chunk_file_names = {str(row.get("file_name")) for row in chunks if row.get("file_name")}
    source_inference = infer_corpus_source(chunk_file_names)
    if chunks_status != "ready":
        source_inference["inferred_mode"] = "not_built"
        source_inference["reason"] = "The configured chunks file is missing or unreadable."
    distributions = {
        "title": value_counter(chunks, "title"),
        "year": value_counter(chunks, "year"),
        "section": value_counter(chunks, "section_name"),
        "chunk_type": value_counter(chunks, "chunk_type"),
    }
    collection_count = collection.count() if collection is not None else 0
    both_ready = chunks_status == "ready" and chroma_status == "ready"
    count_consistent = len(chunks) == collection_count == len(chroma_metadatas) if both_ready else None
    paper_ids_consistent = chunk_paper_ids == chroma_paper_ids if both_ready else None
    chunk_ids_consistent = chunk_ids == chroma_chunk_ids if both_ready else None
    fully_consistent = (
        bool(count_consistent and paper_ids_consistent and chunk_ids_consistent)
        if both_ready
        else None
    )
    collection_metadata = dict(collection.metadata or {}) if collection is not None else {}

    result = {
        "corpus_mode": settings.corpus_mode,
        "build_status": "ready" if both_ready else "not_built_or_incomplete",
        "chunks_status": chunks_status,
        "chroma_status": chroma_status,
        "errors": errors,
        "chunks_path": project_display_path(settings.chunks_path),
        "configured_chunks_path": settings.configured_chunks_path or "not configured",
        "configured_persist_dir": settings.configured_persist_dir or "not configured",
        "configured_collection_name": settings.configured_collection_name or "not configured",
        "configured_corpus_mode": settings.configured_corpus_mode or "not configured",
        "chunks_count": len(chunks),
        "chunks_unique_paper_ids": len(chunk_paper_ids),
        "chunks_unique_file_names": len(chunk_file_names),
        "chunks_missing_file_name": sum(1 for row in chunks if not row.get("file_name")),
        "persist_dir": project_display_path(settings.persist_dir),
        "runtime_persist_dir": str(runtime_persist_dir(settings.persist_dir)),
        "collection_name": settings.collection_name,
        "collection_count": collection_count,
        "chroma_metadata_count": len(chroma_metadatas),
        "chroma_unique_paper_ids": len(chroma_paper_ids),
        "collection_source": str(collection_metadata.get("source", "")),
        "count_consistent": count_consistent,
        "paper_ids_consistent": paper_ids_consistent,
        "chunk_ids_consistent": chunk_ids_consistent,
        "fully_consistent": fully_consistent,
        "inferred_corpus_mode": source_inference["inferred_mode"],
        "inference_reason": source_inference["reason"],
        "source_counts": source_inference["source_counts"],
        "source_matches": source_inference["source_matches"],
        "distributions": distributions,
    }
    write_stats_csv(stats_path, result)
    write_report(report_path, result)
    return result


def write_stats_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "key", "value", "count", "percentage"]
    summary_keys = [
        "corpus_mode",
        "build_status",
        "chunks_status",
        "chroma_status",
        "errors",
        "chunks_path",
        "configured_chunks_path",
        "configured_persist_dir",
        "configured_collection_name",
        "configured_corpus_mode",
        "chunks_count",
        "chunks_unique_paper_ids",
        "chunks_unique_file_names",
        "chunks_missing_file_name",
        "persist_dir",
        "runtime_persist_dir",
        "collection_name",
        "collection_count",
        "chroma_metadata_count",
        "chroma_unique_paper_ids",
        "collection_source",
        "count_consistent",
        "paper_ids_consistent",
        "chunk_ids_consistent",
        "fully_consistent",
        "inferred_corpus_mode",
        "inference_reason",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in summary_keys:
            writer.writerow({"category": "summary", "key": key, "value": result[key]})
        for source_name, count in result["source_counts"].items():
            writer.writerow({"category": "source_directory", "key": source_name, "count": count})
        for source_name, count in result["source_matches"].items():
            writer.writerow({"category": "source_match", "key": source_name, "count": count})
        denominator = result["chunks_count"]
        for category, counter in result["distributions"].items():
            for key, count in counter.most_common():
                writer.writerow(
                    {
                        "category": f"distribution.{category}",
                        "key": key,
                        "count": count,
                        "percentage": f"{count / denominator:.6f}" if denominator else "0.000000",
                    }
                )


def markdown_distribution(title: str, counter: Counter[str], total: int) -> list[str]:
    lines = [f"### {title}", "", "| Value | Count | Percentage |", "|---|---:|---:|"]
    if not counter:
        lines.append("| Not built / no data | 0 | 0.00% |")
    for value, count in counter.most_common():
        safe_value = value.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {safe_value} | {count} | {count / total:.2%} |")
    lines.append("")
    return lines


def write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_counts = result["source_counts"]
    source_matches = result["source_matches"]
    inferred = result["inferred_corpus_mode"]
    if result["build_status"] != "ready":
        finding = (
            f"**Finding: corpus mode `{result['corpus_mode']}` is 未构建/缺失. "
            f"chunks status=`{result['chunks_status']}`, Chroma status=`{result['chroma_status']}`.**"
        )
    elif inferred == "dev_papers":
        finding = (
            "**Finding: the active App corpus is derived from `data/dev_papers`, not the selected or full "
            "paper library. It currently represents 30 unique papers.**"
        )
    elif inferred == "selected_papers":
        finding = "**Finding: the active App corpus is derived from `data/selected_papers`.**"
    elif inferred == "raw_papers":
        finding = "**Finding: the corpus covers the complete `data/raw_papers` directory.**"
    else:
        finding = f"**Finding: corpus source could not be identified conclusively (`{inferred}`).**"

    lines = [
        "# Corpus Audit Report",
        "",
        finding,
        "",
        "## Build Status",
        "",
        f"- Corpus mode: `{result['corpus_mode']}`",
        f"- Overall status: `{result['build_status']}`",
        f"- Chunks status: `{result['chunks_status']}`",
        f"- Chroma status: `{result['chroma_status']}`",
        "",
        "## Effective App Configuration",
        "",
        f"- Effective chunks path: `{result['chunks_path']}`",
        f"- Configured chunks path: `{result['configured_chunks_path']}`",
        f"- Configured vector persist dir: `{result['configured_persist_dir']}`",
        f"- Configured collection: `{result['configured_collection_name']}`",
        f"- Configured corpus mode: `{result['configured_corpus_mode']}`",
        "- Streamlit/BM25 path source: `load_corpus_config()`",
        f"- Chroma persist dir: `{result['persist_dir']}`",
        f"- Chroma runtime dir: `{result['runtime_persist_dir']}`",
        f"- Collection: `{result['collection_name']}`",
        f"- Collection source metadata: `{result['collection_source']}`",
        "",
        "## Corpus Counts",
        "",
        f"- Chunks: {result['chunks_count']}",
        f"- Unique chunk `paper_id`: {result['chunks_unique_paper_ids']}",
        f"- Unique non-empty chunk `file_name`: {result['chunks_unique_file_names']}",
        f"- Chunks without `file_name`: {result['chunks_missing_file_name']} (knowledge-card chunks may omit it)",
        f"- Chroma vectors: {result['collection_count']}",
        f"- Chroma metadata rows: {result['chroma_metadata_count']}",
        f"- Unique Chroma metadata `paper_id`: {result['chroma_unique_paper_ids']}",
        "",
        "## Consistency",
        "",
        f"- Counts consistent: {result['count_consistent']}",
        f"- Paper ID sets consistent: {result['paper_ids_consistent']}",
        f"- Chunk ID multisets consistent: {result['chunk_ids_consistent']}",
        f"- Overall chunks/Chroma consistency: {result['fully_consistent']}",
        "",
        "## Source Attribution",
        "",
        f"- Inferred corpus mode: `{inferred}`",
        f"- Reason: {result['inference_reason']}",
        f"- `data/dev_papers`: {source_counts['dev_papers']} PDFs; {source_matches['dev_papers']} chunk file-name matches",
        f"- `data/selected_papers`: {source_counts['selected_papers']} PDFs; {source_matches['selected_papers']} chunk file-name matches",
        f"- `data/raw_papers`: {source_counts['raw_papers']} PDFs; {source_matches['raw_papers']} chunk file-name matches",
        "",
        "The filename comparison reads directory entries only; no PDF full text is opened.",
        "",
    ]
    if result["errors"]:
        lines.extend(["## Missing / Error Details", ""])
        lines.extend(f"- {error}" for error in result["errors"])
        lines.append("")
    lines.extend(["## Distributions", ""])
    total = result["chunks_count"]
    lines.extend(markdown_distribution("Title", result["distributions"]["title"], total))
    lines.extend(markdown_distribution("Year", result["distributions"]["year"], total))
    lines.extend(markdown_distribution("Section", result["distributions"]["section"], total))
    lines.extend(markdown_distribution("Chunk Type", result["distributions"]["chunk_type"], total))
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_summary(result: dict[str, Any], report_path: Path, stats_path: Path) -> None:
    print(f"corpus_mode: {result['corpus_mode']}")
    print(f"build_status: {result['build_status']}")
    print(f"chunks_status: {result['chunks_status']}")
    print(f"chroma_status: {result['chroma_status']}")
    print(f"chunks_path: {result['chunks_path']}")
    print(f"chunks_count: {result['chunks_count']}")
    print(f"chunks_unique_paper_ids: {result['chunks_unique_paper_ids']}")
    print(f"persist_dir: {result['persist_dir']}")
    print(f"collection_name: {result['collection_name']}")
    print(f"collection_count: {result['collection_count']}")
    print(f"chroma_unique_paper_ids: {result['chroma_unique_paper_ids']}")
    print(f"chunks_chroma_consistent: {result['fully_consistent']}")
    print(f"inferred_corpus_mode: {result['inferred_corpus_mode']}")
    print(f"report: {project_display_path(report_path)}")
    print(f"stats: {project_display_path(stats_path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the effective Streamlit chunks and Chroma corpus.")
    parser.add_argument("--mode", choices=SUPPORTED_CORPUS_MODES, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--persist_dir", type=Path, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--corpus_config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    return parser.parse_args()


def mode_output_paths(mode: str) -> tuple[Path, Path]:
    log_dir = PROJECT_ROOT / "data" / "logs"
    return (
        log_dir / f"corpus_audit_{mode}_report.md",
        log_dir / f"corpus_audit_{mode}_stats.csv",
    )


def main() -> int:
    args = parse_args()
    settings = load_corpus_settings(
        retrieval_config_path=args.corpus_config,
        mode=args.mode,
        prefer_configured=args.mode is not None,
        chunks_override=resolve_project_path(args.chunks) if args.chunks else None,
        persist_override=resolve_project_path(args.persist_dir) if args.persist_dir else None,
        collection_override=args.collection,
    )
    default_report, default_stats = mode_output_paths(settings.corpus_mode)
    report_path = resolve_project_path(args.report) if args.report else default_report
    stats_path = resolve_project_path(args.stats) if args.stats else default_stats
    result = audit_corpus(settings, report_path=report_path, stats_path=stats_path)
    print_summary(result, report_path, stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
