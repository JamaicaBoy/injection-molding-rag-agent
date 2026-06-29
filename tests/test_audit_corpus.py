from dataclasses import replace
from pathlib import Path

from src.index.audit_corpus import audit_corpus, infer_corpus_source, load_corpus_settings, mode_output_paths
from src.retrieval.bm25_retriever import DEFAULT_CHUNKS


def test_infer_corpus_source_identifies_dev_exact_match(tmp_path: Path) -> None:
    dev = tmp_path / "data" / "dev_papers"
    selected = tmp_path / "data" / "selected_papers"
    raw = tmp_path / "data" / "raw_papers"
    for directory in (dev, selected, raw):
        directory.mkdir(parents=True)
    (dev / "a.pdf").touch()
    (selected / "a.pdf").touch()
    (selected / "b.pdf").touch()
    (raw / "a.pdf").touch()
    (raw / "b.pdf").touch()

    result = infer_corpus_source({"a.pdf"}, project_root=tmp_path)

    assert result["inferred_mode"] == "dev_papers"
    assert result["source_counts"] == {"dev_papers": 1, "selected_papers": 2, "raw_papers": 2}


def test_infer_corpus_source_identifies_full_raw_match(tmp_path: Path) -> None:
    for name in ("dev_papers", "selected_papers", "raw_papers"):
        (tmp_path / "data" / name).mkdir(parents=True)
    (tmp_path / "data/raw_papers/a.pdf").touch()
    (tmp_path / "data/raw_papers/b.pdf").touch()

    result = infer_corpus_source({"a.pdf", "b.pdf"}, project_root=tmp_path)

    assert result["inferred_mode"] == "raw_papers"


def test_effective_chunks_path_matches_streamlit_default() -> None:
    settings = load_corpus_settings()

    assert settings.chunks_path == DEFAULT_CHUNKS
    assert settings.collection_name == "injection_papers_full"


def test_explicit_mode_uses_configured_paths_and_allows_overrides(tmp_path: Path) -> None:
    settings = load_corpus_settings(mode="full", prefer_configured=True)
    override_chunks = tmp_path / "override.jsonl"

    overridden = load_corpus_settings(
        mode="full",
        prefer_configured=True,
        chunks_override=override_chunks,
        collection_override="override_collection",
    )

    assert settings.chunks_path.name == "full_chunks.jsonl"
    assert settings.collection_name == "injection_papers_full"
    assert overridden.chunks_path == override_chunks
    assert overridden.collection_name == "override_collection"


def test_missing_mode_writes_report_instead_of_raising(tmp_path: Path) -> None:
    settings = load_corpus_settings(mode="full", prefer_configured=True)
    settings = replace(
        settings,
        chunks_path=tmp_path / "missing_chunks.jsonl",
        persist_dir=tmp_path / "missing_chroma",
    )
    report = tmp_path / "report.md"
    stats = tmp_path / "stats.csv"

    result = audit_corpus(settings, report_path=report, stats_path=stats)

    assert result["build_status"] == "not_built_or_incomplete"
    assert result["chunks_status"] == "missing"
    assert result["chroma_status"] == "missing"
    assert "未构建/缺失" in report.read_text(encoding="utf-8")
    assert stats.is_file()


def test_mode_outputs_are_named_by_mode() -> None:
    report, stats = mode_output_paths("public_sample")

    assert report.name == "corpus_audit_public_sample_report.md"
    assert stats.name == "corpus_audit_public_sample_stats.csv"
