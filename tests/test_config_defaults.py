from pathlib import Path

import yaml

from src.config import SUPPORTED_CORPUS_MODES, load_corpus_config


def test_default_models_are_local() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "app_config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["model"]
    assert config["embedding"]["provider"] == "sentence-transformers"
    assert config["embedding"]["model"]


def test_all_corpus_modes_are_configured(monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "corpus_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    monkeypatch.delenv("CORPUS_MODE", raising=False)
    monkeypatch.delenv("PUBLIC_DEMO", raising=False)
    monkeypatch.delenv("IS_PUBLIC_DEMO", raising=False)
    monkeypatch.delenv("STREAMLIT_SHARING_MODE", raising=False)

    assert set(config["modes"]) == set(SUPPORTED_CORPUS_MODES)
    assert config["local_default_mode"] == "full"
    assert config["public_default_mode"] == "public_full_artifact"
    assert load_corpus_config().corpus_mode == "full"
    for mode in SUPPORTED_CORPUS_MODES:
        loaded = load_corpus_config(mode=mode)
        assert loaded.corpus_mode == mode
        assert loaded.chunks_path
        assert loaded.vector_persist_dir
        assert loaded.collection_name

    assert load_corpus_config(mode="dev", prefer_configured=True).collection_name == "injection_papers_dev"
    assert load_corpus_config(mode="selected").collection_name == "injection_papers_selected"
    assert load_corpus_config(mode="full").collection_name == "injection_papers_full"


def test_public_artifact_falls_back_when_missing() -> None:
    config = load_corpus_config(mode="public_full_artifact")

    if not config.configured_chunks_path.exists():
        assert config.effective_mode in {"public_sample", "upload_only"}
        assert config.fallback_mode == config.effective_mode


def test_public_runtime_uses_public_default_and_safe_fallback(monkeypatch) -> None:
    monkeypatch.delenv("CORPUS_MODE", raising=False)
    monkeypatch.setenv("PUBLIC_DEMO", "true")

    config = load_corpus_config()

    assert config.corpus_mode == "public_full_artifact"
    if not config.configured_chunks_path.exists():
        assert config.effective_mode in {"public_sample", "upload_only"}


def test_dev_mode_keeps_verified_legacy_baseline_available() -> None:
    config = load_corpus_config(mode="dev")

    assert config.chunks_path.name in {"dev_chunks.jsonl", "chunks.jsonl"}
    if not config.configured_chunks_path.exists():
        assert config.legacy_fallback_used is True
        assert config.chunks_path.name == "chunks.jsonl"
        assert config.collection_name == "injection_molding_chunks"
