from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_CONFIG = PROJECT_ROOT / "configs" / "corpus_config.yaml"
SUPPORTED_CORPUS_MODES = (
    "dev",
    "selected",
    "full",
    "public_sample",
    "public_full_artifact",
    "upload_only",
)
REQUIRED_MODE_FIELDS = (
    "chunks_path",
    "vector_persist_dir",
    "collection_name",
    "raw_papers_dir",
    "upload_dir",
    "allow_upload",
    "is_public_demo",
)


@dataclass(frozen=True)
class CorpusConfig:
    corpus_mode: str
    effective_mode: str
    chunks_path: Path
    vector_persist_dir: Path
    collection_name: str
    raw_papers_dir: Path
    upload_dir: Path
    allow_upload: bool
    is_public_demo: bool
    configured_chunks_path: Path
    configured_vector_persist_dir: Path
    configured_collection_name: str
    legacy_fallback_used: bool = False
    fallback_mode: str | None = None
    fallback_reason: str | None = None

    def display_path(self, path: Path) -> str:
        absolute = path if path.is_absolute() else PROJECT_ROOT / path
        try:
            return absolute.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(absolute)

    @property
    def chunks_path_label(self) -> str:
        return self.display_path(self.chunks_path)

    @property
    def vector_persist_dir_label(self) -> str:
        return self.display_path(self.vector_persist_dir)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("Corpus configuration root must be a mapping.")
    return config


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_modes(modes: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(modes, dict):
        raise ValueError("Corpus configuration must define a `modes` mapping.")
    missing_modes = [mode for mode in SUPPORTED_CORPUS_MODES if mode not in modes]
    if missing_modes:
        raise ValueError(f"Corpus configuration is missing modes: {missing_modes}")
    validated: dict[str, dict[str, Any]] = {}
    for mode in SUPPORTED_CORPUS_MODES:
        values = modes[mode]
        if not isinstance(values, dict):
            raise ValueError(f"Corpus mode `{mode}` must be a mapping.")
        missing_fields = [field for field in REQUIRED_MODE_FIELDS if field not in values]
        if missing_fields:
            raise ValueError(f"Corpus mode `{mode}` is missing fields: {missing_fields}")
        if not isinstance(values["allow_upload"], bool) or not isinstance(values["is_public_demo"], bool):
            raise ValueError(f"Corpus mode `{mode}` upload/public flags must be booleans.")
        validated[mode] = values
    return validated


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_public_runtime() -> bool:
    return any(
        _env_flag(name)
        for name in (
            "PUBLIC_DEMO",
            "IS_PUBLIC_DEMO",
            "STREAMLIT_SHARING_MODE",
            "STREAMLIT_CLOUD",
            "IS_STREAMLIT_CLOUD",
        )
    )


def _mode_artifacts_ready(values: dict[str, Any]) -> bool:
    chunks_path = _resolve_path(values["chunks_path"])
    vector_path = _resolve_path(values["vector_persist_dir"])
    database = vector_path / "chroma.sqlite3"
    if not chunks_path.is_file() or not vector_path.is_dir() or not database.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True, timeout=2)
        try:
            row = connection.execute(
                "SELECT 1 FROM collections WHERE name = ? LIMIT 1",
                (str(values["collection_name"]),),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return False
    return row is not None


def load_corpus_config(
    mode: str | None = None,
    config_path: Path = DEFAULT_CORPUS_CONFIG,
    prefer_configured: bool = False,
) -> CorpusConfig:
    config = _load_yaml(Path(config_path))
    modes = _validate_modes(config.get("modes"))
    configured_default = (
        config.get("public_default_mode") if _is_public_runtime() else config.get("local_default_mode")
    )
    selected_mode = mode or os.getenv("CORPUS_MODE") or str(
        configured_default or config.get("corpus_mode") or "dev"
    )
    if selected_mode not in SUPPORTED_CORPUS_MODES:
        raise ValueError(
            f"Unsupported corpus mode `{selected_mode}`. Expected one of: {', '.join(SUPPORTED_CORPUS_MODES)}"
        )
    effective_mode = selected_mode
    values = modes[selected_mode]
    configured_chunks = _resolve_path(values["chunks_path"])
    configured_vector = _resolve_path(values["vector_persist_dir"])
    configured_collection = str(values["collection_name"])
    effective_chunks = configured_chunks
    effective_vector = configured_vector
    effective_collection = configured_collection
    fallback_used = False
    fallback_mode: str | None = None
    fallback_reason: str | None = None

    if selected_mode == "public_full_artifact" and not prefer_configured and not _mode_artifacts_ready(values):
        fallback_modes = config.get("public_fallback_modes") or ["public_sample", "upload_only"]
        for candidate in fallback_modes:
            if candidate not in {"public_sample", "upload_only"}:
                continue
            candidate_values = modes[candidate]
            if candidate == "upload_only" or _mode_artifacts_ready(candidate_values):
                effective_mode = candidate
                values = candidate_values
                effective_chunks = _resolve_path(values["chunks_path"])
                effective_vector = _resolve_path(values["vector_persist_dir"])
                effective_collection = str(values["collection_name"])
                fallback_mode = candidate
                fallback_reason = "public full artifact is missing or incomplete"
                break

    fallback = values.get("legacy_fallback")
    preferred_ready = configured_chunks.is_file() and configured_vector.is_dir()
    if not prefer_configured and not preferred_ready and isinstance(fallback, dict):
        fallback_chunks = _resolve_path(fallback.get("chunks_path", ""))
        fallback_vector = _resolve_path(fallback.get("vector_persist_dir", ""))
        fallback_collection = str(fallback.get("collection_name") or "")
        if fallback_chunks.is_file() and fallback_vector.is_dir() and fallback_collection:
            effective_chunks = fallback_chunks
            effective_vector = fallback_vector
            effective_collection = fallback_collection
            fallback_used = True

    return CorpusConfig(
        corpus_mode=selected_mode,
        effective_mode=effective_mode,
        chunks_path=effective_chunks,
        vector_persist_dir=effective_vector,
        collection_name=effective_collection,
        raw_papers_dir=_resolve_path(values["raw_papers_dir"]),
        upload_dir=_resolve_path(values["upload_dir"]),
        allow_upload=values["allow_upload"],
        is_public_demo=values["is_public_demo"],
        configured_chunks_path=configured_chunks,
        configured_vector_persist_dir=configured_vector,
        configured_collection_name=configured_collection,
        legacy_fallback_used=fallback_used,
        fallback_mode=fallback_mode,
        fallback_reason=fallback_reason,
    )
