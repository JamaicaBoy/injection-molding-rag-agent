from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts.pipeline_common import (
    PROJECT_ROOT,
    RAW_PAPERS_DIR,
    dev_paths,
    require_full_run_confirmation,
    run_step,
    selected_paths,
)
from scripts.run_build_index import configured_model


def test_raw_papers_requires_explicit_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm_full_run yes"):
        require_full_run_confirmation(RAW_PAPERS_DIR, None)

    require_full_run_confirmation(RAW_PAPERS_DIR, "yes")
    require_full_run_confirmation(PROJECT_ROOT / "data/dev_papers", None)


def test_selected_outputs_do_not_overwrite_dev_outputs() -> None:
    dev = dev_paths()
    selected = selected_paths()

    assert dev.parsed_docs != selected.parsed_docs
    assert dev.cleaned_sections != selected.cleaned_sections
    assert dev.paper_cards != selected.paper_cards
    assert dev.chunks != selected.chunks


def test_run_step_captures_command_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_path = tmp_path / "pipeline.log"
    logger = logging.getLogger(f"test-pipeline-{id(tmp_path)}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))

    result = run_step("example", lambda: print("verbose inner output") or 7, logger)

    assert result == 7
    assert capsys.readouterr().out == ""
    assert "verbose inner output" in log_path.read_text(encoding="utf-8")


def test_index_pipeline_prefers_configured_local_model() -> None:
    assert Path(configured_model()).resolve() == Path(r"E:\AI_Models\BAAI\bge-m3").resolve()
