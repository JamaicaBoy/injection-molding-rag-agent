from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_common import (  # noqa: E402
    create_logger,
    print_ingest_summary,
    print_pipeline_failure,
    require_full_run_confirmation,
    resolve_project_path,
    run_ingest_pipeline,
    selected_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the selected-paper ingest pipeline through chunking.")
    parser.add_argument("--input", type=Path, default=Path("data/selected_papers"))
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--confirm_full_run", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger, log_path = create_logger("run_ingest_selected")
    try:
        input_dir = resolve_project_path(args.input)
        require_full_run_confirmation(input_dir, args.confirm_full_run)
        summary = run_ingest_pipeline(input_dir, selected_paths(), args.max_files, logger)
    except Exception as exc:
        logger.exception("pipeline_failed")
        print_pipeline_failure(exc, log_path)
        return 1
    print_ingest_summary(summary, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
