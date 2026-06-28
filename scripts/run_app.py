from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "src" / "app" / "streamlit_app.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run the local Streamlit RAG demo.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1")
    return parser.parse_known_args()


def main() -> int:
    args, extra = parse_args()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        args.address,
        "--server.port",
        str(args.port),
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
        *extra,
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
