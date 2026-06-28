from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = {
    "pymupdf": "fitz",
    "pandas": "pandas",
    "numpy": "numpy",
    "tqdm": "tqdm",
    "pyyaml": "yaml",
    "chromadb": "chromadb",
    "sentence-transformers": "sentence_transformers",
    "rank-bm25": "rank_bm25",
    "streamlit": "streamlit",
    "pytest": "pytest",
    "loguru": "loguru",
    "pydantic": "pydantic",
    "scikit-learn": "sklearn",
}

REQUIRED_PATHS = [
    "configs/app_config.yaml",
    "configs/retrieval_config.yaml",
    "configs/prompt_config.yaml",
    "data/dev_papers",
    "data/raw_papers",
    "data/metadata",
    "data/interim",
    "data/processed",
    "data/chunks",
    "data/eval",
    "data/logs",
    "src/ingest/__init__.py",
    "src/index/__init__.py",
    "src/retrieval/__init__.py",
    "src/rag/__init__.py",
    "src/agent/__init__.py",
    "src/app/__init__.py",
    "src/eval/__init__.py",
    "src/utils/__init__.py",
    "tests",
    "scripts",
    "outputs",
    "docs",
]


def check_python() -> list[str]:
    errors: list[str] = []
    version = sys.version_info
    print(f"Python: {sys.version.split()[0]}")
    if (version.major, version.minor) not in {(3, 10), (3, 11)}:
        print("WARN: Python 3.10 or 3.11 is recommended for this project.")
    if version.major != 3 or version.minor < 10:
        errors.append("Python 3.10+ is required.")
    return errors


def check_packages() -> list[str]:
    errors: list[str] = []
    print("\nPackages:")
    for package_name, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            errors.append(f"{package_name} import failed: {exc}")
            print(f"  FAIL {package_name}")
        else:
            print(f"  OK   {package_name}")
    return errors


def check_paths() -> list[str]:
    errors: list[str] = []
    print("\nProject paths:")
    for relative_path in REQUIRED_PATHS:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            print(f"  OK   {relative_path}")
        else:
            errors.append(f"Missing path: {relative_path}")
            print(f"  FAIL {relative_path}")
    return errors


def check_config() -> list[str]:
    errors: list[str] = []
    config_path = PROJECT_ROOT / "configs" / "app_config.yaml"
    print("\nConfig:")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except Exception as exc:
        return [f"Failed to read app_config.yaml: {exc}"]

    llm = config.get("llm", {})
    embedding = config.get("embedding", {})
    if llm.get("provider") != "ollama":
        errors.append("Default LLM provider must be ollama.")
    if embedding.get("provider") != "sentence-transformers":
        errors.append("Default embedding provider must be sentence-transformers.")

    print(f"  LLM provider: {llm.get('provider')}")
    print(f"  LLM model: {llm.get('model')}")
    print(f"  Embedding provider: {embedding.get('provider')}")
    print(f"  Embedding model: {embedding.get('model')}")
    return errors


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    errors = []
    errors.extend(check_python())
    errors.extend(check_packages())
    errors.extend(check_paths())
    errors.extend(check_config())

    if errors:
        print("\nEnvironment check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
