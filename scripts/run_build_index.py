from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_common import (  # noqa: E402
    create_logger,
    display_path,
    print_pipeline_failure,
    resolve_project_path,
    run_step,
)
from src.config import SUPPORTED_CORPUS_MODES, load_corpus_config  # noqa: E402
from src.index.build_vector_index import build_index  # noqa: E402


def configured_model() -> str | None:
    config_path = PROJECT_ROOT / "configs" / "retrieval_config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    embedding = config.get("embedding", {})
    return embedding.get("local_model_path") or embedding.get("model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Chroma index from an existing chunks JSONL file.")
    parser.add_argument("--corpus_mode", choices=SUPPORTED_CORPUS_MODES, default=None)
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--persist_dir", type=Path, default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger, log_path = create_logger("run_build_index")
    try:
        effective_corpus = load_corpus_config(mode=args.corpus_mode)
        configured_corpus = load_corpus_config(mode=args.corpus_mode, prefer_configured=True)
        corpus = configured_corpus if configured_corpus.chunks_path.is_file() else effective_corpus
        chunks_path = resolve_project_path(args.chunks) if args.chunks else corpus.chunks_path
        persist_dir = resolve_project_path(args.persist_dir) if args.persist_dir else corpus.vector_persist_dir
        collection_name = args.collection or corpus.collection_name
        model = args.model or configured_model()
        if not model:
            raise ValueError("No embedding model is configured. Pass --model explicitly.")
        stats = run_step(
            "build_vector_index",
            lambda: build_index(
                chunks_path=chunks_path,
                persist_dir=persist_dir,
                collection_name=collection_name,
                model_name=model,
                reset=args.reset,
                limit=args.limit,
                batch_size=args.batch_size,
                backend="sentence-transformers",
            ),
            logger,
        )
    except Exception as exc:
        logger.exception("pipeline_failed")
        print_pipeline_failure(exc, log_path)
        return 1

    print("Index build completed.")
    print(f"chunks_input: {display_path(chunks_path)}")
    print(f"collection: {stats['collection_name']}")
    print(f"chunk_count: {stats['collection_count']}")
    print(f"embedding_model: {stats['model_name']}")
    print(f"embedding_dimension: {stats['embedding_dim']}")
    print(f"persist_dir: {display_path(persist_dir)}")
    print(f"log: {display_path(log_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
