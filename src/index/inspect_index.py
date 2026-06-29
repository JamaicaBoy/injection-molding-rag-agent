from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import chromadb

from src.config import load_corpus_config
from src.index.build_vector_index import collection_names, runtime_persist_dir


_ACTIVE_CORPUS = load_corpus_config()
DEFAULT_PERSIST_DIR = _ACTIVE_CORPUS.vector_persist_dir
DEFAULT_COLLECTION = _ACTIVE_CORPUS.collection_name


def inspect_index(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    sample: int = 3,
) -> dict[str, Any]:
    if not persist_dir.exists():
        raise FileNotFoundError(f"Chroma persist_dir does not exist: {persist_dir}")

    runtime_dir = runtime_persist_dir(persist_dir)
    client = chromadb.PersistentClient(path=str(runtime_dir))
    names = collection_names(client)
    if collection_name not in names:
        raise ValueError(f"Collection `{collection_name}` not found. Available collections: {names}")

    collection = client.get_collection(collection_name)
    collection_metadata = collection.metadata or {}
    metadata_rows = collection.get(limit=sample, include=["metadatas"]).get("metadatas", [])
    embedding_rows = collection.get(limit=1, include=["embeddings"]).get("embeddings", [])
    embedding_dim = len(embedding_rows[0]) if embedding_rows is not None and len(embedding_rows) else None
    stats = {
        "persist_dir": str(persist_dir),
        "runtime_persist_dir": str(runtime_dir),
        "collections": names,
        "collection_name": collection_name,
        "collection_metadata": collection_metadata,
        "embedding_backend": collection_metadata.get("embedding_backend"),
        "embedding_model": collection_metadata.get("embedding_model"),
        "embedding_local_path": collection_metadata.get("embedding_local_path"),
        "collection_count": collection.count(),
        "embedding_dim": embedding_dim,
        "sample_metadatas": metadata_rows,
    }
    print_inspection(stats)
    return stats


def print_inspection(stats: dict[str, Any]) -> None:
    print(f"persist_dir: {stats['persist_dir']}")
    print(f"runtime_persist_dir: {stats['runtime_persist_dir']}")
    print(f"collection 数量: {len(stats['collections'])}")
    print(f"collections: {stats['collections']}")
    print(f"当前 collection: {stats['collection_name']}")
    print(f"collection metadata: {stats['collection_metadata']}")
    print(f"embedding backend: {stats['embedding_backend']}")
    print(f"embedding model: {stats['embedding_model']}")
    print(f"embedding local path: {stats['embedding_local_path']}")
    print(f"chunk 数: {stats['collection_count']}")
    print(f"embedding 维度: {stats['embedding_dim']}")
    print("metadata 样例:")
    for metadata in stats["sample_metadatas"]:
        print(f"  {metadata}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local Chroma collection statistics.")
    parser.add_argument("--persist_dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--sample", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_index(persist_dir=args.persist_dir, collection_name=args.collection, sample=args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
