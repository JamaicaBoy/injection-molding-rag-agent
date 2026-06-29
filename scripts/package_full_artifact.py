from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.full_artifact_common import (  # noqa: E402
    ArtifactError,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    SOURCE_ROOTS,
    sha256_file,
    validate_manifest,
    write_json,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "injection-molding-rag-full-v1.zip"
ALREADY_COMPRESSED_SUFFIXES = {
    ".pdf",
    ".zip",
    ".7z",
    ".rar",
    ".gz",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def collect_source_files(project_root: Path) -> list[tuple[Path, str]]:
    collected: list[tuple[Path, str]] = []
    for relative_root in SOURCE_ROOTS:
        source_root = project_root / Path(relative_root)
        if not source_root.is_dir():
            raise ArtifactError(f"Required artifact source directory is missing: {relative_root}")

        visited_directories: set[str] = set()
        root_file_count = 0
        for current, directories, files in os.walk(source_root, followlinks=True):
            directories.sort(key=str.casefold)
            files.sort(key=str.casefold)
            resolved_current = os.path.normcase(str(Path(current).resolve()))
            if resolved_current in visited_directories:
                directories.clear()
                continue
            visited_directories.add(resolved_current)
            relative_directory = Path(current).relative_to(source_root)
            for file_name in files:
                source_path = Path(current) / file_name
                if not source_path.is_file():
                    continue
                archive_path = (Path(relative_root) / relative_directory / file_name).as_posix()
                collected.append((source_path, archive_path))
                root_file_count += 1
        if root_file_count == 0:
            raise ArtifactError(f"Required artifact source directory is empty: {relative_root}")
    return sorted(collected, key=lambda item: item[1].casefold())


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_chunks(chunks_path: Path) -> tuple[int, int]:
    chunk_count = 0
    paper_ids: set[str] = set()
    with chunks_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"Invalid JSON in {chunks_path} at line {line_number}.") from exc
            chunk_count += 1
            if record.get("paper_id"):
                paper_ids.add(str(record["paper_id"]))
    if chunk_count == 0:
        raise ArtifactError(f"Chunks file is empty: {chunks_path}")
    return chunk_count, len(paper_ids)


def inspect_chroma(project_root: Path, collection_name: str) -> dict[str, Any]:
    try:
        import chromadb
    except Exception as exc:
        raise ArtifactError("chromadb is required to inspect the vector store before packaging.") from exc

    persist_dir = project_root / "vector_store" / "chroma"
    if not persist_dir.exists():
        raise ArtifactError("Chroma persist directory is missing: vector_store/chroma")
    client = chromadb.PersistentClient(path=str(persist_dir.resolve()))
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise ArtifactError(f"Chroma collection does not exist: {collection_name}") from exc
    collection_count = collection.count()
    if collection_count <= 0:
        raise ArtifactError(f"Chroma collection is empty: {collection_name}")
    response = collection.get(limit=1, include=["embeddings"])
    embeddings = response.get("embeddings")
    embedding_dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) else 0
    return {
        "persist_dir": "vector_store/chroma",
        "collection_name": collection_name,
        "collection_count": collection_count,
        "embedding_dimension": embedding_dimension,
        "metadata": dict(collection.metadata or {}),
    }


def build_manifest(
    project_root: Path,
    files: list[tuple[Path, str]],
    expected_pdf_count: int,
    collection_name: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    root_bytes: Counter[str] = Counter()
    for index, (source_path, archive_path) in enumerate(files, start=1):
        size_bytes = source_path.stat().st_size
        records.append(
            {
                "path": archive_path,
                "size_bytes": size_bytes,
                "sha256": sha256_file(source_path),
            }
        )
        root_name = next(root for root in SOURCE_ROOTS if archive_path.startswith(root + "/"))
        root_counts[root_name] += 1
        root_bytes[root_name] += size_bytes
        if index % 100 == 0:
            print(f"hashed_files: {index}/{len(files)}")

    raw_pdf_count = sum(
        1 for record in records if record["path"].startswith("data/raw_papers/") and record["path"].lower().endswith(".pdf")
    )
    if raw_pdf_count != expected_pdf_count:
        raise ArtifactError(f"Expected {expected_pdf_count} raw PDFs, found {raw_pdf_count}.")
    chunk_count, unique_paper_count = inspect_chunks(project_root / "data" / "chunks" / "chunks.jsonl")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_name": "injection-molding-rag-full",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": git_commit(project_root),
        "source_roots": list(SOURCE_ROOTS),
        "summary": {
            "file_count": len(records),
            "total_bytes": sum(record["size_bytes"] for record in records),
            "raw_pdf_count": raw_pdf_count,
            "chunks_count": chunk_count,
            "unique_chunk_paper_ids": unique_paper_count,
            "source_file_counts": dict(root_counts),
            "source_bytes": dict(root_bytes),
        },
        "vector_store": inspect_chroma(project_root, collection_name),
        "files": records,
    }
    validate_manifest(manifest, expected_pdf_count=expected_pdf_count)
    return manifest


def compression_for(path: str, mode: str) -> int:
    if mode == "stored":
        return zipfile.ZIP_STORED
    if mode == "deflate":
        return zipfile.ZIP_DEFLATED
    return zipfile.ZIP_STORED if Path(path).suffix.lower() in ALREADY_COMPRESSED_SUFFIXES else zipfile.ZIP_DEFLATED


def write_archive(
    output_path: Path,
    files: list[tuple[Path, str]],
    manifest: dict[str, Any],
    compression: str,
) -> None:
    temporary_path = output_path.with_name(output_path.name + ".building")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as archive:
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            for index, (source_path, archive_path) in enumerate(files, start=1):
                archive.write(source_path, archive_path, compress_type=compression_for(archive_path, compression))
                if index % 100 == 0:
                    print(f"archived_files: {index}/{len(files)}")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def split_archive(archive_path: Path, split_size_bytes: int) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    with archive_path.open("rb") as source:
        part_number = 1
        while True:
            part_path = archive_path.with_name(f"{archive_path.name}.part{part_number:03d}")
            written = 0
            with part_path.open("wb") as destination:
                while written < split_size_bytes:
                    block = source.read(min(8 * 1024 * 1024, split_size_bytes - written))
                    if not block:
                        break
                    destination.write(block)
                    written += len(block)
            if written == 0:
                part_path.unlink(missing_ok=True)
                break
            parts.append(
                {
                    "name": part_path.name,
                    "size_bytes": written,
                    "sha256": sha256_file(part_path),
                }
            )
            part_number += 1
    return parts


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ArtifactError(f"Output already exists; pass --overwrite to replace it: {existing[0]}")
    if overwrite:
        for path in existing:
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the complete local paper and vector-store artifact.")
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected_pdf_count", type=int, default=896)
    parser.add_argument("--collection", default="injection_molding_chunks")
    parser.add_argument("--compression", choices=["smart", "stored", "deflate"], default="smart")
    parser.add_argument(
        "--split_size_mb",
        type=int,
        default=1900,
        help="Split the completed ZIP into release-sized parts. Use 0 for one unsplit archive.",
    )
    parser.add_argument("--keep_unsplit", action="store_true")
    parser.add_argument("--confirm_publication_rights", choices=["yes"], default=None)
    parser.add_argument("--plan", action="store_true", help="Only print source counts and sizes; do not hash or package.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".zip":
        raise ArtifactError("Full artifact output must use a .zip extension.")
    if args.expected_pdf_count <= 0:
        raise ArtifactError("--expected_pdf_count must be greater than zero.")
    if args.split_size_mb < 0:
        raise ArtifactError("--split_size_mb cannot be negative.")

    files = collect_source_files(project_root)
    raw_pdf_count = sum(
        1 for _, path in files if path.startswith("data/raw_papers/") and path.lower().endswith(".pdf")
    )
    total_bytes = sum(path.stat().st_size for path, _ in files)
    print(f"source_files: {len(files)}")
    print(f"raw_pdf_count: {raw_pdf_count}")
    print(f"source_size_gb: {total_bytes / (1024 ** 3):.3f}")
    if raw_pdf_count != args.expected_pdf_count:
        raise ArtifactError(f"Expected {args.expected_pdf_count} raw PDFs, found {raw_pdf_count}.")
    if args.plan:
        print("plan_only: true")
        return 0
    if args.confirm_publication_rights != "yes":
        raise ArtifactError("Packaging requires --confirm_publication_rights yes.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_manifest = output_path.with_name(output_path.name + ".manifest.json")
    checksum_path = output_path.with_name(output_path.name + ".sha256")
    parts_descriptor = output_path.with_name(output_path.name + ".parts.json")
    old_parts = list(output_path.parent.glob(output_path.name + ".part[0-9][0-9][0-9]"))
    ensure_outputs_available(
        [output_path, sidecar_manifest, checksum_path, parts_descriptor, *old_parts],
        args.overwrite,
    )

    manifest = build_manifest(project_root, files, args.expected_pdf_count, args.collection)
    write_json(sidecar_manifest, manifest)
    write_archive(output_path, files, manifest, args.compression)
    archive_sha256 = sha256_file(output_path)
    checksum_path.write_text(f"{archive_sha256}  {output_path.name}\n", encoding="ascii")

    parts: list[dict[str, Any]] = []
    if args.split_size_mb:
        parts = split_archive(output_path, args.split_size_mb * 1024 * 1024)
        descriptor = {
            "schema_version": SCHEMA_VERSION,
            "archive_name": output_path.name,
            "archive_size_bytes": output_path.stat().st_size,
            "archive_sha256": archive_sha256,
            "manifest_name": sidecar_manifest.name,
            "raw_pdf_count": args.expected_pdf_count,
            "parts": parts,
        }
        write_json(parts_descriptor, descriptor)
        if not args.keep_unsplit:
            output_path.unlink()

    print("Full artifact packaging completed.")
    print(f"manifest: {sidecar_manifest}")
    print(f"archive_sha256: {archive_sha256}")
    print(f"parts: {len(parts)}")
    print(f"output_directory: {output_path.parent}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArtifactError as exc:
        print(f"Packaging failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
