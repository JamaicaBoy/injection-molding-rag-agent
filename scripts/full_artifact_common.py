from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "full_artifact_manifest.json"
SCHEMA_VERSION = 1
SOURCE_ROOTS = (
    "data/raw_papers",
    "data/metadata",
    "data/chunks",
    "vector_store",
)
READ_BUFFER_SIZE = 8 * 1024 * 1024


class ArtifactError(RuntimeError):
    pass


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        block = stream.read(READ_BUFFER_SIZE)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as file:
        return sha256_stream(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object: {path}")
    return value


def safe_member_name(name: str) -> str:
    if not name or "\\" in name:
        raise ArtifactError(f"Unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArtifactError(f"Unsafe archive member path: {name!r}")
    if ":" in path.parts[0]:
        raise ArtifactError(f"Unsafe archive member path: {name!r}")
    return path.as_posix()


def safe_destination(root: Path, member_name: str) -> Path:
    normalized = safe_member_name(member_name)
    root = Path(root).resolve()
    destination = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ArtifactError(f"Archive member escapes extraction directory: {member_name}") from exc
    return destination


def validate_manifest(manifest: dict[str, Any], expected_pdf_count: int | None = None) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(
            f"Unsupported manifest schema: {manifest.get('schema_version')}; expected {SCHEMA_VERSION}."
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError("Manifest does not contain a non-empty files list.")

    seen: set[str] = set()
    source_counts = {root: 0 for root in SOURCE_ROOTS}
    pdf_count = 0
    for record in files:
        if not isinstance(record, dict):
            raise ArtifactError("Manifest file entry is not an object.")
        name = safe_member_name(str(record.get("path", "")))
        if name in seen:
            raise ArtifactError(f"Duplicate manifest path: {name}")
        seen.add(name)
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
            raise ArtifactError(f"Invalid file size in manifest: {name}")
        checksum = str(record.get("sha256", ""))
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum.lower()):
            raise ArtifactError(f"Invalid SHA-256 in manifest: {name}")
        for root in SOURCE_ROOTS:
            if name == root or name.startswith(root + "/"):
                source_counts[root] += 1
                break
        if name.startswith("data/raw_papers/") and name.lower().endswith(".pdf"):
            pdf_count += 1

    missing_roots = [root for root, count in source_counts.items() if count == 0]
    if missing_roots:
        raise ArtifactError(f"Manifest is missing required source roots: {missing_roots}")
    recorded_pdf_count = int(manifest.get("summary", {}).get("raw_pdf_count", -1))
    if recorded_pdf_count != pdf_count:
        raise ArtifactError(
            f"Manifest PDF count mismatch: summary={recorded_pdf_count}, files={pdf_count}."
        )
    if expected_pdf_count is not None and pdf_count != expected_pdf_count:
        raise ArtifactError(f"Expected {expected_pdf_count} PDFs, found {pdf_count}.")
    return {
        "file_count": len(files),
        "raw_pdf_count": pdf_count,
        "total_bytes": sum(int(record["size_bytes"]) for record in files),
        "source_counts": source_counts,
    }


def iter_blocks(paths: Iterable[Path]) -> Iterable[bytes]:
    for path in paths:
        with Path(path).open("rb") as file:
            while True:
                block = file.read(READ_BUFFER_SIZE)
                if not block:
                    break
                yield block
