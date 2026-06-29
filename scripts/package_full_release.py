"""Build the fixed full_release_no_pdf_v1 GitHub Release archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "artifact_config.yaml"
ARTIFACT_NAME = "full_release_no_pdf_v1"
DEFAULT_SPLIT_SIZE_MB = 1500.0
MAX_PART_SIZE_BYTES = int(DEFAULT_SPLIT_SIZE_MB * 1024**2)
READ_BLOCK_SIZE = 8 * 1024 * 1024
REQUIRED_SOURCE_FILES = (
    "data/chunks/full_chunks.jsonl",
    "data/metadata/paper_metadata.csv",
    "data/processed/full_paper_cards.jsonl",
)
OPTIONAL_SOURCE_FILES = (
    "data/processed/defect_cards.jsonl",
    "data/processed/parameter_cards.jsonl",
)
SOURCE_FILES = REQUIRED_SOURCE_FILES + OPTIONAL_SOURCE_FILES
VECTOR_DIR = "vector_store/chroma_full"
REQUIRED_CONFIG_KEYS = (
    "artifact_name",
    "artifact_source",
    "github_owner",
    "github_repo",
    "release_tag",
    "artifact_dir",
    "chunks_path",
    "vector_persist_dir",
    "collection_name",
)


class ReleasePackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseFile:
    source: Path
    archive_path: str
    size_bytes: int
    sha256: str


def load_artifact_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    missing = [key for key in REQUIRED_CONFIG_KEYS if not config.get(key)]
    if missing:
        raise ReleasePackageError(f"artifact_config.yaml missing fields: {missing}")
    if config["artifact_name"] != ARTIFACT_NAME:
        raise ReleasePackageError(f"artifact_name must be {ARTIFACT_NAME}")
    if config["artifact_source"] != "github_release":
        raise ReleasePackageError("artifact_source must be github_release")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_archive_path(path: str) -> None:
    normalized = path.replace("\\", "/")
    lowered = normalized.lower()
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ReleasePackageError(f"unsafe release path: {path}")
    if "raw_papers" in lowered or lowered.endswith(".pdf"):
        raise ReleasePackageError(f"PDF/raw paper content is forbidden: {path}")


def _walk_directory(source_dir: Path, archive_root: str) -> Iterable[tuple[Path, str]]:
    visited: set[str] = set()
    for current, directories, file_names in os.walk(source_dir, followlinks=True):
        directories.sort(key=str.casefold)
        file_names.sort(key=str.casefold)
        resolved = os.path.normcase(str(Path(current).resolve()))
        if resolved in visited:
            directories.clear()
            continue
        visited.add(resolved)
        relative_dir = Path(current).relative_to(source_dir)
        for file_name in file_names:
            source = Path(current) / file_name
            archive_path = (Path(archive_root) / relative_dir / file_name).as_posix()
            yield source, archive_path


def collect_release_files(project_root: Path = PROJECT_ROOT) -> list[ReleaseFile]:
    project_root = Path(project_root)
    candidates: list[tuple[Path, str]] = []
    missing: list[str] = []
    for relative in REQUIRED_SOURCE_FILES:
        source = project_root / relative
        if not source.is_file():
            missing.append(relative)
        else:
            candidates.append((source, relative))

    for relative in OPTIONAL_SOURCE_FILES:
        source = project_root / relative
        if source.is_file():
            candidates.append((source, relative))

    vector_root = project_root / VECTOR_DIR
    if not vector_root.is_dir():
        missing.append(VECTOR_DIR)
    else:
        vector_files = list(_walk_directory(vector_root, VECTOR_DIR))
        if not vector_files:
            missing.append(f"{VECTOR_DIR} (empty)")
        candidates.extend(vector_files)

    if missing:
        raise ReleasePackageError("missing release sources: " + ", ".join(missing))

    release_files: list[ReleaseFile] = []
    for source, archive_path in sorted(candidates, key=lambda item: item[1].casefold()):
        _validate_archive_path(archive_path)
        release_files.append(
            ReleaseFile(
                source=source,
                archive_path=archive_path,
                size_bytes=source.stat().st_size,
                sha256=sha256_file(source),
            )
        )
    return release_files


def create_release_snapshot(project_root: Path, snapshot_root: Path) -> Path:
    """Copy immutable inputs from an offline, checkpointed Chroma directory."""
    project_root = Path(project_root)
    snapshot_root = Path(snapshot_root)
    for relative in REQUIRED_SOURCE_FILES:
        source = project_root / relative
        if not source.is_file():
            raise ReleasePackageError(f"missing release source: {relative}")
        _validate_archive_path(relative)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for relative in OPTIONAL_SOURCE_FILES:
        source = project_root / relative
        if not source.is_file():
            continue
        _validate_archive_path(relative)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    source_vector = project_root / VECTOR_DIR
    if not source_vector.is_dir():
        raise ReleasePackageError(f"missing release source: {VECTOR_DIR}")
    destination_vector = snapshot_root / VECTOR_DIR
    copied_vector_files = 0
    for source, archive_path in _walk_directory(source_vector, VECTOR_DIR):
        _validate_archive_path(archive_path)
        if source.name in {"chroma.sqlite3", "chroma.sqlite3-wal", "chroma.sqlite3-shm"}:
            continue
        destination = snapshot_root / archive_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_vector_files += 1

    source_database = source_vector.resolve() / "chroma.sqlite3"
    if not source_database.is_file():
        raise ReleasePackageError("Chroma database is missing: vector_store/chroma_full/chroma.sqlite3")
    live_sidecars = [
        source_database.with_name(source_database.name + suffix)
        for suffix in ("-wal", "-shm")
        if source_database.with_name(source_database.name + suffix).exists()
    ]
    if live_sidecars:
        raise ReleasePackageError(
            "Chroma has active WAL files; stop Streamlit/index writers and run packaging again"
        )
    destination_database = destination_vector / "chroma.sqlite3"
    destination_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_database, destination_database)
    if copied_vector_files == 0:
        raise ReleasePackageError("Chroma vector files are missing")
    return snapshot_root


def inspect_chunks(path: Path) -> tuple[int, int]:
    chunk_count = 0
    paper_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleasePackageError(f"invalid chunks JSON at line {line_number}") from exc
            chunk_count += 1
            if record.get("paper_id"):
                paper_ids.add(str(record["paper_id"]))
    if chunk_count == 0:
        raise ReleasePackageError("full_chunks.jsonl is empty")
    return chunk_count, len(paper_ids)


def inspect_chroma(persist_dir: Path, collection_name: str) -> dict[str, Any]:
    inspection_code = """
import json
import sys
import chromadb

client = chromadb.PersistentClient(path=sys.argv[1])
collection = client.get_collection(sys.argv[2])
sample = collection.get(limit=1, include=[\"embeddings\"])
embeddings = sample.get(\"embeddings\")
dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) else 0
print(json.dumps({
    \"collection_name\": sys.argv[2],
    \"vector_count\": collection.count(),
    \"embedding_dimension\": dimension,
    \"embedding_model\": str((collection.metadata or {}).get(\"embedding_model\", \"\")),
}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            inspection_code,
            str(Path(persist_dir).resolve()),
            collection_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise ReleasePackageError(f"cannot open Chroma collection {collection_name}: {error}")
    try:
        stats = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ReleasePackageError("Chroma inspection returned invalid output") from exc
    if int(stats["vector_count"]) <= 0 or int(stats["embedding_dimension"]) <= 0:
        raise ReleasePackageError("Chroma collection is empty or has no embeddings")
    return stats


def git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_manifest(
    release_files: list[ReleaseFile],
    *,
    chunk_count: int,
    paper_count: int,
    chroma: dict[str, Any],
    source_commit: str | None,
    optional_missing: list[str] | None = None,
) -> dict[str, Any]:
    if chroma["vector_count"] != chunk_count:
        raise ReleasePackageError(
            f"chunks/Chroma mismatch: chunks={chunk_count}, vectors={chroma['vector_count']}"
        )
    return {
        "schema_version": 1,
        "artifact_name": ARTIFACT_NAME,
        "artifact_source": "github_release",
        "contains_pdf": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "optional_missing": sorted(optional_missing or []),
        "summary": {
            "payload_file_count": len(release_files),
            "payload_size_bytes": sum(item.size_bytes for item in release_files),
            "chunk_count": chunk_count,
            "unique_paper_id_count": paper_count,
        },
        "vector_store": chroma,
        "files": [
            {
                "path": item.archive_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in release_files
        ],
    }


def write_manifest_files(manifest: dict[str, Any], manifest_dir: Path) -> tuple[Path, Path]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "MANIFEST.json"
    sums_path = manifest_dir / "SHA256SUMS.txt"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in manifest["files"]),
        encoding="utf-8",
    )
    return manifest_path, sums_path


def materialize_artifact_directory(
    release_files: list[ReleaseFile],
    manifest_path: Path,
    sums_path: Path,
    output_dir: Path,
) -> Path:
    """Create the extracted artifact tree expected by verify_full_release.py."""
    output_dir = Path(output_dir).resolve()
    if output_dir.name != ARTIFACT_NAME:
        raise ReleasePackageError(
            f"output_dir must end with {ARTIFACT_NAME}: {output_dir}"
        )

    staging = output_dir.with_name(f"{output_dir.name}.building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for item in release_files:
            _validate_archive_path(item.archive_path)
            destination = staging / Path(item.archive_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, destination)

        manifest_destination = staging / "release_manifest" / "MANIFEST.json"
        sums_destination = staging / "release_manifest" / "SHA256SUMS.txt"
        manifest_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, manifest_destination)
        shutil.copy2(sums_path, sums_destination)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_dir


def build_archive(
    release_files: list[ReleaseFile],
    manifest_path: Path,
    sums_path: Path,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".building")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for item in release_files:
            archive.write(item.source, item.archive_path)
        archive.write(manifest_path, "release_manifest/MANIFEST.json")
        archive.write(sums_path, "release_manifest/SHA256SUMS.txt")
    output_path.unlink(missing_ok=True)
    temporary.replace(output_path)
    return output_path


def verify_archive_checksums(archive_path: Path, manifest: dict[str, Any]) -> int:
    expected = {str(item["path"]): str(item["sha256"]) for item in manifest["files"]}
    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
        names = {info.filename for info in archive.infolist() if not info.is_dir()}
        required = set(expected) | {
            "release_manifest/MANIFEST.json",
            "release_manifest/SHA256SUMS.txt",
        }
        if names != required:
            raise ReleasePackageError("ZIP member list does not match the manifest")
        for relative, checksum in expected.items():
            digest = hashlib.sha256()
            with archive.open(relative) as stream:
                for block in iter(lambda: stream.read(READ_BLOCK_SIZE), b""):
                    digest.update(block)
            if digest.hexdigest() != checksum:
                raise ReleasePackageError(f"ZIP checksum mismatch: {relative}")
    return len(expected)


def split_size_bytes(split_size_mb: float) -> int:
    if split_size_mb <= 0:
        raise ValueError("split_size_mb must be greater than zero")
    size_bytes = int(split_size_mb * 1024**2)
    if size_bytes <= 0:
        raise ValueError("split_size_mb is too small")
    return size_bytes


def archive_metadata(
    archive_path: Path,
    *,
    split_size_mb: float,
    max_part_size: int,
) -> dict[str, Any]:
    archive_size = archive_path.stat().st_size
    split = archive_size > max_part_size
    part_count = (
        (archive_size + max_part_size - 1) // max_part_size if split else 1
    )
    assets = (
        [f"{archive_path.name}.{index:03d}" for index in range(1, part_count + 1)]
        if split
        else [archive_path.name]
    )
    return {
        "split": split,
        "split_size_mb": split_size_mb,
        "part_count": part_count,
        "assets": assets,
    }


def build_archive_with_manifest(
    release_files: list[ReleaseFile],
    manifest: dict[str, Any],
    manifest_dir: Path,
    archive_path: Path,
    *,
    split_size_mb: float,
    max_part_size: int,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Build until the embedded archive metadata matches the final ZIP size."""
    info: dict[str, Any] = {
        "split": False,
        "split_size_mb": split_size_mb,
        "part_count": 1,
        "assets": [archive_path.name],
    }
    for _ in range(5):
        manifest["archive"] = info
        manifest_path, sums_path = write_manifest_files(manifest, manifest_dir)
        archive = build_archive(
            release_files,
            manifest_path,
            sums_path,
            archive_path,
        )
        updated = archive_metadata(
            archive,
            split_size_mb=split_size_mb,
            max_part_size=max_part_size,
        )
        if updated == info:
            return archive, manifest_path, sums_path, info
        info = updated
    raise ReleasePackageError("archive split metadata did not stabilize")


def split_archive(archive_path: Path, max_part_size: int = MAX_PART_SIZE_BYTES) -> list[Path]:
    if max_part_size <= 0:
        raise ValueError("max_part_size must be positive")
    for stale in archive_path.parent.glob(f"{archive_path.name}.[0-9][0-9][0-9]"):
        stale.unlink()
    if archive_path.stat().st_size <= max_part_size:
        return [archive_path]

    parts: list[Path] = []
    with archive_path.open("rb") as source:
        index = 1
        while True:
            block = source.read(max_part_size)
            if not block:
                break
            part = archive_path.with_name(f"{archive_path.name}.{index:03d}")
            part.write_bytes(block)
            parts.append(part)
            index += 1
    archive_path.unlink()
    return parts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Extracted artifact root. The ZIP or numbered ZIP parts are written "
            "beside this directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Alias for --output_dir. Takes precedence when both are provided.",
    )
    parser.add_argument(
        "--split_size_mb",
        type=float,
        default=DEFAULT_SPLIT_SIZE_MB,
        help="Maximum ZIP part size in MiB-compatible MB units (default: 1500).",
    )
    args = parser.parse_args(argv)
    args.output_dir = (
        args.output
        or args.output_dir
        or PROJECT_ROOT / "dist" / ARTIFACT_NAME
    )
    return args


def main() -> int:
    args = parse_args()
    config = load_artifact_config()
    output_dir = args.output_dir.resolve()
    max_part_size = split_size_bytes(args.split_size_mb)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Chroma/HNSW cannot reliably open a copied index below this project's Unicode path.
    # The system temporary directory is ASCII on the supported Windows setup.
    with tempfile.TemporaryDirectory(prefix="release-snapshot-") as temporary:
        snapshot_root = create_release_snapshot(PROJECT_ROOT, Path(temporary))
        chroma = inspect_chroma(snapshot_root / VECTOR_DIR, str(config["collection_name"]))
        release_files = collect_release_files(snapshot_root)
        chunk_count, paper_count = inspect_chunks(
            snapshot_root / "data" / "chunks" / "full_chunks.jsonl"
        )
        optional_missing = [
            relative
            for relative in OPTIONAL_SOURCE_FILES
            if not (snapshot_root / relative).is_file()
        ]
        manifest = build_manifest(
            release_files,
            chunk_count=chunk_count,
            paper_count=paper_count,
            chroma=chroma,
            source_commit=git_commit(PROJECT_ROOT),
            optional_missing=optional_missing,
        )
        archive, manifest_path, sums_path, archive_info = build_archive_with_manifest(
            release_files,
            manifest,
            snapshot_root / "release_manifest",
            output_dir.parent / f"{ARTIFACT_NAME}.zip",
            split_size_mb=args.split_size_mb,
            max_part_size=max_part_size,
        )
        verified_files = verify_archive_checksums(archive, manifest)
        parts = split_archive(archive, max_part_size=max_part_size)
        actual_assets = [part.name for part in parts]
        if actual_assets != archive_info["assets"]:
            raise ReleasePackageError(
                "archive assets do not match MANIFEST.json: "
                f"manifest={archive_info['assets']}, actual={actual_assets}"
            )
        materialize_artifact_directory(
            release_files,
            manifest_path,
            sums_path,
            output_dir,
        )
    print(f"artifact_name: {ARTIFACT_NAME}")
    print(f"payload_files: {len(release_files)}")
    print(f"chunk_count: {chunk_count}")
    print(f"unique_paper_ids: {paper_count}")
    print(f"vector_count: {chroma['vector_count']}")
    print(f"embedding_dimension: {chroma['embedding_dimension']}")
    print(f"artifact_dir: {output_dir}")
    print(f"optional_missing: {','.join(optional_missing) if optional_missing else 'none'}")
    print(f"archive_checksums_verified: {verified_files}")
    print(f"archive_split: {str(archive_info['split']).lower()}")
    print(f"split_size_mb: {args.split_size_mb:g}")
    print(f"release_assets: {len(parts)}")
    for part in parts:
        print(f"asset: {part.name} ({part.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleasePackageError, OSError, ValueError) as exc:
        print(f"package failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
