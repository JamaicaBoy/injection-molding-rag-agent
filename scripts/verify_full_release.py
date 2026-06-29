"""Verify an extracted full_release_no_pdf_v1 directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "artifact_config.yaml"
ARTIFACT_NAME = "full_release_no_pdf_v1"
READ_BLOCK_SIZE = 8 * 1024 * 1024
REQUIRED_FILES = (
    "data/chunks/full_chunks.jsonl",
    "data/metadata/paper_metadata.csv",
    "data/processed/full_paper_cards.jsonl",
)
OPTIONAL_FILES = (
    "data/processed/defect_cards.jsonl",
    "data/processed/parameter_cards.jsonl",
)
VECTOR_ROOT = "vector_store/chroma_full"


class ReleaseVerificationError(RuntimeError):
    pass


def load_artifact_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = ("artifact_name", "artifact_dir", "collection_name")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ReleaseVerificationError(f"artifact_config.yaml missing fields: {missing}")
    if config["artifact_name"] != ARTIFACT_NAME:
        raise ReleaseVerificationError(f"artifact_name must be {ARTIFACT_NAME}")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_artifact_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
        raise ReleaseVerificationError(f"unsafe manifest path: {relative}")
    destination = (root / Path(*posix.parts)).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as exc:
        raise ReleaseVerificationError(f"manifest path escapes artifact: {relative}") from exc
    return destination


def read_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "release_manifest" / "MANIFEST.json"
    if not path.is_file():
        raise ReleaseVerificationError("release_manifest/MANIFEST.json is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseVerificationError("MANIFEST.json is invalid") from exc
    if manifest.get("artifact_name") != ARTIFACT_NAME or manifest.get("contains_pdf") is not False:
        raise ReleaseVerificationError("manifest artifact identity is invalid")
    archive = manifest.get("archive")
    if archive is not None:
        split = archive.get("split")
        part_count = archive.get("part_count")
        assets = archive.get("assets")
        if not isinstance(split, bool) or not isinstance(part_count, int) or part_count <= 0:
            raise ReleaseVerificationError("manifest archive metadata is invalid")
        if not isinstance(assets, list) or len(assets) != part_count:
            raise ReleaseVerificationError("manifest archive asset list is invalid")
        if split != (part_count > 1):
            raise ReleaseVerificationError("manifest archive split state is inconsistent")
    return manifest


def verify_checksums(artifact_dir: Path, manifest: dict[str, Any]) -> int:
    sums_path = artifact_dir / "release_manifest" / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise ReleaseVerificationError("release_manifest/SHA256SUMS.txt is missing")
    expected = {str(item["path"]): str(item["sha256"]) for item in manifest.get("files", [])}
    missing_required = [relative for relative in REQUIRED_FILES if relative not in expected]
    if missing_required:
        raise ReleaseVerificationError(
            "required files are absent from MANIFEST.json: " + ", ".join(missing_required)
        )
    if not any(relative.startswith(f"{VECTOR_ROOT}/") for relative in expected):
        raise ReleaseVerificationError(
            "vector_store/chroma_full is absent from MANIFEST.json"
        )
    recorded: dict[str, str] = {}
    for line_number, raw_line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            checksum, relative = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ReleaseVerificationError(f"invalid SHA256SUMS line {line_number}") from exc
        recorded[relative] = checksum
    if recorded != expected:
        raise ReleaseVerificationError("SHA256SUMS does not match MANIFEST.json")

    for relative, checksum in recorded.items():
        if "raw_papers" in relative.lower() or relative.lower().endswith(".pdf"):
            raise ReleaseVerificationError(f"forbidden PDF/raw paper entry: {relative}")
        path = safe_artifact_path(artifact_dir, relative)
        if not path.is_file():
            raise ReleaseVerificationError(f"artifact file is missing: {relative}")
        if sha256_file(path) != checksum:
            raise ReleaseVerificationError(f"SHA256 mismatch: {relative}")
    return len(recorded)


def inspect_chunks(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise ReleaseVerificationError("data/chunks/full_chunks.jsonl is missing")
    count = 0
    paper_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseVerificationError(f"invalid chunks JSON at line {line_number}") from exc
            if not record.get("paper_id"):
                raise ReleaseVerificationError(f"chunk line {line_number} has no paper_id")
            count += 1
            paper_ids.add(str(record["paper_id"]))
    if count == 0:
        raise ReleaseVerificationError("full_chunks.jsonl is empty")
    return count, len(paper_ids)


def inspect_chroma(persist_dir: Path, collection_name: str) -> tuple[int, int]:
    if not persist_dir.is_dir():
        raise ReleaseVerificationError("vector_store/chroma_full is missing")
    inspection_code = """
import json
import sys
import chromadb

client = chromadb.PersistentClient(path=sys.argv[1])
collection = client.get_collection(sys.argv[2])
sample = collection.get(limit=1, include=[\"embeddings\"])
embeddings = sample.get(\"embeddings\")
dimension = len(embeddings[0]) if embeddings is not None and len(embeddings) else 0
print(json.dumps({\"count\": collection.count(), \"dimension\": dimension}))
"""
    result = subprocess.run(
        [sys.executable, "-c", inspection_code, str(persist_dir.resolve()), collection_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise ReleaseVerificationError(f"cannot open Chroma collection: {error}")
    try:
        stats = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ReleaseVerificationError("Chroma inspection returned invalid output") from exc
    count = int(stats["count"])
    dimension = int(stats["dimension"])
    if count <= 0 or dimension <= 0:
        raise ReleaseVerificationError("Chroma collection is empty or unreadable")
    return count, dimension


def inspect_chroma_copy(persist_dir: Path, collection_name: str) -> tuple[int, int]:
    """Open a disposable copy because Chroma may update index bookkeeping on read."""
    with tempfile.TemporaryDirectory(prefix="verify-chroma-") as temporary:
        copied_store = Path(temporary) / "chroma_full"
        shutil.copytree(persist_dir, copied_store)
        return inspect_chroma(copied_store, collection_name)


def verify_release(artifact_dir: Path, collection_name: str) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir).resolve()
    manifest = read_manifest(artifact_dir)
    checked_files = verify_checksums(artifact_dir, manifest)
    chunk_count, paper_count = inspect_chunks(
        artifact_dir / "data" / "chunks" / "full_chunks.jsonl"
    )
    vector_count, dimension = inspect_chroma_copy(
        artifact_dir / VECTOR_ROOT, collection_name
    )
    optional_missing = [
        relative for relative in OPTIONAL_FILES if not (artifact_dir / relative).is_file()
    ]
    declared_optional_missing = sorted(str(item) for item in manifest.get("optional_missing", []))
    if sorted(optional_missing) != declared_optional_missing:
        raise ReleaseVerificationError(
            "optional_missing does not match artifact contents: "
            f"manifest={declared_optional_missing}, actual={sorted(optional_missing)}"
        )
    summary = manifest.get("summary", {})
    vector_manifest = manifest.get("vector_store", {})
    expected = (
        int(summary.get("chunk_count", -1)),
        int(summary.get("unique_paper_id_count", -1)),
        int(vector_manifest.get("vector_count", -1)),
        int(vector_manifest.get("embedding_dimension", -1)),
    )
    actual = (chunk_count, paper_count, vector_count, dimension)
    if actual != expected or chunk_count != vector_count:
        raise ReleaseVerificationError(f"manifest/data count mismatch: expected={expected}, actual={actual}")
    return {
        "artifact_name": ARTIFACT_NAME,
        "checked_files": checked_files,
        "chunk_count": chunk_count,
        "unique_paper_ids": paper_count,
        "collection_name": collection_name,
        "vector_count": vector_count,
        "embedding_dimension": dimension,
        "optional_missing": ",".join(optional_missing) if optional_missing else "none",
        "archive_split": bool((manifest.get("archive") or {}).get("split", False)),
        "release_assets": int((manifest.get("archive") or {}).get("part_count", 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_artifact_config()
    artifact_dir = args.artifact_dir or PROJECT_ROOT / str(config["artifact_dir"])
    stats = verify_release(artifact_dir, str(config["collection_name"]))
    print("verification: PASS")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseVerificationError, OSError, ValueError) as exc:
        print(f"verification: FAIL ({exc})", file=sys.stderr)
        raise SystemExit(1) from exc
