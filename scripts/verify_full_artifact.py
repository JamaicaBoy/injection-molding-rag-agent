from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.full_artifact_common import (  # noqa: E402
    ArtifactError,
    MANIFEST_NAME,
    iter_blocks,
    read_json,
    safe_destination,
    safe_member_name,
    sha256_file,
    sha256_stream,
    validate_manifest,
)


def read_archive_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        raw_manifest = archive.read(MANIFEST_NAME)
    except KeyError as exc:
        raise ArtifactError(f"Archive does not contain {MANIFEST_NAME}.") from exc
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("Archive manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(manifest, dict):
        raise ArtifactError("Archive manifest must be a JSON object.")
    return manifest


def verify_archive(archive_path: Path, expected_pdf_count: int | None = None) -> dict[str, Any]:
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise ArtifactError(f"Artifact archive does not exist: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ArtifactError(f"Artifact is not a valid ZIP file: {archive_path}")

    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [safe_member_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise ArtifactError("Archive contains duplicate member paths.")
        manifest = read_archive_manifest(archive)
        summary = validate_manifest(manifest, expected_pdf_count=expected_pdf_count)
        records = {str(record["path"]): record for record in manifest["files"]}
        expected_names = set(records) | {MANIFEST_NAME}
        actual_names = set(names)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ArtifactError(f"Archive member set mismatch; missing={missing[:3]}, extra={extra[:3]}")

        info_by_name = {safe_member_name(info.filename): info for info in infos}
        for index, (name, record) in enumerate(records.items(), start=1):
            info = info_by_name[name]
            if info.file_size != int(record["size_bytes"]):
                raise ArtifactError(f"Size mismatch for archive member: {name}")
            with archive.open(info, "r") as member:
                checksum = sha256_stream(member)
            if checksum != str(record["sha256"]):
                raise ArtifactError(f"SHA-256 mismatch for archive member: {name}")
            if index % 100 == 0:
                print(f"verified_files: {index}/{len(records)}")
    return {
        **summary,
        "artifact_type": "zip",
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "chunks_count": int(manifest.get("summary", {}).get("chunks_count", 0)),
        "unique_chunk_paper_ids": int(manifest.get("summary", {}).get("unique_chunk_paper_ids", 0)),
        "collection_name": str(manifest.get("vector_store", {}).get("collection_name", "")),
        "collection_count": int(manifest.get("vector_store", {}).get("collection_count", 0)),
    }


def verify_extracted(root: Path, expected_pdf_count: int | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = read_json(manifest_path)
    summary = validate_manifest(manifest, expected_pdf_count=expected_pdf_count)
    records = manifest["files"]
    for index, record in enumerate(records, start=1):
        name = str(record["path"])
        path = safe_destination(root, name)
        if not path.is_file():
            raise ArtifactError(f"Extracted artifact file is missing: {name}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ArtifactError(f"Size mismatch for extracted file: {name}")
        if sha256_file(path) != str(record["sha256"]):
            raise ArtifactError(f"SHA-256 mismatch for extracted file: {name}")
        if index % 100 == 0:
            print(f"verified_files: {index}/{len(records)}")
    return {
        **summary,
        "artifact_type": "extracted_directory",
        "root": str(root),
        "chunks_count": int(manifest.get("summary", {}).get("chunks_count", 0)),
        "unique_chunk_paper_ids": int(manifest.get("summary", {}).get("unique_chunk_paper_ids", 0)),
        "collection_name": str(manifest.get("vector_store", {}).get("collection_name", "")),
        "collection_count": int(manifest.get("vector_store", {}).get("collection_count", 0)),
    }


def verify_parts_descriptor(descriptor_path: Path, expected_pdf_count: int | None = None) -> dict[str, Any]:
    descriptor_path = Path(descriptor_path)
    descriptor = read_json(descriptor_path)
    parts = descriptor.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ArtifactError("Parts descriptor does not contain any parts.")
    recorded_pdf_count = int(descriptor.get("raw_pdf_count", -1))
    if expected_pdf_count is not None and recorded_pdf_count != expected_pdf_count:
        raise ArtifactError(
            f"Parts descriptor PDF count mismatch: expected {expected_pdf_count}, found {recorded_pdf_count}."
        )

    aggregate = hashlib.sha256()
    total_bytes = 0
    for record in parts:
        name = safe_member_name(str(record.get("name", "")))
        if "/" in name:
            raise ArtifactError(f"Part files must be beside the descriptor: {name}")
        path = descriptor_path.parent / name
        if not path.is_file():
            raise ArtifactError(f"Artifact part is missing: {name}")
        size_bytes = path.stat().st_size
        if size_bytes != int(record.get("size_bytes", -1)):
            raise ArtifactError(f"Artifact part size mismatch: {name}")
        part_checksum = sha256_file(path)
        if part_checksum != str(record.get("sha256", "")):
            raise ArtifactError(f"Artifact part SHA-256 mismatch: {name}")
        total_bytes += size_bytes
        for block in iter_blocks([path]):
            aggregate.update(block)

    if total_bytes != int(descriptor.get("archive_size_bytes", -1)):
        raise ArtifactError("Combined artifact part size does not match the descriptor.")
    archive_checksum = aggregate.hexdigest()
    if archive_checksum != str(descriptor.get("archive_sha256", "")):
        raise ArtifactError("Combined artifact SHA-256 does not match the descriptor.")
    return {
        "artifact_type": "split_parts",
        "descriptor": str(descriptor_path),
        "parts": len(parts),
        "total_bytes": total_bytes,
        "raw_pdf_count": recorded_pdf_count,
        "archive_sha256": archive_checksum,
    }


def verify_artifact(path: Path, expected_pdf_count: int | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        return verify_extracted(path, expected_pdf_count=expected_pdf_count)
    if path.name.endswith(".parts.json"):
        return verify_parts_descriptor(path, expected_pdf_count=expected_pdf_count)
    return verify_archive(path, expected_pdf_count=expected_pdf_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a full artifact archive, split parts, or extracted directory.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected_pdf_count", type=int, default=896)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = verify_artifact(args.artifact, expected_pdf_count=args.expected_pdf_count)
    print("Full artifact verification passed.")
    for key, value in stats.items():
        if key != "source_counts":
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArtifactError as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
