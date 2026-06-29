from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.download_full_artifact import extract_archive
from scripts.full_artifact_common import ArtifactError, MANIFEST_NAME, SCHEMA_VERSION, safe_member_name, sha256_file
from scripts.verify_full_artifact import verify_archive


def build_tiny_artifact(tmp_path: Path) -> Path:
    payloads = {
        "data/raw_papers/paper.pdf": b"tiny pdf fixture",
        "data/metadata/papers.csv": b"paper_id,title\n1,fixture\n",
        "data/chunks/chunks.jsonl": b'{"paper_id":"1","text":"fixture"}\n',
        "vector_store/chroma/chroma.sqlite3": b"tiny vector fixture",
    }
    source_dir = tmp_path / "source"
    records = []
    for name, content in payloads.items():
        path = source_dir / Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append({"path": name, "size_bytes": len(content), "sha256": sha256_file(path)})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "raw_pdf_count": 1,
            "chunks_count": 1,
            "unique_chunk_paper_ids": 1,
        },
        "vector_store": {
            "collection_name": "injection_molding_chunks",
            "collection_count": 1,
        },
        "files": records,
    }
    archive_path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest))
        for name in payloads:
            archive.write(source_dir / Path(name), name)
    return archive_path


def test_verify_tiny_full_artifact(tmp_path: Path) -> None:
    stats = verify_archive(build_tiny_artifact(tmp_path), expected_pdf_count=1)

    assert stats["raw_pdf_count"] == 1
    assert stats["file_count"] == 4
    assert stats["collection_name"] == "injection_molding_chunks"


def test_safe_member_name_rejects_path_traversal() -> None:
    with pytest.raises(ArtifactError):
        safe_member_name("../outside.txt")
    with pytest.raises(ArtifactError):
        safe_member_name("C:/outside.txt")


def test_extractor_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ArtifactError):
        extract_archive(archive_path, tmp_path / "extract")
