from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest

from scripts.download_full_release import (
    ReleaseDownloadError,
    install_release,
    select_configured_release_assets,
    select_release_assets,
)
from scripts.package_full_release import (
    ARTIFACT_NAME,
    ReleasePackageError,
    build_archive,
    build_archive_with_manifest,
    build_manifest,
    collect_release_files,
    create_release_snapshot,
    inspect_chroma,
    inspect_chunks,
    materialize_artifact_directory,
    parse_args,
    split_archive,
    verify_archive_checksums,
    write_manifest_files,
)
from scripts.verify_full_release import verify_release


def create_release_payload(root: Path) -> None:
    files = {
        "data/chunks/full_chunks.jsonl": json.dumps(
            {"chunk_id": "chunk-1", "paper_id": "paper-1", "text": "short evidence"}
        )
        + "\n",
        "data/metadata/paper_metadata.csv": "paper_id,title\npaper-1,Example\n",
        "data/processed/full_paper_cards.jsonl": '{"paper_id":"paper-1"}\n',
        "data/processed/defect_cards.jsonl": '{"evidence_paper_id":"paper-1"}\n',
        "data/processed/parameter_cards.jsonl": '{"evidence_paper_id":"paper-1"}\n',
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    persist_dir = root / "vector_store" / "chroma_full"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        "injection_papers_full",
        metadata={"embedding_model": "test-model"},
    )
    collection.add(
        ids=["chunk-1"],
        documents=["short evidence"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"paper_id": "paper-1"}],
    )
    del collection
    client.clear_system_cache()


def test_release_package_and_verify_round_trip(tmp_path: Path) -> None:
    create_release_payload(tmp_path)
    chunk_count, paper_count = inspect_chunks(tmp_path / "data/chunks/full_chunks.jsonl")
    chroma = inspect_chroma(
        tmp_path / "vector_store/chroma_full", "injection_papers_full"
    )
    release_files = collect_release_files(tmp_path)
    assert all(not item.archive_path.lower().endswith(".pdf") for item in release_files)
    assert {item.archive_path for item in release_files}.issuperset(
        {
            "data/chunks/full_chunks.jsonl",
            "data/metadata/paper_metadata.csv",
            "data/processed/full_paper_cards.jsonl",
            "data/processed/defect_cards.jsonl",
            "data/processed/parameter_cards.jsonl",
        }
    )

    manifest = build_manifest(
        release_files,
        chunk_count=chunk_count,
        paper_count=paper_count,
        chroma=chroma,
        source_commit=None,
    )
    write_manifest_files(manifest, tmp_path / "release_manifest")

    stats = verify_release(tmp_path, "injection_papers_full")
    assert stats["chunk_count"] == 1
    assert stats["unique_paper_ids"] == 1
    assert stats["vector_count"] == 1
    assert stats["embedding_dimension"] == 2
    assert verify_release(tmp_path, "injection_papers_full") == stats


def test_packager_rejects_pdf_inside_vector_directory(tmp_path: Path) -> None:
    create_release_payload(tmp_path)
    forbidden = tmp_path / "vector_store/chroma_full/paper.pdf"
    forbidden.write_bytes(b"not a real PDF")
    with pytest.raises(ReleasePackageError, match="forbidden"):
        collect_release_files(tmp_path)


def test_snapshot_archive_hashes_remain_consistent(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    snapshot_root = tmp_path / "snapshot"
    output_root = tmp_path / "output"
    create_release_payload(source_root)
    create_release_snapshot(source_root, snapshot_root)
    chroma = inspect_chroma(snapshot_root / "vector_store/chroma_full", "injection_papers_full")
    release_files = collect_release_files(snapshot_root)
    chunk_count, paper_count = inspect_chunks(
        snapshot_root / "data/chunks/full_chunks.jsonl"
    )
    manifest = build_manifest(
        release_files,
        chunk_count=chunk_count,
        paper_count=paper_count,
        chroma=chroma,
        source_commit=None,
    )
    manifest_path, sums_path = write_manifest_files(manifest, output_root / "release_manifest")
    archive = build_archive(
        release_files,
        manifest_path,
        sums_path,
        output_root / f"{ARTIFACT_NAME}.zip",
    )
    assert verify_archive_checksums(archive, manifest) == len(release_files)


def test_materialized_artifact_directory_matches_verifier(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    snapshot_root = tmp_path / "snapshot"
    artifact_root = tmp_path / ARTIFACT_NAME
    create_release_payload(source_root)
    create_release_snapshot(source_root, snapshot_root)
    chunk_count, paper_count = inspect_chunks(
        snapshot_root / "data/chunks/full_chunks.jsonl"
    )
    chroma = inspect_chroma(
        snapshot_root / "vector_store/chroma_full", "injection_papers_full"
    )
    release_files = collect_release_files(snapshot_root)
    manifest = build_manifest(
        release_files,
        chunk_count=chunk_count,
        paper_count=paper_count,
        chroma=chroma,
        source_commit=None,
    )
    manifest_path, sums_path = write_manifest_files(
        manifest, snapshot_root / "release_manifest"
    )

    materialize_artifact_directory(
        release_files, manifest_path, sums_path, artifact_root
    )

    assert (artifact_root / "data/chunks/full_chunks.jsonl").is_file()
    assert (artifact_root / "vector_store/chroma_full/chroma.sqlite3").is_file()
    assert verify_release(artifact_root, "injection_papers_full")["chunk_count"] == 1


def test_optional_cards_may_be_missing(tmp_path: Path) -> None:
    create_release_payload(tmp_path)
    optional = [
        "data/processed/defect_cards.jsonl",
        "data/processed/parameter_cards.jsonl",
    ]
    for relative in optional:
        (tmp_path / relative).unlink()
    chunk_count, paper_count = inspect_chunks(
        tmp_path / "data/chunks/full_chunks.jsonl"
    )
    chroma = inspect_chroma(
        tmp_path / "vector_store/chroma_full", "injection_papers_full"
    )
    release_files = collect_release_files(tmp_path)
    manifest = build_manifest(
        release_files,
        chunk_count=chunk_count,
        paper_count=paper_count,
        chroma=chroma,
        source_commit=None,
        optional_missing=optional,
    )
    write_manifest_files(manifest, tmp_path / "release_manifest")

    stats = verify_release(tmp_path, "injection_papers_full")

    assert stats["optional_missing"] == ",".join(optional)


def test_split_archive_uses_numbered_zip_parts(tmp_path: Path) -> None:
    archive = tmp_path / f"{ARTIFACT_NAME}.zip"
    archive.write_bytes(b"abcdefghij")
    parts = split_archive(archive, max_part_size=4)
    assert [part.name for part in parts] == [
        f"{ARTIFACT_NAME}.zip.001",
        f"{ARTIFACT_NAME}.zip.002",
        f"{ARTIFACT_NAME}.zip.003",
    ]
    assert b"".join(part.read_bytes() for part in parts) == b"abcdefghij"


def test_cli_output_alias_takes_precedence() -> None:
    legacy = parse_args(["--output_dir", "legacy/full_release_no_pdf_v1"])
    args = parse_args(
        [
            "--output_dir",
            "old/full_release_no_pdf_v1",
            "--output",
            "new/full_release_no_pdf_v1",
            "--split_size_mb",
            "42",
        ]
    )

    assert legacy.output_dir == Path("legacy/full_release_no_pdf_v1")
    assert legacy.split_size_mb == 1500
    assert args.output_dir == Path("new/full_release_no_pdf_v1")
    assert args.split_size_mb == 42


def test_manifest_records_numbered_archive_parts(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"abcdefghij")
    release_files = [
        type("ReleaseFileStub", (), {
            "source": source,
            "archive_path": "data/payload.bin",
            "size_bytes": source.stat().st_size,
            "sha256": "unused",
        })()
    ]
    manifest = {"files": []}
    archive_path = tmp_path / f"{ARTIFACT_NAME}.zip"

    archive, _, _, archive_info = build_archive_with_manifest(
        release_files,
        manifest,
        tmp_path / "manifest",
        archive_path,
        split_size_mb=0.0001,
        max_part_size=100,
    )
    parts = split_archive(archive, max_part_size=100)

    assert archive_info["split"] is True
    assert archive_info["assets"] == [part.name for part in parts]
    assert manifest["archive"] == archive_info


def test_release_asset_selection_requires_consecutive_parts() -> None:
    assets = [
        {"name": f"{ARTIFACT_NAME}.zip.002"},
        {"name": f"{ARTIFACT_NAME}.zip.001"},
    ]
    selected = select_release_assets(assets)
    assert [item["name"] for item in selected] == [
        f"{ARTIFACT_NAME}.zip.001",
        f"{ARTIFACT_NAME}.zip.002",
    ]
    with pytest.raises(ReleaseDownloadError, match="not consecutive"):
        select_release_assets(
            [
                {"name": f"{ARTIFACT_NAME}.zip.001"},
                {"name": f"{ARTIFACT_NAME}.zip.003"},
            ]
        )


def test_configured_assets_preserve_yaml_order_and_single_zip_compatibility() -> None:
    release_assets = [
        {"name": f"{ARTIFACT_NAME}_part002.zip"},
        {"name": f"{ARTIFACT_NAME}.zip"},
        {"name": f"{ARTIFACT_NAME}_part001.zip"},
    ]

    parts = select_configured_release_assets(
        release_assets,
        [f"{ARTIFACT_NAME}_part001.zip", f"{ARTIFACT_NAME}_part002.zip"],
    )
    single = select_configured_release_assets(
        release_assets,
        [f"{ARTIFACT_NAME}.zip"],
    )

    assert [item["name"] for item in parts] == [
        f"{ARTIFACT_NAME}_part001.zip",
        f"{ARTIFACT_NAME}_part002.zip",
    ]
    assert [item["name"] for item in single] == [f"{ARTIFACT_NAME}.zip"]


def test_install_release_merges_configured_parts_and_keeps_zip(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    create_release_payload(source_root)
    chunk_count, paper_count = inspect_chunks(
        source_root / "data/chunks/full_chunks.jsonl"
    )
    chroma = inspect_chroma(
        source_root / "vector_store/chroma_full", "injection_papers_full"
    )
    release_files = collect_release_files(source_root)
    manifest = build_manifest(
        release_files,
        chunk_count=chunk_count,
        paper_count=paper_count,
        chroma=chroma,
        source_commit=None,
    )
    manifest_path, sums_path = write_manifest_files(
        manifest, tmp_path / "manifest"
    )
    archive = build_archive(
        release_files,
        manifest_path,
        sums_path,
        tmp_path / f"{ARTIFACT_NAME}.zip",
    )
    archive_bytes = archive.read_bytes()
    midpoint = len(archive_bytes) // 2
    names = [
        f"{ARTIFACT_NAME}_part001.zip",
        f"{ARTIFACT_NAME}_part002.zip",
    ]
    payloads = {
        names[0]: archive_bytes[:midpoint],
        names[1]: archive_bytes[midpoint:],
    }
    available = [
        {"name": names[1], "browser_download_url": "test://part2"},
        {"name": names[0], "browser_download_url": "test://part1"},
    ]

    monkeypatch.setattr(
        "scripts.download_full_release.list_release_assets",
        lambda owner, repo, tag: available,
    )

    def fake_download(asset, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = payloads[asset["name"]]
        destination.write_bytes(content)
        return len(content)

    monkeypatch.setattr("scripts.download_full_release.download_asset", fake_download)
    artifact_dir = tmp_path / "artifacts" / ARTIFACT_NAME
    config = {
        "github_owner": "owner",
        "github_repo": "repo",
        "release_tag": "tag",
        "collection_name": "injection_papers_full",
        "archive_name": f"{ARTIFACT_NAME}.zip",
        "assets": names,
    }

    stats = install_release(config, artifact_dir)

    assert (artifact_dir.parent / f"{ARTIFACT_NAME}.zip").read_bytes() == archive_bytes
    assert (artifact_dir / "data/chunks/full_chunks.jsonl").is_file()
    assert (artifact_dir / "vector_store/chroma_full").is_dir()
    assert stats["release_assets"] == 2
    assert stats["asset_names"] == ",".join(names)
