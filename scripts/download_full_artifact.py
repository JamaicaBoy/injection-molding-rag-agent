from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.full_artifact_common import (  # noqa: E402
    ArtifactError,
    MANIFEST_NAME,
    read_json,
    safe_destination,
    safe_member_name,
    sha256_file,
)
from scripts.verify_full_artifact import verify_archive, verify_extracted  # noqa: E402


DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "artifacts" / "downloads"


def download_file(url: str, destination: Path, expected_sha256: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing_bytes = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "injection-molding-rag-artifact-downloader/1.0"}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = existing_bytes > 0 and getattr(response, "status", 200) == 206
            mode = "ab" if append else "wb"
            if not append:
                existing_bytes = 0
            with partial.open(mode) as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except OSError as exc:
        raise ArtifactError(f"Download failed: {url}") from exc
    partial.replace(destination)
    if expected_sha256 and sha256_file(destination) != expected_sha256:
        raise ArtifactError(f"Downloaded SHA-256 mismatch: {destination.name}")
    return destination


def remote_json(url: str, download_dir: Path) -> tuple[dict[str, Any], Path]:
    name = Path(urllib.parse.urlparse(url).path).name or "artifact.parts.json"
    local_path = download_file(url, download_dir / name)
    return read_json(local_path), local_path


def download_split_artifact(descriptor_url: str, download_dir: Path) -> Path:
    descriptor, descriptor_path = remote_json(descriptor_url, download_dir)
    parts = descriptor.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ArtifactError("Remote parts descriptor does not contain parts.")
    base_url = descriptor_url.rsplit("/", 1)[0] + "/"
    local_parts: list[Path] = []
    for record in parts:
        name = safe_member_name(str(record.get("name", "")))
        if "/" in name:
            raise ArtifactError(f"Part name must not contain directories: {name}")
        part_url = urllib.parse.urljoin(base_url, urllib.parse.quote(name))
        local_parts.append(download_file(part_url, descriptor_path.parent / name, str(record.get("sha256", ""))))

    archive_name = safe_member_name(str(descriptor.get("archive_name", "")))
    if "/" in archive_name:
        raise ArtifactError("Archive name in parts descriptor must be a file name.")
    archive_path = download_dir / archive_name
    temporary_path = archive_path.with_name(archive_path.name + ".joining")
    with temporary_path.open("wb") as output:
        for part_path in local_parts:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output, length=8 * 1024 * 1024)
    temporary_path.replace(archive_path)
    if archive_path.stat().st_size != int(descriptor.get("archive_size_bytes", -1)):
        raise ArtifactError("Reassembled archive size does not match the parts descriptor.")
    if sha256_file(archive_path) != str(descriptor.get("archive_sha256", "")):
        raise ArtifactError("Reassembled archive SHA-256 does not match the parts descriptor.")
    return archive_path


def download_single_artifact(url: str, download_dir: Path, expected_sha256: str | None) -> Path:
    name = Path(urllib.parse.urlparse(url).path).name
    if not name:
        raise ArtifactError("Cannot determine the archive file name from --url.")
    return download_file(url, download_dir / name, expected_sha256)


def extract_archive(archive_path: Path, destination: Path, overwrite: bool = False) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
        infos = archive.infolist()
        names = [safe_member_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise ArtifactError("Archive contains duplicate member paths.")
        for info, name in zip(infos, names):
            target = safe_destination(destination, name)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not overwrite:
                raise ArtifactError(f"Extraction target already exists; use --overwrite to replace it: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, verify, and safely extract the public full artifact.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Direct URL to one complete ZIP artifact.")
    source.add_argument("--parts_manifest_url", help="URL to a .parts.json descriptor and sibling part files.")
    parser.add_argument("--download_dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--extract_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 for a direct --url download.")
    parser.add_argument("--expected_pdf_count", type=int, default=896)
    parser.add_argument("--no_extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_dir = args.download_dir.resolve()
    archive_path = (
        download_split_artifact(args.parts_manifest_url, download_dir)
        if args.parts_manifest_url
        else download_single_artifact(args.url, download_dir, args.sha256)
    )
    archive_stats = verify_archive(archive_path, expected_pdf_count=args.expected_pdf_count)
    print("Full artifact download verified.")
    print(f"archive: {archive_path}")
    print(f"raw_pdf_count: {archive_stats['raw_pdf_count']}")
    print(f"collection_count: {archive_stats['collection_count']}")
    if args.no_extract:
        return 0

    extract_archive(archive_path, args.extract_dir, overwrite=args.overwrite)
    extracted_stats = verify_extracted(args.extract_dir, expected_pdf_count=args.expected_pdf_count)
    print("Full artifact extraction verified.")
    print(f"extract_dir: {args.extract_dir.resolve()}")
    print(f"files: {extracted_stats['file_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArtifactError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
