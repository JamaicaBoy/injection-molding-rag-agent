"""Download, assemble, extract, and verify full_release_no_pdf_v1."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_full_release import verify_release  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "artifact_config.yaml"
ARTIFACT_NAME = "full_release_no_pdf_v1"
READ_BLOCK_SIZE = 8 * 1024 * 1024
PART_PATTERN = re.compile(rf"^{re.escape(ARTIFACT_NAME)}\.zip\.(\d{{3}})$")


class ReleaseDownloadError(RuntimeError):
    pass


def load_artifact_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = (
        "artifact_name",
        "artifact_source",
        "github_owner",
        "github_repo",
        "release_tag",
        "artifact_dir",
        "collection_name",
    )
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ReleaseDownloadError(f"artifact_config.yaml missing fields: {missing}")
    if config["artifact_name"] != ARTIFACT_NAME or config["artifact_source"] != "github_release":
        raise ReleaseDownloadError("artifact_config.yaml does not describe full_release_no_pdf_v1")
    if str(config["github_owner"]).upper() == "YOUR_GITHUB_USERNAME":
        raise ReleaseDownloadError("replace github_owner placeholder in configs/artifact_config.yaml")
    return config


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "injection-molding-rag-full-release-downloader/1.0",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def list_release_assets(owner: str, repo: str, tag: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    request = urllib.request.Request(url, headers=github_headers())
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise ReleaseDownloadError(f"cannot read GitHub Release {tag}: {exc}") from exc
    return list(payload.get("assets", []))


def select_release_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    single_name = f"{ARTIFACT_NAME}.zip"
    singles = [asset for asset in assets if asset.get("name") == single_name]
    parts = [asset for asset in assets if PART_PATTERN.match(str(asset.get("name", "")))]
    if singles and parts:
        raise ReleaseDownloadError("Release contains both a full ZIP and numbered parts")
    if singles:
        return singles
    if not parts:
        raise ReleaseDownloadError("Release has no full_release_no_pdf_v1 ZIP assets")
    parts.sort(key=lambda asset: int(PART_PATTERN.match(str(asset["name"])).group(1)))
    numbers = [int(PART_PATTERN.match(str(asset["name"])).group(1)) for asset in parts]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ReleaseDownloadError(f"Release parts are not consecutive: {numbers}")
    return parts


def download_asset(asset: dict[str, Any], destination: Path) -> int:
    url = str(asset.get("browser_download_url", ""))
    if not url:
        raise ReleaseDownloadError(f"asset has no download URL: {asset.get('name')}")
    request = urllib.request.Request(url, headers=github_headers())
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
            while True:
                block = response.read(READ_BLOCK_SIZE)
                if not block:
                    break
                stream.write(block)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ReleaseDownloadError(f"download failed for {asset.get('name')}: {exc}") from exc
    partial.replace(destination)
    expected_size = int(asset.get("size", 0) or 0)
    if expected_size and destination.stat().st_size != expected_size:
        raise ReleaseDownloadError(f"downloaded size mismatch: {asset.get('name')}")
    return destination.stat().st_size


def merge_assets(parts: list[Path], output_zip: Path) -> Path:
    output_zip.unlink(missing_ok=True)
    if len(parts) == 1 and parts[0].name == output_zip.name:
        shutil.copy2(parts[0], output_zip)
        return output_zip
    with output_zip.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, READ_BLOCK_SIZE)
    return output_zip


def safe_extract(archive_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ReleaseDownloadError(f"unsafe ZIP member: {info.filename}")
            unix_mode = info.external_attr >> 16
            if unix_mode & 0o170000 == 0o120000:
                raise ReleaseDownloadError(f"ZIP symlink is not allowed: {info.filename}")
            target = destination / Path(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, READ_BLOCK_SIZE)
            count += 1
    return count


def install_release(config: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    assets = select_release_assets(
        list_release_assets(
            str(config["github_owner"]),
            str(config["github_repo"]),
            str(config["release_tag"]),
        )
    )
    artifact_dir = artifact_dir.resolve()
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="full-release-", dir=artifact_dir.parent) as temporary:
        temporary_root = Path(temporary)
        downloads = temporary_root / "downloads"
        local_parts: list[Path] = []
        downloaded_bytes = 0
        for asset in assets:
            destination = downloads / str(asset["name"])
            downloaded_bytes += download_asset(asset, destination)
            local_parts.append(destination)
        merged = merge_assets(local_parts, temporary_root / f"{ARTIFACT_NAME}.zip")
        extracted = temporary_root / "extracted"
        extracted_files = safe_extract(merged, extracted)
        stats = verify_release(extracted, str(config["collection_name"]))
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        shutil.move(str(extracted), str(artifact_dir))
    return {
        **stats,
        "release_assets": len(assets),
        "downloaded_bytes": downloaded_bytes,
        "extracted_files": extracted_files,
        "artifact_dir": str(artifact_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_artifact_config()
    artifact_dir = args.artifact_dir or PROJECT_ROOT / str(config["artifact_dir"])
    stats = install_release(config, artifact_dir)
    print("download_and_verify: PASS")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReleaseDownloadError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"download_and_verify: FAIL ({exc})", file=sys.stderr)
        raise SystemExit(1) from exc
