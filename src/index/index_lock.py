from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK_DIR = PROJECT_ROOT / "data" / "runtime" / "index_locks"


class IndexLockError(RuntimeError):
    pass


def collection_lock_path(
    persist_dir: Path,
    collection_name: str,
    *,
    lock_dir: Path = DEFAULT_LOCK_DIR,
) -> Path:
    identity = f"{Path(persist_dir).resolve()}::{collection_name}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    safe_collection = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in collection_name
    )[:60]
    return Path(lock_dir) / f"{safe_collection}_{digest}.lock"


@dataclass
class IndexFileLock:
    persist_dir: Path
    collection_name: str
    lock_dir: Path = DEFAULT_LOCK_DIR
    timeout: float = 0.0
    poll_interval: float = 0.1
    stale_after: float | None = 6 * 60 * 60

    def __post_init__(self) -> None:
        self.persist_dir = Path(self.persist_dir)
        self.lock_dir = Path(self.lock_dir)
        self.path = collection_lock_path(
            self.persist_dir,
            self.collection_name,
            lock_dir=self.lock_dir,
        )
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _payload(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "pid": os.getpid(),
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }

    def _remove_stale_lock(self) -> bool:
        if self.stale_after is None or not self.path.exists():
            return False
        try:
            age = time.time() - self.path.stat().st_mtime
            if age <= self.stale_after:
                return False
            self.path.unlink()
            return True
        except OSError:
            return False

    def acquire(self) -> "IndexFileLock":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(self.timeout, 0.0)
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                    json.dump(self._payload(), file, ensure_ascii=False, indent=2)
                self.acquired = True
                return self
            except FileExistsError as exc:
                if self._remove_stale_lock():
                    continue
                if time.monotonic() >= deadline:
                    raise IndexLockError(
                        f"索引正在更新，请稍后: {self.collection_name}"
                    ) from exc
                time.sleep(max(self.poll_interval, 0.01))

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        finally:
            self.acquired = False

    def __enter__(self) -> "IndexFileLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def index_write_lock(
    persist_dir: Path,
    collection_name: str,
    *,
    lock_dir: Path = DEFAULT_LOCK_DIR,
    timeout: float = 0.0,
    stale_after: float | None = 6 * 60 * 60,
) -> IndexFileLock:
    return IndexFileLock(
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        lock_dir=Path(lock_dir),
        timeout=timeout,
        stale_after=stale_after,
    )
