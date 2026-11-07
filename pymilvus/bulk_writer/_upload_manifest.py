"""Request-local upload planning with bounded memory and an automatic SQLite spill."""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .upload_policy import UploadPolicy

logger = logging.getLogger(__name__)


@dataclass
class _UploadEntry:
    path: str
    size: int
    mtime_ns: int
    matched: bool = False


class _UploadManifest:
    FILE_LIMIT = 10_000
    DIRECTORY_LIMIT = 10_000
    MEMORY_LIMIT = 8 * 1024 * 1024

    def __init__(self, temporary_directory: str | None = None):
        self._temporary_directory = temporary_directory
        self._scratch = None
        self._db = None
        self._files = {}
        self._directories = deque()
        self._memory_bytes = 0
        self.file_count = self.total_bytes = 0
        self.remaining_count = self.remaining_bytes = self.unmatched_count = 0
        self._last_log = 0.0
        self._scan_started_at = 0.0
        self._scan_root = None
        self._scanned_directories = 0
        self._operations = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        try:
            if self._db is not None:
                self._db.close()
        finally:
            if self._scratch is not None:
                self._scratch.cleanup()

    @property
    def storage(self) -> str:
        return "sqlite" if self._db is not None else "memory"

    def _spill(self, reason: str) -> None:
        self._scratch = tempfile.TemporaryDirectory(
            prefix="milvus-volume-", dir=self._temporary_directory
        )
        self._db = sqlite3.connect(str(Path(self._scratch.name) / "manifest.db"))
        self._db.execute("PRAGMA cache_size=-8192")
        self._db.execute("PRAGMA mmap_size=0")
        self._db.execute("PRAGMA temp_store=FILE")
        self._db.execute(
            "CREATE TABLE files (path TEXT PRIMARY KEY, size INTEGER, "
            "mtime INTEGER, matched INTEGER) WITHOUT ROWID"
        )
        self._db.execute("CREATE TABLE directories (id INTEGER PRIMARY KEY, path TEXT)")
        self._db.executemany(
            "INSERT INTO files VALUES (?, ?, ?, ?)",
            ((e.path, e.size, e.mtime_ns, e.matched) for e in self._files.values()),
        )
        self._db.executemany(
            "INSERT INTO directories(path) VALUES (?)", ((p,) for p in self._directories)
        )
        self._db.commit()
        self._files = self._directories = None
        logger.info(
            "Volume upload index switched to SQLite: reason:%s, scannedFiles:%s, estimatedMemoryBytes:%s",
            reason,
            self.file_count,
            self._memory_bytes,
        )
        self._memory_bytes = 0

    def _reserve(self, cost: int, count_limit_reason: str | None) -> None:
        if self._db is None:
            if count_limit_reason:
                self._spill(count_limit_reason)
            elif self._memory_bytes + cost > self.MEMORY_LIMIT:
                self._spill(f"estimated index memory exceeds {self.MEMORY_LIMIT}")
        if self._db is None:
            self._memory_bytes += cost

    def _commit_batch(self) -> None:
        self._operations += 1
        if self._db is not None and self._operations % 1000 == 0:
            self._db.commit()

    def add(self, path: str, size: int, mtime_ns: int) -> None:
        self._reserve(
            256 + 4 * len(path),
            f"file count exceeds {self.FILE_LIMIT}" if self.file_count >= self.FILE_LIMIT else None,
        )
        if self._db is None:
            if path in self._files:
                msg = f"Duplicate upload path: {path}"
                raise ValueError(msg)
            self._files[path] = _UploadEntry(path, size, mtime_ns)
        else:
            self._db.execute("INSERT INTO files VALUES (?, ?, ?, 0)", (path, size, mtime_ns))
        self.file_count += 1
        self.total_bytes += size
        self.remaining_count += 1
        self.remaining_bytes += size
        self.unmatched_count += 1
        self._commit_batch()

    def _enqueue(self, path: str) -> None:
        self._reserve(
            128 + 4 * len(path),
            (
                f"pending directory count exceeds {self.DIRECTORY_LIMIT}"
                if self._directories is not None and len(self._directories) >= self.DIRECTORY_LIMIT
                else None
            ),
        )
        if self._db is None:
            self._directories.append(path)
        else:
            self._db.execute("INSERT INTO directories(path) VALUES (?)", (path,))
        self._commit_batch()

    def _poll(self) -> str | None:
        if self._db is None:
            if not self._directories:
                return None
            path = self._directories.popleft()
            self._memory_bytes -= 128 + 4 * len(path)
            return path
        row = self._db.execute("SELECT id, path FROM directories ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return None
        self._db.execute("DELETE FROM directories WHERE id=?", (row[0],))
        return row[1]

    def scan(self, root: Path) -> bool:
        self._scan_root = root
        self._scan_started_at = self._last_log = time.monotonic()
        logger.info("Volume local scan started: sourcePath:%s", root)
        attrs = root.stat()
        if stat.S_ISREG(attrs.st_mode):
            self.add(root.name, attrs.st_size, attrs.st_mtime_ns)
            self._log_scan(True)
            return True
        if not stat.S_ISDIR(attrs.st_mode):
            msg = f"Upload source must be a regular file or directory: {root}"
            raise ValueError(msg)
        self._enqueue("")
        while (relative := self._poll()) is not None:
            current = root / relative
            self._check_cycle(root, current)
            with os.scandir(current) as children:
                for child in children:
                    path = Path(child.path)
                    if (
                        self._scratch is not None
                        and path.name == Path(self._scratch.name).name
                        and path.samefile(self._scratch.name)
                    ):
                        continue
                    attrs = child.stat()
                    relative = path.relative_to(root).as_posix()
                    if stat.S_ISREG(attrs.st_mode):
                        self.add(relative, attrs.st_size, attrs.st_mtime_ns)
                    elif stat.S_ISDIR(attrs.st_mode):
                        self._enqueue(relative)
                    else:
                        msg = f"Special files are not supported: {path}"
                        raise ValueError(msg)
                    self._log_scan(False)
            self._scanned_directories += 1
            self._commit_batch()
            self._log_scan(False)
        if self._db is not None:
            self._db.commit()
        self._log_scan(True)
        return False

    @staticmethod
    def _check_cycle(root: Path, current: Path) -> None:
        if current == root or not current.is_symlink():
            return
        target = current.resolve(strict=True)
        ancestor = current.parent
        while True:
            if ancestor.resolve(strict=True).is_relative_to(target) or ancestor.samefile(current):
                msg = f"Directory cycle in upload source: {current}"
                raise ValueError(msg)
            if ancestor == root:
                return
            ancestor = ancestor.parent

    def _log_scan(self, completed: bool) -> None:
        now = time.monotonic()
        if completed or now - self._last_log >= 5:
            logger.info(
                "Volume local scan %s: sourcePath:%s, scannedDirectories:%s, files:%s, "
                "bytes:%s, elapsedMillis:%s, indexStorage:%s",
                "completed" if completed else "progress",
                self._scan_root,
                self._scanned_directories,
                self.file_count,
                self.total_bytes,
                int((now - self._scan_started_at) * 1000),
                self.storage,
            )
            self._last_log = now

    def match(
        self, path: str, size: int | None, mtime_ns: int | None, policy: UploadPolicy
    ) -> None:
        if self._db is None:
            entry = self._files.get(path)
        else:
            row = self._db.execute(
                "SELECT size, mtime, matched FROM files WHERE path=?", (path,)
            ).fetchone()
            entry = _UploadEntry(path, *row) if row is not None else None
        if entry is None:
            return
        if not entry.matched:
            self.unmatched_count -= 1
        if policy.should_skip(entry.size, entry.mtime_ns, size, mtime_ns):
            if self._db is None:
                del self._files[path]
                self._memory_bytes -= 256 + 4 * len(path)
            else:
                self._db.execute("DELETE FROM files WHERE path=?", (path,))
            self.remaining_count -= 1
            self.remaining_bytes -= entry.size
        elif self._db is None:
            entry.matched = True
        else:
            self._db.execute("UPDATE files SET matched=1 WHERE path=?", (path,))
        self._commit_batch()

    def entries(self) -> Iterator[_UploadEntry]:
        if self._db is None:
            yield from self._files.values()
        else:
            self._db.commit()
            cursor = self._db.execute("SELECT path, size, mtime FROM files ORDER BY path")
            try:
                for row in cursor:
                    yield _UploadEntry(*row)
            finally:
                cursor.close()
