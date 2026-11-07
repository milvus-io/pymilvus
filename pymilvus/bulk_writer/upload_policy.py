from __future__ import annotations

from enum import Enum


class UploadPolicy(str, Enum):
    """How LIST metadata is used to decide whether a target object can be skipped.

    Keys must match exactly. Size and server modification time are heuristics, not
    content checksums. SIZE_AND_MTIME skips when sizes match and the local file is
    no newer than the remote object, comparing epoch milliseconds (as in Java).
    LIST cannot recover the original source mtime. Local change detection still uses
    the original nanosecond timestamps.
    """

    OVERWRITE = "OVERWRITE"
    SKIP_IF_EXISTS = "SKIP_IF_EXISTS"
    SKIP_IF_SAME_SIZE = "SKIP_IF_SAME_SIZE"
    SIZE_AND_MTIME = "SIZE_AND_MTIME"

    def should_skip(
        self,
        local_size: int,
        local_mtime_ns: int,
        remote_size: int | None,
        remote_mtime_ns: int | None,
    ) -> bool:
        if self is UploadPolicy.OVERWRITE:
            return False
        if self is UploadPolicy.SKIP_IF_EXISTS:
            return True
        if remote_size is None or remote_size < 0 or local_size != remote_size:
            return False
        return self is UploadPolicy.SKIP_IF_SAME_SIZE or (
            remote_mtime_ns is not None
            and local_mtime_ns // 1_000_000 <= remote_mtime_ns // 1_000_000
        )
