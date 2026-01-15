import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Union

from pymilvus.exceptions import MilvusException, ParamError

unit_to_bytes = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
    "pb": 1024**5,
}


def parse_target_size(target_size: Union[str, float, int, None]) -> int:
    """Parse target size string or number and return size in MB.

    Args:
        target_size: Size as string with optional unit (B/KB/MB/GB/TB/PB) or number in bytes.
                    If unit is omitted, treats value as bytes.

    Returns:
        Size in megabytes (MB).

    Raises:
        ParamError: If format is invalid or unit is not supported,
        or size is less than or equal to zero

    Examples:
        >>> parse_target_size("1GB")
        1024
        >>> parse_target_size("500MB")
        500
        >>> parse_target_size(1048576)
        1
    """
    if target_size is None:
        return 1 << 63 - 1

    if not isinstance(target_size, (str, int, float)):
        raise ParamError(
            message=f"target_size must be a string or number, got {type(target_size).__name__}"
        )

    size_bytes = 0
    if isinstance(target_size, (int, float)):
        size_bytes = target_size
    else:
        target_str = str(target_size).strip().lower()
        pattern = r"^(\d+(?:\.\d+)?)\s*([a-z]*)$"
        match = re.match(pattern, target_str)

        if not match:
            raise ParamError(
                message=f"Invalid target_size format: '{target_size}'. "
                f"Expected format: '1000', '1000MB', '1GB', '1.2gb', '1000B', '500KB', '2TB'"
            )

        value = float(match.group(1))
        unit = match.group(2) or "b"

        if unit not in unit_to_bytes:
            raise ParamError(
                message=f"Invalid unit: '{unit}'. Supported units: B, KB, MB, GB, TB, PB"
            )

        size_bytes = value * unit_to_bytes[unit]

    size_mb = int(size_bytes / (1024**2))
    if size_mb <= 0:
        raise ParamError(message=f"target size too small: {target_size}, must be at least 1MB")
    return size_mb


class ProgressStage(str, Enum):
    INITIALIZING = "initializing"
    WAITING_FOR_INDEXES = "waiting for indexes"
    COMPACTING = "compacting"
    WAITING_FOR_COMPACTION = "waiting for compaction"
    WAITING_FOR_INDEX_REBUILD = "waiting for index rebuild"
    REFRESHING_LOAD = "refreshing load"
    CANCELLED = "cancelled"


@dataclass
class OptimizeResult:
    """Result of an optimization operation.

    Attributes:
        status: Status of the optimization ("success").
        collection_name: Name of the optimized collection.
        compaction_id: ID of the compaction job.
        target_size: Target segment size that was requested.
        progress: List of progress stages completed during optimization.
    """

    status: str
    collection_name: str
    compaction_id: int
    target_size: Optional[str]
    progress: List[str]


class OptimizeTask(threading.Thread):
    def __init__(
        self,
        collection_name: str,
        target_size: Optional[str],
        task_timeout: Optional[float],
        execute_fn: Callable,
        **kwargs,
    ) -> None:
        super().__init__(daemon=True)
        self._collection_name = collection_name
        self._target_size = target_size
        self._task_timeout = task_timeout
        self._execute_fn = execute_fn
        self._kwargs = kwargs

        self._condition = threading.Condition()
        self._done = False
        self._cancelled = False
        self._result: Optional[OptimizeResult] = None
        self._exception: Optional[Exception] = None
        self._progress_stage = ProgressStage.INITIALIZING
        self._progress_history = [ProgressStage.INITIALIZING]

    def run(self) -> None:
        """Execute the optimization task in the background thread."""
        try:
            result = self._execute_fn(
                task=self,
                collection_name=self._collection_name,
                size_mb=parse_target_size(self._target_size),
                timeout=self._task_timeout,
                **self._kwargs,
            )
            if not self._cancelled:
                self._set_result(result)
        except Exception as e:
            if not self._cancelled:
                self._set_exception(e)

    def done(self) -> bool:
        with self._condition:
            return self._done

    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    def progress(self) -> str:
        with self._condition:
            return self._progress_stage

    def progress_history(self) -> List[str]:
        with self._condition:
            return list(self._progress_history)

    def cancel(self) -> bool:
        """Cancel the optimization task.

        Returns:
            True if the task was successfully cancelled, False if it was already done.
        """
        with self._condition:
            if self._cancelled:
                return True
            if self._done:
                return False
            self._cancelled = True
            self._progress_stage = ProgressStage.CANCELLED
            self._progress_history.append(ProgressStage.CANCELLED)
            self._exception = MilvusException(message="Optimization task was cancelled")
            self._done = True
            self._condition.notify_all()
            return True

    def check_cancelled(self) -> None:
        """Check if task is cancelled and raise exception if so."""
        with self._condition:
            if self._cancelled:
                raise MilvusException(message="Optimization task was cancelled")

    def result(self, timeout: Optional[float] = None) -> OptimizeResult:
        with self._condition:
            if timeout is not None:
                self._condition.wait_for(lambda: self._done, timeout=timeout)
            else:
                self._condition.wait_for(lambda: self._done)

            if self._exception:
                raise self._exception

            return self._result

    def set_progress(self, stage: ProgressStage) -> None:
        """Update the current progress stage."""
        with self._condition:
            if self._cancelled:
                return
            self._progress_stage = stage
            self._progress_history.append(stage)

    def _set_result(self, result: OptimizeResult) -> None:
        with self._condition:
            self._result = result
            self._done = True
            self._condition.notify_all()

    def _set_exception(self, exception: Exception) -> None:
        with self._condition:
            self._exception = exception
            self._done = True
            self._condition.notify_all()
