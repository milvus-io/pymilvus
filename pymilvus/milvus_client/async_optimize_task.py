import asyncio
from typing import Callable, List, Optional

from pymilvus.exceptions import MilvusException

from .optimize_task import OptimizeResult, ProgressStage, parse_target_size


class AsyncOptimizeTask:
    def __init__(
        self,
        collection_name: str,
        target_size: Optional[str],
        task_timeout: Optional[float],
        execute_fn: Callable,
        **kwargs,
    ) -> None:
        self._collection_name = collection_name
        self._target_size = target_size
        self._task_timeout = task_timeout
        self._execute_fn = execute_fn
        self._kwargs = kwargs

        self._task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._progress_stage = ProgressStage.INITIALIZING
        self._progress_history = [ProgressStage.INITIALIZING]

    def start(self) -> None:
        async def _run() -> OptimizeResult:
            try:
                result = await self._execute_fn(
                    task=self,
                    collection_name=self._collection_name,
                    size_mb=parse_target_size(self._target_size),
                    timeout=self._task_timeout,
                    **self._kwargs,
                )
            except asyncio.CancelledError as e:
                self._progress_stage = ProgressStage.CANCELLED
                self._progress_history.append(ProgressStage.CANCELLED)
                raise MilvusException(message="Optimization task was cancelled") from e
            else:
                return result

        self._task = asyncio.create_task(_run())

    def done(self) -> bool:
        return self._task is not None and self._task.done()

    def cancelled(self) -> bool:
        return self._cancelled

    def progress(self) -> str:
        return self._progress_stage

    def progress_history(self) -> List[str]:
        return list(self._progress_history)

    def cancel(self) -> bool:
        if self._cancelled:
            return True
        if self.done():
            return False
        self._cancelled = True
        self._progress_stage = ProgressStage.CANCELLED
        self._progress_history.append(ProgressStage.CANCELLED)
        if self._task:
            self._task.cancel()
        return True

    def check_cancelled(self) -> None:
        if self._cancelled:
            raise MilvusException(message="Optimization task was cancelled")

    async def result(self, timeout: Optional[float] = None) -> OptimizeResult:
        if not self._task:
            raise MilvusException(message="Task has not been started")

        try:
            if timeout is not None:
                return await asyncio.wait_for(self._task, timeout=timeout)
            return await self._task
        except asyncio.TimeoutError as e:
            raise MilvusException(message="Timeout waiting for optimization to complete") from e
        except asyncio.CancelledError as e:
            raise MilvusException(message="Optimization task was cancelled") from e

    def set_progress(self, stage: ProgressStage) -> None:
        if self._cancelled:
            return
        self._progress_stage = stage
        self._progress_history.append(stage)
