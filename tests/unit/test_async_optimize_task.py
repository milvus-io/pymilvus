import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client.async_optimize_task import AsyncOptimizeTask
from pymilvus.milvus_client.optimize_task import ProgressStage


def _make_task(execute_fn=None):
    if execute_fn is None:
        execute_fn = AsyncMock(return_value=MagicMock(collection_name="col"))
    return AsyncOptimizeTask("col", None, None, execute_fn)


class TestAsyncOptimizeTaskCancel:
    def test_cancel_before_start(self):
        task = _make_task()
        result = task.cancel()
        assert result is True
        assert task.cancelled() is True
        assert task.progress() == ProgressStage.CANCELLED

    def test_cancel_already_cancelled(self):
        task = _make_task()
        task.cancel()
        result = task.cancel()
        assert result is True

    def test_cancel_done_returns_false(self):
        async def run():
            execute_fn = AsyncMock(return_value=MagicMock())
            task = _make_task(execute_fn)
            task.start()
            await asyncio.sleep(0)
            result = task.cancel()
            assert result is False

        asyncio.run(run())

    def test_cancel_running_task(self):
        async def run():
            execute_fn = AsyncMock(return_value=MagicMock())
            task = AsyncOptimizeTask("col", None, 10.0, execute_fn)
            task.start()
            result = task.cancel()
            assert result is True
            assert task.cancelled() is True

        asyncio.run(run())


class TestAsyncOptimizeTaskCheckCancelled:
    def test_check_cancelled_not_cancelled(self):
        task = _make_task()
        task.check_cancelled()

    def test_check_cancelled_when_cancelled_raises(self):
        task = _make_task()
        task.cancel()
        with pytest.raises(MilvusException, match="cancelled"):
            task.check_cancelled()


class TestAsyncOptimizeTaskResult:
    def test_result_without_start_raises(self):
        task = _make_task()

        async def run():
            with pytest.raises(MilvusException, match="not been started"):
                await task.result()

        asyncio.run(run())

    def test_result_success(self):
        mock_result = MagicMock(collection_name="col")
        execute_fn = AsyncMock(return_value=mock_result)

        async def run():
            task = _make_task(execute_fn)
            task.start()
            result = await task.result()
            assert result == mock_result

        asyncio.run(run())

    def test_result_timeout_raises(self):
        async def slow_execute(**kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        async def run():
            task = AsyncOptimizeTask("col", None, None, slow_execute)
            task.start()
            with pytest.raises(MilvusException):
                await task.result(timeout=0.01)

        asyncio.run(run())

    def test_result_cancelled_raises(self):
        async def run():
            execute_fn = AsyncMock(return_value=MagicMock())
            task = _make_task(execute_fn)
            task.start()
            task._task.cancel()
            with pytest.raises(MilvusException):
                await task.result()

        asyncio.run(run())


class TestAsyncOptimizeTaskSetProgress:
    def test_set_progress_updates_stage(self):
        task = _make_task()
        task.set_progress(ProgressStage.COMPACTING)
        assert task.progress() == ProgressStage.COMPACTING
        assert ProgressStage.COMPACTING in task.progress_history()

    def test_set_progress_noop_when_cancelled(self):
        task = _make_task()
        task.cancel()
        task.set_progress(ProgressStage.COMPACTING)
        assert task.progress() == ProgressStage.CANCELLED

    def test_progress_history_initial(self):
        task = _make_task()
        assert task.progress_history() == [ProgressStage.INITIALIZING]

    def test_done_before_start(self):
        task = _make_task()
        assert task.done() is False
