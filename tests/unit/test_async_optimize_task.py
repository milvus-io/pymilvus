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
    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [None, 10])
    async def test_cancelled_result_waiter_preserves_background_task(self, timeout):
        started = asyncio.Event()
        finish = asyncio.Event()
        expected = MagicMock(collection_name="col")

        async def execute(**kwargs):
            started.set()
            await finish.wait()
            return expected

        task = _make_task(execute)
        task.start()
        await started.wait()
        waiter = asyncio.create_task(task.result(timeout=timeout))
        other_waiter = asyncio.create_task(task.result())
        await asyncio.sleep(0)
        try:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not task.done()
            assert not task.cancelled()
            assert not other_waiter.done()
            finish.set()
            assert await other_waiter is expected
        finally:
            finish.set()
            await asyncio.gather(task._task, waiter, other_waiter, return_exceptions=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [None, 10])
    async def test_external_wait_for_preserves_background_task(self, timeout):
        started = asyncio.Event()
        finish = asyncio.Event()
        expected = MagicMock(collection_name="col")

        async def execute(**kwargs):
            started.set()
            await finish.wait()
            return expected

        task = _make_task(execute)
        task.start()
        await started.wait()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(task.result(timeout=timeout), timeout=0.001)
            assert not task.done()
            assert not task.cancelled()
            finish.set()
            assert await task.result() is expected
        finally:
            finish.set()
            await asyncio.gather(task._task, return_exceptions=True)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not hasattr(asyncio, "timeout"), reason="Requires Python 3.11+")
    async def test_external_timeout_context_preserves_background_task(self):
        started = asyncio.Event()
        finish = asyncio.Event()
        expected = MagicMock(collection_name="col")

        async def execute(**kwargs):
            started.set()
            await finish.wait()
            return expected

        task = _make_task(execute)
        task.start()
        await started.wait()
        try:
            with pytest.raises(asyncio.TimeoutError):
                async with asyncio.timeout(0.001):
                    await task.result()
            assert not task.done()
            assert not task.cancelled()
            finish.set()
            assert await task.result() is expected
        finally:
            finish.set()
            await asyncio.gather(task._task, return_exceptions=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [0, 0.001])
    @pytest.mark.parametrize("fails", [False, True])
    async def test_wait_timeout_preserves_background_result(self, timeout, fails):
        started = asyncio.Event()
        finish = asyncio.Event()
        expected = MagicMock(collection_name="col")

        async def execute(**kwargs):
            started.set()
            await finish.wait()
            if fails:
                raise MilvusException(message="Index rebuild failed")
            return expected

        task = _make_task(execute)
        task.start()
        await started.wait()
        try:
            with pytest.raises(MilvusException, match="Timeout waiting"):
                await task.result(timeout=timeout)
            assert not task.done()
            assert not task.cancelled()
            assert ProgressStage.CANCELLED not in task.progress_history()
            finish.set()
            if fails:
                with pytest.raises(MilvusException, match="Index rebuild failed"):
                    await task.result()
            else:
                assert await task.result() is expected
        finally:
            finish.set()
            await asyncio.gather(task._task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_explicit_cancel_interrupts_timed_wait(self):
        started = asyncio.Event()

        async def execute(**kwargs):
            started.set()
            await asyncio.Event().wait()

        task = _make_task(execute)
        task.start()
        await started.wait()
        waiter = asyncio.create_task(task.result(timeout=10))
        await asyncio.sleep(0)
        assert task.cancel()
        with pytest.raises(MilvusException, match="cancelled"):
            await waiter
        assert task.done()
        assert task.cancelled()

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
