import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import AsyncMilvusClient
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.milvus_client.async_optimize_task import AsyncOptimizeTask
from pymilvus.milvus_client.optimize_task import (
    _OPTIMIZE_DEFAULT_SIZE_MB,
    OptimizeResult,
    OptimizeTask,
    ProgressStage,
    parse_target_size,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(collection_name="test", compaction_id=1, size_mb=100):
    return OptimizeResult(
        status="success",
        collection_name=collection_name,
        compaction_id=compaction_id,
        target_size=str(size_mb),
        progress=[],
    )


def _make_task(execute_fn=None, collection_name="col", target_size="1GB", task_timeout=None):
    if execute_fn is None:

        def execute_fn(**kw):
            pass

    return OptimizeTask(
        collection_name=collection_name,
        target_size=target_size,
        task_timeout=task_timeout,
        execute_fn=execute_fn,
    )


# ── Async optimize tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_optimize_wait_true_completes_successfully() -> None:
    optimize_result = OptimizeResult(
        status="success",
        collection_name="test_collection",
        compaction_id=1,
        target_size="1GB",
        progress=[ProgressStage.INITIALIZING],
    )

    with patch(
        "pymilvus.milvus_client.async_milvus_client.AsyncConnectionManager",
        return_value=MagicMock(),
    ), patch("pymilvus.orm.connections.Connections._fetch_handler") as mock_fetch:
        handler = MagicMock()
        handler.get_server_type.return_value = "milvus"
        mock_fetch.return_value = handler
        client = AsyncMilvusClient()
        client._execute_optimize = AsyncMock(return_value=optimize_result)

        response = await client.optimize("test_collection", target_size="1GB", wait=True)

    assert response == optimize_result
    client._execute_optimize.assert_awaited_once()
    call_args = client._execute_optimize.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    assert call_args.kwargs["size_mb"] == 1024


@pytest.mark.asyncio
async def test_optimize_wait_false_returns_task() -> None:
    optimize_result = OptimizeResult(
        status="success",
        collection_name="test_collection",
        compaction_id=2,
        target_size="512MB",
        progress=[ProgressStage.INITIALIZING],
    )

    with patch(
        "pymilvus.milvus_client.async_milvus_client.AsyncConnectionManager",
        return_value=MagicMock(),
    ), patch("pymilvus.orm.connections.Connections._fetch_handler") as mock_fetch:
        handler = MagicMock()
        handler.get_server_type.return_value = "milvus"
        mock_fetch.return_value = handler
        client = AsyncMilvusClient()
        client._execute_optimize = AsyncMock(return_value=optimize_result)

        task = await client.optimize("test_collection", target_size="512MB", wait=False)

    assert isinstance(task, AsyncOptimizeTask)
    assert task.done() is False
    final_result = await task.result()
    assert final_result == optimize_result
    client._execute_optimize.assert_awaited_once()
    assert client._execute_optimize.call_args.kwargs["size_mb"] == 512
    assert task.done() is True


@pytest.mark.asyncio
async def test_async_optimize_task_can_be_cancelled_before_completion() -> None:
    async def long_running_execute(
        task: AsyncOptimizeTask,
        collection_name: str,
        size_mb: int,
        timeout: Optional[float],
    ) -> OptimizeResult:
        await asyncio.sleep(1)
        return OptimizeResult(
            status="success",
            collection_name=collection_name,
            compaction_id=3,
            target_size=str(size_mb),
            progress=[ProgressStage.COMPACTING],
        )

    task = AsyncOptimizeTask(
        collection_name="cancel_collection",
        target_size="1MB",
        task_timeout=None,
        execute_fn=long_running_execute,
    )
    task.start()
    await asyncio.sleep(0)
    assert task.cancel() is True

    with pytest.raises(MilvusException):
        await task.result(timeout=0.5)

    assert task.cancelled() is True
    assert task.progress() == ProgressStage.CANCELLED
    assert ProgressStage.CANCELLED in task.progress_history()


# ── parse_target_size ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_value, expected_mb",
    [
        ("1GB", 1024),
        ("512mb", 512),
        ("1.5GB", 1536),
        (1048576, 1),
        (1.5 * 1024 * 1024, 1),
    ],
)
def test_parse_target_size_converts_valid_values(input_value: Any, expected_mb: int) -> None:
    assert parse_target_size(input_value) == expected_mb


@pytest.mark.parametrize(
    "invalid_value",
    ["abc", "10XB", "1KB", "-1GB", "-100", object()],
)
def test_parse_target_size_invalid_or_negative_values(invalid_value: Any) -> None:
    with pytest.raises(ParamError):
        parse_target_size(invalid_value)


def test_parse_target_size_none_raises():
    with pytest.raises(ParamError):
        parse_target_size(None)


def test_optimize_default_size_mb_is_sentinel():
    assert _OPTIMIZE_DEFAULT_SIZE_MB == 1 << 63 - 1


def test_parse_target_size_float_bytes():
    result = parse_target_size(2.5 * 1024 * 1024)
    assert result == 2


# ── TestOptimizeTask ──────────────────────────────────────────────────────────


class TestOptimizeTask:
    def test_initial_state(self):
        task = _make_task()
        assert task.done() is False
        assert task.cancelled() is False
        assert task.progress() == ProgressStage.INITIALIZING
        assert task.progress_history() == [ProgressStage.INITIALIZING]

    def test_run_success(self):
        expected = _make_result()
        task = _make_task(execute_fn=lambda task, collection_name, size_mb, timeout, **kw: expected)
        task.run()
        assert task.done() is True
        assert task.result() == expected

    def test_run_exception(self):
        error = MilvusException(message="exec error")

        def raise_fn(task, collection_name, size_mb, timeout, **kw):
            raise error

        task = _make_task(execute_fn=raise_fn)
        task.run()
        assert task.done() is True
        with pytest.raises(MilvusException, match="exec error"):
            task.result()

    def test_cancel_before_done(self):
        task = _make_task()
        assert task.cancel() is True
        assert task.cancelled() is True
        assert task.done() is True
        assert task.progress() == ProgressStage.CANCELLED
        assert ProgressStage.CANCELLED in task.progress_history()

    def test_cancel_already_cancelled(self):
        task = _make_task()
        task.cancel()
        assert task.cancel() is True

    def test_cancel_when_done(self):
        expected = _make_result()
        task = _make_task(execute_fn=lambda task, collection_name, size_mb, timeout, **kw: expected)
        task.run()
        assert task.cancel() is False

    def test_check_cancelled_raises_when_cancelled(self):
        task = _make_task()
        task.cancel()
        with pytest.raises(MilvusException):
            task.check_cancelled()

    def test_check_cancelled_no_raise_when_not_cancelled(self):
        task = _make_task()
        task.check_cancelled()  # Should not raise

    def test_set_progress(self):
        task = _make_task()
        task.set_progress(ProgressStage.COMPACTING)
        assert task.progress() == ProgressStage.COMPACTING
        assert ProgressStage.COMPACTING in task.progress_history()

    def test_set_progress_noop_when_cancelled(self):
        task = _make_task()
        task.cancel()
        task.set_progress(ProgressStage.COMPACTING)
        assert task.progress() == ProgressStage.CANCELLED

    def test_result_timeout_returns_none_when_not_started(self):
        task = _make_task()
        result = task.result(timeout=0.05)
        assert result is None

    @pytest.mark.parametrize(
        "execute_fn,match",
        [
            # cancelled before run: execute_fn returns → cancellation exception
            (lambda task, collection_name, size_mb, timeout, **kw: _make_result(), "cancelled"),
            # cancelled before run: execute_fn raises → cancellation exception (not the inner error)
            (
                lambda task, collection_name, size_mb, timeout, **kw: (_ for _ in ()).throw(
                    MilvusException(message="runtime error")
                ),
                "cancelled",
            ),
        ],
    )
    def test_run_cancelled_before_run(self, execute_fn, match):
        task = _make_task(execute_fn=execute_fn)
        task.cancel()
        task.run()
        with pytest.raises(MilvusException, match=match):
            task.result()

    def test_threading_run_and_wait(self):
        expected = _make_result(compaction_id=42)
        task = _make_task(
            execute_fn=lambda task, collection_name, size_mb, timeout, **kw: expected,
            task_timeout=5.0,
            target_size="100MB",
        )
        task.start()
        result = task.result(timeout=5.0)
        assert result == expected
        assert task.done() is True

    def test_progress_history_returns_copy(self):
        task = _make_task()
        history = task.progress_history()
        history.append("extra")
        assert "extra" not in task.progress_history()
