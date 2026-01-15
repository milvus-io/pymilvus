import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import AsyncMilvusClient
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.milvus_client.async_optimize_task import AsyncOptimizeTask
from pymilvus.milvus_client.optimize_task import OptimizeResult, ProgressStage, parse_target_size


@pytest.mark.asyncio
async def test_optimize_wait_true_completes_successfully() -> None:
    optimize_result = OptimizeResult(
        status="success",
        collection_name="test_collection",
        compaction_id=1,
        target_size="1GB",
        progress=[ProgressStage.INITIALIZING],
    )

    with patch("pymilvus.milvus_client.async_milvus_client.create_connection", return_value="test"), \
            patch("pymilvus.orm.connections.Connections._fetch_handler") as mock_fetch:
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

    with patch("pymilvus.milvus_client.async_milvus_client.create_connection", return_value="test"), \
            patch("pymilvus.orm.connections.Connections._fetch_handler") as mock_fetch:
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
    [
        "abc",
        "10XB",
        "1KB",
        "-1GB",
        "-100",
        object(),
    ],
)
def test_parse_target_size_invalid_or_negative_values(invalid_value: Any) -> None:
    with pytest.raises(ParamError):
        parse_target_size(invalid_value)
