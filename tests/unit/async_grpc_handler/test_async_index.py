"""Tests for AsyncGrpcHandler index and alias operations.

Coverage: Index create, drop, describe, wait_for_creating, alias operations.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.types import Status
from pymilvus.exceptions import AmbiguousIndexName, MilvusException, ParamError


class TestAsyncGrpcHandlerIndex:
    """Tests for index operations."""

    @pytest.mark.asyncio
    async def test_create_index_with_nested_field(self) -> None:
        """Test that create_index works with nested field names."""
        mock_channel = MagicMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        handler.wait_for_creating_index = AsyncMock(return_value=(True, ""))
        handler.alloc_timestamp = AsyncMock(return_value=12345)

        mock_create_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.reason = ""
        mock_create_response.status = mock_status
        mock_stub.CreateIndex = AsyncMock(return_value=mock_create_response)

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_index_request = MagicMock()
            mock_prepare.create_index_request.return_value = mock_index_request

            nested_field_name = "chunks[text_vector]"
            index_params = {
                "metric_type": "MAX_SIM_COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }

            await handler.create_index(
                collection_name="test_collection",
                field_name=nested_field_name,
                params=index_params,
                index_name="test_index",
            )

            mock_prepare.create_index_request.assert_called_once_with(
                "test_collection", nested_field_name, index_params, index_name="test_index"
            )
            mock_stub.CreateIndex.assert_called_once()
            handler.wait_for_creating_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_index(self) -> None:
        """Test drop_index async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.DropIndex = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.drop_index_request.return_value = mock_request

            await handler.drop_index("test_collection", "vector", "test_index", timeout=30)

            mock_stub.DropIndex.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_for_creating_index_finished(self) -> None:
        """Test wait_for_creating_index when index finishes"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.alloc_timestamp = AsyncMock(return_value=12345)

        # IndexState.Finished is 3
        handler.get_index_state = AsyncMock(return_value=(3, ""))

        result, _reason = await handler.wait_for_creating_index(
            "test_coll", "test_index", timeout=10
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_creating_index_failed(self) -> None:
        """Test wait_for_creating_index when index fails"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.alloc_timestamp = AsyncMock(return_value=12345)

        # IndexState.Failed is 4
        handler.get_index_state = AsyncMock(return_value=(4, "Build failed"))

        result, reason = await handler.wait_for_creating_index(
            "test_coll", "test_index", timeout=10
        )
        assert result is False
        assert reason == "Build failed"

    @pytest.mark.asyncio
    async def test_wait_for_creating_index_timeout(self) -> None:
        """Test wait_for_creating_index times out"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.alloc_timestamp = AsyncMock(return_value=12345)

        # IndexState.InProgress is 2
        handler.get_index_state = AsyncMock(return_value=(2, ""))

        with pytest.raises(MilvusException) as exc_info:
            await handler.wait_for_creating_index("test_coll", "test_index", timeout=1)
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_get_index_state(self) -> None:
        """Test get_index_state async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_index_desc = MagicMock()
        mock_index_desc.state = 3  # Finished
        mock_index_desc.index_state_fail_reason = ""
        mock_index_desc.field_name = "vector"
        mock_response.index_descriptions = [mock_index_desc]
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_index_request.return_value = MagicMock()
            state, reason = await handler.get_index_state("test_coll", "test_index")
            assert state == 3
            assert reason == ""

    @pytest.mark.asyncio
    async def test_describe_index(self) -> None:
        """Test describe_index async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_index = MagicMock()
        mock_index.field_name = "vector"
        mock_index.index_name = "test_index"
        mock_index.params = []
        mock_index.total_rows = 100
        mock_index.indexed_rows = 100
        mock_index.pending_index_rows = 0
        mock_index.state = 3  # Finished
        mock_response.index_descriptions = [mock_index]
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.describe_index_request.return_value = MagicMock()
            result = await handler.describe_index("test_coll", "test_index")
            assert result is not None

    @pytest.mark.asyncio
    async def test_describe_index_not_found(self) -> None:
        """Test describe_index returns None when index not found"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 700  # INDEX_NOT_FOUND
        mock_response.status.error_code = 0
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ):
            mock_prepare.describe_index_request.return_value = MagicMock()
            result = await handler.describe_index("test_coll", "test_index")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_index_state_with_field_name(self) -> None:
        """Test get_index_state with multiple index descriptions uses field_name"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_index_desc1 = MagicMock()
        mock_index_desc1.state = 3
        mock_index_desc1.index_state_fail_reason = ""
        mock_index_desc1.field_name = "vector1"
        mock_index_desc2 = MagicMock()
        mock_index_desc2.state = 2
        mock_index_desc2.index_state_fail_reason = ""
        mock_index_desc2.field_name = "vector2"
        mock_response.index_descriptions = [mock_index_desc1, mock_index_desc2]
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_index_request.return_value = MagicMock()
            state, _reason = await handler.get_index_state(
                "test_coll", "test_index", field_name="vector2"
            )
            assert state == 2

    @pytest.mark.asyncio
    async def test_get_index_state_ambiguous(self) -> None:
        """Test get_index_state raises AmbiguousIndexName for multiple indexes"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_index_desc1 = MagicMock()
        mock_index_desc1.state = 3
        mock_index_desc1.index_state_fail_reason = ""
        mock_index_desc1.field_name = "vector1"
        mock_index_desc2 = MagicMock()
        mock_index_desc2.state = 2
        mock_index_desc2.index_state_fail_reason = ""
        mock_index_desc2.field_name = "vector2"
        mock_response.index_descriptions = [mock_index_desc1, mock_index_desc2]
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_index_request.return_value = MagicMock()
            with pytest.raises(AmbiguousIndexName):
                await handler.get_index_state("test_coll", "test_index")

    @pytest.mark.asyncio
    async def test_list_indexes(self) -> None:
        """Test list_indexes async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_index = MagicMock()
        mock_index.index_name = "idx1"
        mock_response.index_descriptions = [mock_index]
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.is_successful", return_value=True):
            mock_prepare.describe_index_request.return_value = MagicMock()
            result = await handler.list_indexes("test_coll")
            assert result == [mock_index]

    @pytest.mark.asyncio
    async def test_list_indexes_not_found(self) -> None:
        """Test list_indexes returns empty when not found"""

        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 700
        mock_response.status.error_code = Status.INDEX_NOT_EXIST
        mock_stub.DescribeIndex = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.is_successful", return_value=False):
            mock_prepare.describe_index_request.return_value = MagicMock()
            result = await handler.list_indexes("test_coll")
            assert result == []

    @pytest.mark.asyncio
    async def test_alter_index_properties(self) -> None:
        """Test alter_index_properties async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterIndex = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_index_request.return_value = MagicMock()
            await handler.alter_index_properties("test_coll", "test_index", {"mmap": True})
            mock_stub.AlterIndex.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_index_properties(self) -> None:
        """Test drop_index_properties async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterIndex = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_index_request.return_value = MagicMock()
            await handler.drop_index_properties("test_coll", "test_index", ["key1"])
            mock_stub.AlterIndex.assert_called_once()

    @pytest.mark.asyncio
    async def test_alter_index_properties_none_error(self) -> None:
        """Test alter_index_properties raises on None properties"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        with pytest.raises(ParamError):
            await handler.alter_index_properties("test_coll", "test_index", None)


class TestAsyncGrpcHandlerAlias:
    """Tests for alias operations."""

    @pytest.mark.asyncio
    async def test_create_alias(self) -> None:
        """Test create_alias async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.CreateAlias = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.create_alias_request.return_value = mock_request

            await handler.create_alias("test_collection", "test_alias", timeout=30)

            mock_stub.CreateAlias.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_alias(self) -> None:
        """Test drop_alias async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_stub.DropAlias = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_request = MagicMock()
            mock_prepare.drop_alias_request.return_value = mock_request

            await handler.drop_alias("test_alias", timeout=30)

            mock_stub.DropAlias.assert_called_once()

    @pytest.mark.asyncio
    async def test_alter_alias(self) -> None:
        """Test alter_alias async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_stub.AlterAlias = AsyncMock(return_value=mock_status)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.alter_alias_request.return_value = MagicMock()
            await handler.alter_alias("test_coll", "test_alias")
            mock_stub.AlterAlias.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_aliases(self) -> None:
        """Test list_aliases async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.aliases = ["alias1", "alias2"]
        mock_stub.ListAliases = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.list_aliases_request.return_value = MagicMock()
            result = await handler.list_aliases("test_coll")
            assert result == ["alias1", "alias2"]

    @pytest.mark.asyncio
    async def test_describe_alias(self) -> None:
        """Test describe_alias async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.alias = "test_alias"
        mock_response.collection = "test_coll"
        mock_response.db_name = "default"
        mock_stub.DescribeAlias = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_alias_request.return_value = MagicMock()
            result = await handler.describe_alias("test_alias")
            assert result["alias"] == "test_alias"
