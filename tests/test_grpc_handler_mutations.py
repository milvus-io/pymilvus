import logging
from unittest.mock import MagicMock, patch

from pymilvus.client.cache import GlobalCache
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.orm.types import DataType

log = logging.getLogger(__name__)


class TestGrpcHandlerHelperMethods:
    def test_get_info(self) -> None:
        handler = GrpcHandler(channel=None)

        # Test with schema provided
        schema = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vector", "type": DataType.FLOAT_VECTOR},
            ],
            "enable_dynamic_field": True,
        }

        fields_info, enable_dynamic = handler._get_info("test_collection", schema=schema)

        assert fields_info == schema["fields"]
        assert enable_dynamic is True

    def test_get_info_without_schema(self) -> None:
        handler = GrpcHandler(channel=None)

        with patch.object(handler, "describe_collection") as mock_describe:
            mock_describe.return_value = {
                "fields": [{"name": "id", "type": DataType.INT64}],
                "enable_dynamic_field": False,
            }

            fields_info, enable_dynamic = handler._get_info("test_collection")

            assert fields_info == [{"name": "id", "type": DataType.INT64}]
            assert enable_dynamic is False

    def test_get_schema_cached(self) -> None:

        # Reset singleton for clean test
        GlobalCache._reset_for_testing()

        handler = GrpcHandler(channel=None)

        # Add to global cache
        cached_schema = {
            "fields": [{"name": "id", "type": DataType.INT64}],
            "update_timestamp": 100,
        }
        GlobalCache.schema.set(handler.server_address, "", "test_collection", cached_schema)

        schema, timestamp = handler._get_schema("test_collection")

        assert schema == cached_schema
        assert timestamp == 100

        # Cleanup
        GlobalCache._reset_for_testing()

    def test_get_schema_not_cached(self) -> None:
        # Reset singleton for clean test
        GlobalCache._reset_for_testing()

        handler = GrpcHandler(channel=None)

        with patch.object(handler, "describe_collection") as mock_describe:
            remote_schema = {
                "fields": [{"name": "id", "type": DataType.INT64}],
                "update_timestamp": 200,
            }
            mock_describe.return_value = remote_schema

            schema, timestamp = handler._get_schema("test_collection")

            assert schema == remote_schema
            assert timestamp == 200

            # Check it was cached in global cache
            cached = GlobalCache.schema.get(handler.server_address, "", "test_collection")
            assert cached == remote_schema

        # Cleanup
        GlobalCache._reset_for_testing()


class TestGrpcHandlerGetServerVersion:
    def test_get_server_version_without_detail(self) -> None:
        """Test get_server_version returns version string when detail=False (default)"""
        handler = GrpcHandler(channel=None)

        mock_stub = MagicMock()
        handler._stub = mock_stub

        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_response.version = "2.6.6"
        mock_stub.GetVersion = MagicMock(return_value=mock_response)

        with patch("pymilvus.client.grpc_handler.check_status"):
            result = handler.get_server_version()

        assert result == "2.6.6"
        mock_stub.GetVersion.assert_called_once()

    def test_get_server_version_with_detail(self) -> None:
        """Test get_server_version returns server info dict when detail=True"""
        handler = GrpcHandler(channel=None)

        mock_stub = MagicMock()
        handler._stub = mock_stub

        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status

        mock_server_info = MagicMock()
        mock_server_info.build_tags = "2.6.6"
        mock_server_info.build_time = "Fri Jan 23 03:05:45 UTC 2026"
        mock_server_info.git_commit = "cebbe1e4da"
        mock_server_info.go_version = "go version go1.24.11 linux/amd64"
        mock_server_info.deploy_mode = "STANDALONE"
        mock_response.server_info = mock_server_info

        mock_stub.Connect = MagicMock(return_value=mock_response)

        with patch("pymilvus.client.grpc_handler.check_status"):
            result = handler.get_server_version(detail=True)

        expected = {
            "version": "2.6.6",
            "build_time": "Fri Jan 23 03:05:45 UTC 2026",
            "git_commit": "cebbe1e4da",
            "go_version": "go version go1.24.11 linux/amd64",
            "deploy_mode": "STANDALONE",
        }
        assert result == expected
        mock_stub.Connect.assert_called_once()

    def test_get_server_version_with_detail_uses_cache(self) -> None:
        """Test get_server_version caches server info and returns cached value"""
        handler = GrpcHandler(channel=None)

        mock_stub = MagicMock()
        handler._stub = mock_stub

        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status

        mock_server_info = MagicMock()
        mock_server_info.build_tags = "2.6.6"
        mock_server_info.build_time = "Fri Jan 23 03:05:45 UTC 2026"
        mock_server_info.git_commit = "cebbe1e4da"
        mock_server_info.go_version = "go version go1.24.11 linux/amd64"
        mock_server_info.deploy_mode = "STANDALONE"
        mock_response.server_info = mock_server_info

        mock_stub.Connect = MagicMock(return_value=mock_response)

        with patch("pymilvus.client.grpc_handler.check_status"):
            # First call should fetch from server
            result1 = handler.get_server_version(detail=True)
            # Second call should use cache
            result2 = handler.get_server_version(detail=True)

        assert result1 == result2
        # Connect should only be called once due to caching
        assert mock_stub.Connect.call_count == 1
