import logging
from unittest.mock import patch

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
        GlobalCache.schema.set(
            handler.server_address, handler._get_db_name(), "test_collection", cached_schema
        )

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
            cached = GlobalCache.schema.get(
                handler.server_address, handler._get_db_name(), "test_collection"
            )
            assert cached == remote_schema

        # Cleanup
        GlobalCache._reset_for_testing()
