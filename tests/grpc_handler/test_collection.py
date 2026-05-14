"""Tests for GrpcHandler collection operations."""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus import FieldSchema
from pymilvus.client.cache import GlobalCache
from pymilvus.client.types import DataType
from pymilvus.exceptions import DescribeCollectionException
from pymilvus.grpc_gen import common_pb2

from .conftest import (
    COLLECTION_VALIDATION_CASES,
    HAS_COLLECTION_RESPONSE_CASES,
    make_response,
    make_status,
)


class TestGrpcHandlerCollectionOps:
    """Tests for collection operations."""

    @pytest.mark.parametrize("coll_name,expected_error", COLLECTION_VALIDATION_CASES)
    def test_create_collection_validation(self, handler, coll_name, expected_error):
        with pytest.raises(expected_error):
            handler.create_collection(coll_name, fields={})

    @pytest.mark.parametrize("coll_name,expected_error", COLLECTION_VALIDATION_CASES)
    def test_drop_collection_validation(self, handler, coll_name, expected_error):
        with pytest.raises(expected_error):
            handler.drop_collection(coll_name)

    def test_drop_collection_success(self, handler):
        handler._stub.DropCollection.return_value = make_status()
        handler.drop_collection("test_coll")
        handler._stub.DropCollection.assert_called_once()

    @pytest.mark.parametrize(
        "error_code,status_code,reason,expected", HAS_COLLECTION_RESPONSE_CASES
    )
    def test_has_collection_responses(self, handler, error_code, status_code, reason, expected):
        mock_resp = MagicMock()
        mock_resp.status.error_code = error_code
        mock_resp.status.code = status_code
        mock_resp.status.reason = reason
        handler._stub.DescribeCollection.return_value = mock_resp
        assert handler.has_collection("test") == expected

    def test_describe_collection_success(self, handler):
        mock_resp = self._create_describe_collection_response()
        handler._stub.DescribeCollection.return_value = mock_resp
        result = handler.describe_collection("test_coll")
        assert result["collection_name"] == "test_coll"

    def test_describe_collection_not_found(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 100
        mock_resp.status.error_code = common_pb2.CollectionNotExists
        mock_resp.status.reason = "not found"
        handler._stub.DescribeCollection.return_value = mock_resp
        with pytest.raises(DescribeCollectionException):
            handler.describe_collection("nonexistent")

    def test_list_collections(self, handler):
        handler._stub.ShowCollections.return_value = make_response(collection_names=["c1", "c2"])
        assert handler.list_collections() == ["c1", "c2"]

    def test_rename_collections(self, handler):
        handler._stub.RenameCollection.return_value = make_status()
        handler.rename_collections("old", "new")
        handler._stub.RenameCollection.assert_called_once()

    def test_truncate_collection(self, handler):
        handler._stub.TruncateCollection.return_value = make_response()
        handler.truncate_collection("test")
        handler._stub.TruncateCollection.assert_called_once()

    def test_alter_collection_properties(self, handler):
        handler._stub.AlterCollection.return_value = make_status()
        handler.alter_collection_properties("test", {"ttl": "3600"})
        handler._stub.AlterCollection.assert_called_once()

    def test_get_collection_stats(self, handler):
        mock_stat = MagicMock(key="row_count", value="1000")
        handler._stub.GetCollectionStatistics.return_value = make_response(stats=[mock_stat])
        result = handler.get_collection_stats("test")
        assert len(result) == 1

    @staticmethod
    def _create_describe_collection_response():
        """Helper to create a valid describe collection response."""
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.collection_name = "test_coll"
        mock_resp.collectionID = 1
        mock_resp.shards_num = 1
        mock_resp.consistency_level = 0
        mock_resp.created_timestamp = 0
        mock_resp.created_utc_timestamp = 0
        mock_resp.num_partitions = 1
        mock_resp.properties = []
        mock_resp.db_name = "default"
        mock_resp.db_id = 1
        mock_resp.virtual_channel_names = []
        mock_resp.physical_channel_names = []
        mock_resp.aliases = []

        mock_schema = MagicMock()
        mock_schema.name = "test_coll"
        mock_schema.enable_dynamic_field = False
        mock_schema.functions = []

        pk_field = MagicMock()
        pk_field.name = "id"
        pk_field.data_type = DataType.INT64
        pk_field.is_primary_key = True
        pk_field.autoID = False
        pk_field.is_dynamic = False
        pk_field.is_partition_key = False
        pk_field.is_clustering_key = False
        pk_field.nullable = False
        pk_field.default_value = MagicMock()
        pk_field.default_value.WhichOneof.return_value = None
        pk_field.type_params = []
        pk_field.element_type = DataType.NONE

        mock_schema.fields = [pk_field]
        mock_resp.schema = mock_schema
        return mock_resp


class TestGrpcHandlerLoadOps:
    """Tests for load and release operations."""

    def test_load_collection(self, handler):
        handler._stub.LoadCollection.return_value = make_status()
        with patch.object(handler, "wait_for_loading_collection"):
            handler.load_collection("coll")
            handler._stub.LoadCollection.assert_called_once()

    def test_load_collection_no_wait(self, handler):
        handler._stub.LoadCollection.return_value = make_status()
        handler.load_collection("coll", _async=True)
        handler._stub.LoadCollection.assert_called_once()

    def test_release_collection(self, handler):
        handler._stub.ReleaseCollection.return_value = make_status()
        handler.release_collection("coll")
        handler._stub.ReleaseCollection.assert_called_once()

    def test_get_load_state_partition(self, handler):
        handler._stub.GetLoadState.return_value = make_response(state=3)
        result = handler.get_load_state("coll", partition_names=["p1"])
        assert result is not None


class TestGrpcHandlerCacheOps:
    """Tests for cache operations."""

    def test_get_schema_from_cache(self, handler):
        GlobalCache._reset_for_testing()
        cached = {"fields": [], "update_timestamp": 100}
        GlobalCache.schema.set(handler.server_address, "", "coll", cached)
        schema, ts = handler._get_schema("coll")
        assert schema == cached
        assert ts == 100
        GlobalCache._reset_for_testing()

    def test_get_schema_fetch_on_miss(self, handler):
        GlobalCache._reset_for_testing()
        remote = {"fields": [], "update_timestamp": 200}
        with patch.object(handler, "describe_collection", return_value=remote):
            schema, ts = handler._get_schema("coll")
            assert schema == remote
            assert ts == 200
            cached = GlobalCache.schema.get(handler.server_address, "", "coll")
            assert cached == remote
        GlobalCache._reset_for_testing()

    def test_invalidate_schema(self, handler):
        GlobalCache._reset_for_testing()
        GlobalCache.schema.set(handler.server_address, "", "coll", {"fields": []})
        handler._invalidate_schema("coll")
        assert GlobalCache.schema.get(handler.server_address, "", "coll") is None
        GlobalCache._reset_for_testing()


class TestGrpcHandlerCollectionProperties:
    """Tests for collection property operations."""

    def test_drop_collection_properties(self, handler):
        handler._stub.AlterCollection.return_value = make_status()
        handler.drop_collection_properties("coll", ["ttl"])
        handler._stub.AlterCollection.assert_called_once()

    def test_alter_collection_field_properties(self, handler):
        handler._stub.AlterCollectionField.return_value = make_status()
        handler.alter_collection_field_properties("coll", "field", {"mmap.enabled": "true"})
        handler._stub.AlterCollectionField.assert_called_once()

    def test_add_collection_field(self, handler):

        handler._stub.AddCollectionField.return_value = make_status()
        field = FieldSchema(name="new_field", dtype=DataType.VARCHAR, max_length=256)
        handler.add_collection_field("coll", field)
        handler._stub.AddCollectionField.assert_called_once()

    def test_drop_collection_function(self, handler):
        handler._stub.DropCollectionFunction.return_value = make_status()
        handler.drop_collection_function("coll", "func")
        handler._stub.DropCollectionFunction.assert_called_once()
