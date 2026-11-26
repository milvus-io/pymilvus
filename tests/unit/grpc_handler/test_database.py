"""Tests for GrpcHandler database operations."""

from unittest.mock import MagicMock

from pymilvus.client.cache import GlobalCache

from .conftest import make_response, make_status


class TestGrpcHandlerDatabaseOps:
    """Tests for database operations."""

    def test_create_database(self, handler):
        handler._stub.CreateDatabase.return_value = make_status()
        handler.create_database("db")
        handler._stub.CreateDatabase.assert_called_once()

    def test_drop_database(self, handler):
        handler._stub.DropDatabase.return_value = make_status()
        handler.drop_database("db")
        handler._stub.DropDatabase.assert_called_once()

    def test_drop_database_evicts_that_database_from_both_caches(self, handler):
        GlobalCache._reset_for_testing()
        try:
            GlobalCache.schema.set(handler.server_address, "db", "coll", {"fields": []})
            GlobalCache.collection_ts.set(handler.server_address, "db", "coll", 100)
            GlobalCache.schema.set(handler.server_address, "other", "coll", {"fields": []})
            GlobalCache.collection_ts.set(handler.server_address, "other", "coll", 200)

            handler._stub.DropDatabase.return_value = make_status()
            handler.drop_database("db")

            assert GlobalCache.schema.get(handler.server_address, "db", "coll") is None
            assert GlobalCache.collection_ts.get(handler.server_address, "db", "coll") == 0
            # An unrelated database keeps its entries.
            assert GlobalCache.schema.get(handler.server_address, "other", "coll") is not None
            assert GlobalCache.collection_ts.get(handler.server_address, "other", "coll") == 200
        finally:
            GlobalCache._reset_for_testing()

    def test_list_database(self, handler):
        handler._stub.ListDatabases.return_value = make_response(db_names=["default", "db1"])
        assert handler.list_database() == ["default", "db1"]

    def test_describe_database(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.db_name = "test_db"
        mock_resp.created_timestamp = 0
        mock_resp.properties = []
        handler._stub.DescribeDatabase.return_value = mock_resp
        result = handler.describe_database("test_db")
        assert result is not None

    def test_alter_database(self, handler):
        handler._stub.AlterDatabase.return_value = make_status()
        handler.alter_database("db", {"key": "value"})
        handler._stub.AlterDatabase.assert_called_once()

    def test_drop_database_properties(self, handler):
        handler._stub.AlterDatabase.return_value = make_status()
        handler.drop_database_properties("db", ["key"])
        handler._stub.AlterDatabase.assert_called_once()
