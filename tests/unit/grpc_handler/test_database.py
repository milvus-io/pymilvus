"""Tests for GrpcHandler database operations."""

from unittest.mock import MagicMock

import pytest
from pymilvus.exceptions import ParamError

from .conftest import make_response, make_status


class TestGrpcHandlerDatabaseOps:
    """Tests for database operations."""

    def test_create_database(self, handler):
        handler._stub.CreateDatabase.return_value = make_status()
        handler.create_database("db")
        handler._stub.CreateDatabase.assert_called_once()

    def test_create_database_with_properties(self, handler):
        handler._stub.CreateDatabase.return_value = make_status()
        handler.create_database("db", properties={"replica.number": "2"})
        handler._stub.CreateDatabase.assert_called_once()

    @pytest.mark.parametrize("invalid_properties", ["hhh", [], [1, 2], 123, 1.11, True])
    def test_create_database_invalid_properties(self, handler, invalid_properties):
        with pytest.raises(ParamError):
            handler.create_database("db", properties=invalid_properties)

    def test_drop_database(self, handler):
        handler._stub.DropDatabase.return_value = make_status()
        handler.drop_database("db")
        handler._stub.DropDatabase.assert_called_once()

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

    @pytest.mark.parametrize("invalid_properties", ["hhh", [], [1, 2], 123, 1.11, True])
    def test_alter_database_invalid_properties(self, handler, invalid_properties):
        with pytest.raises(ParamError):
            handler.alter_database("db", invalid_properties)

    def test_drop_database_properties(self, handler):
        handler._stub.AlterDatabase.return_value = make_status()
        handler.drop_database_properties("db", ["key"])
        handler._stub.AlterDatabase.assert_called_once()
