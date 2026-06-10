"""Tests for pymilvus/orm/db.py — database management functions."""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus.orm import db

CONNECTIONS_PREFIX = "pymilvus.orm.db.connections"


@pytest.fixture
def mock_handler():
    """Patch connections._fetch_handler to return a MagicMock handler."""
    handler = MagicMock()
    with patch(f"{CONNECTIONS_PREFIX}._fetch_handler", return_value=handler) as fetch:
        yield handler, fetch


@pytest.fixture
def mock_context():
    """Patch connections._generate_call_context to return a sentinel context."""
    ctx = MagicMock(name="context")
    with patch(f"{CONNECTIONS_PREFIX}._generate_call_context", return_value=ctx) as gen:
        yield ctx, gen


class TestUsingDatabase:
    """Tests for db.using_database."""

    def test_using_database_default_alias(self):
        with patch(f"{CONNECTIONS_PREFIX}._update_db_name") as m:
            db.using_database("mydb")
            m.assert_called_once_with("default", "mydb")

    def test_using_database_custom_alias(self):
        with patch(f"{CONNECTIONS_PREFIX}._update_db_name") as m:
            db.using_database("mydb", using="custom")
            m.assert_called_once_with("custom", "mydb")


class TestCreateDatabase:
    """Tests for db.create_database."""

    def test_default_args(self, mock_handler, mock_context):
        handler, _ = mock_handler
        ctx, gen_ctx = mock_context

        db.create_database("testdb")

        gen_ctx.assert_called_once_with("default")
        handler.create_database.assert_called_once_with("testdb", timeout=None, context=ctx)

    def test_custom_using_and_timeout(self, mock_handler, mock_context):
        handler, fetch = mock_handler
        ctx, gen_ctx = mock_context

        db.create_database("testdb", using="alias1", timeout=5.0)

        fetch.assert_called_with("alias1")
        gen_ctx.assert_called_once_with("alias1")
        handler.create_database.assert_called_once_with("testdb", timeout=5.0, context=ctx)

    def test_extra_kwargs_forwarded(self, mock_handler, mock_context):
        handler, _ = mock_handler
        ctx, gen_ctx = mock_context

        db.create_database("testdb", properties={"key": "val"})

        gen_ctx.assert_called_once_with("default", properties={"key": "val"})
        handler.create_database.assert_called_once_with(
            "testdb", timeout=None, context=ctx, properties={"key": "val"}
        )


class TestDropDatabase:
    """Tests for db.drop_database."""

    def test_default_args(self, mock_handler, mock_context):
        handler, _ = mock_handler
        ctx, gen_ctx = mock_context

        db.drop_database("testdb")

        gen_ctx.assert_called_once_with("default")
        handler.drop_database.assert_called_once_with("testdb", timeout=None, context=ctx)

    def test_custom_using_and_timeout(self, mock_handler, mock_context):
        handler, fetch = mock_handler
        ctx, gen_ctx = mock_context

        db.drop_database("testdb", using="alias2", timeout=3.0)

        fetch.assert_called_with("alias2")
        gen_ctx.assert_called_once_with("alias2")
        handler.drop_database.assert_called_once_with("testdb", timeout=3.0, context=ctx)


class TestListDatabase:
    """Tests for db.list_database."""

    def test_returns_handler_result(self, mock_handler, mock_context):
        handler, _ = mock_handler
        ctx, gen_ctx = mock_context
        handler.list_database.return_value = ["default", "testdb"]

        result = db.list_database()

        gen_ctx.assert_called_once_with("default")
        handler.list_database.assert_called_once_with(timeout=None, context=ctx)
        assert result == ["default", "testdb"]

    def test_custom_using_and_timeout(self, mock_handler, mock_context):
        handler, fetch = mock_handler
        ctx, gen_ctx = mock_context
        handler.list_database.return_value = []

        result = db.list_database(using="other", timeout=2.0)

        fetch.assert_called_with("other")
        gen_ctx.assert_called_once_with("other")
        handler.list_database.assert_called_once_with(timeout=2.0, context=ctx)
        assert result == []


class TestSetProperties:
    """Tests for db.set_properties."""

    def test_default_args(self, mock_handler, mock_context):
        handler, _ = mock_handler
        ctx, gen_ctx = mock_context
        props = {"database.replica.number": 2}

        db.set_properties("testdb", props)

        gen_ctx.assert_called_once_with("default")
        handler.alter_database.assert_called_once_with(
            "testdb", properties=props, timeout=None, context=ctx
        )

    def test_custom_using_and_timeout(self, mock_handler, mock_context):
        handler, fetch = mock_handler
        ctx, gen_ctx = mock_context
        props = {"database.resource_groups": ["rg1"]}

        db.set_properties("testdb", props, using="alias3", timeout=10.0)

        fetch.assert_called_with("alias3")
        gen_ctx.assert_called_once_with("alias3")
        handler.alter_database.assert_called_once_with(
            "testdb", properties=props, timeout=10.0, context=ctx
        )


class TestDescribeDatabase:
    """Tests for db.describe_database."""

    def test_returns_handler_result(self, mock_handler, mock_context):
        handler, _ = mock_handler
        ctx, gen_ctx = mock_context
        expected = {"name": "testdb", "properties": {}}
        handler.describe_database.return_value = expected

        result = db.describe_database("testdb")

        gen_ctx.assert_called_once_with("default")
        handler.describe_database.assert_called_once_with("testdb", timeout=None, context=ctx)
        assert result == expected

    def test_custom_using_and_timeout(self, mock_handler, mock_context):
        handler, fetch = mock_handler
        ctx, gen_ctx = mock_context
        handler.describe_database.return_value = {}

        result = db.describe_database("testdb", using="alias4", timeout=7.0)

        fetch.assert_called_with("alias4")
        gen_ctx.assert_called_once_with("alias4")
        handler.describe_database.assert_called_once_with("testdb", timeout=7.0, context=ctx)
        assert result == {}


class TestGetConnection:
    """Tests for db._get_connection helper."""

    def test_delegates_to_fetch_handler(self, mock_handler):
        handler, fetch = mock_handler

        result = db._get_connection("myalias")

        fetch.assert_called_once_with("myalias")
        assert result is handler
