"""Tests for pymilvus/orm/index.py — Index class."""

from unittest.mock import MagicMock

import pytest
from pymilvus.exceptions import CollectionNotExistException
from pymilvus.orm.collection import Collection
from pymilvus.orm.index import Index
from pymilvus.settings import Config


def _make_mock_collection(name="test_collection"):
    """Create a MagicMock that passes isinstance(obj, Collection)."""
    coll = MagicMock(spec=Collection)
    coll.name = name
    coll._name = name
    return coll


@pytest.fixture
def mock_collection():
    """A mock Collection that passes isinstance checks."""
    return _make_mock_collection()


@pytest.fixture
def mock_conn_and_context():
    """A (mock_conn, mock_context) pair for _get_connection."""
    conn = MagicMock(name="connection")
    ctx = MagicMock(name="context")
    return conn, ctx


class TestIndexInit:
    """Tests for Index.__init__."""

    def test_construct_only_stores_fields(self, mock_collection):
        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT", "metric_type": "L2"},
            construct_only=True,
        )
        assert idx._collection is mock_collection
        assert idx._field_name == "vec"
        assert idx._index_params == {"index_type": "IVF_FLAT", "metric_type": "L2"}
        assert idx._index_name == Config.IndexName

    def test_construct_only_with_custom_index_name(self, mock_collection):
        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "HNSW"},
            construct_only=True,
            index_name="my_index",
        )
        assert idx._index_name == "my_index"

    def test_raises_on_non_collection(self):
        with pytest.raises(CollectionNotExistException):
            Index("not_a_collection", "vec", {}, construct_only=True)

    def test_raises_on_none_collection(self):
        with pytest.raises(CollectionNotExistException):
            Index(None, "vec", {}, construct_only=True)

    def test_full_init_calls_create_index_and_list_indexes(
        self, mock_collection, mock_conn_and_context
    ):
        conn, ctx = mock_conn_and_context
        mock_collection._get_connection.return_value = (conn, ctx)

        # Simulate list_indexes returning an index matching our field
        matching_index = MagicMock()
        matching_index.field_name = "vec"
        matching_index.index_name = "server_index_name"
        conn.list_indexes.return_value = [matching_index]

        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
        )

        conn.create_index.assert_called_once_with(
            "test_collection",
            "vec",
            {"index_type": "IVF_FLAT"},
            context=ctx,
        )
        conn.list_indexes.assert_called_once_with("test_collection", context=ctx)
        assert idx._index_name == "server_index_name"

    def test_full_init_no_matching_index_keeps_default_name(
        self, mock_collection, mock_conn_and_context
    ):
        conn, ctx = mock_conn_and_context
        mock_collection._get_connection.return_value = (conn, ctx)

        # list_indexes returns an index for a different field
        other_index = MagicMock()
        other_index.field_name = "other_field"
        other_index.index_name = "other_index"
        conn.list_indexes.return_value = [other_index]

        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
        )

        assert idx._index_name == Config.IndexName

    def test_full_init_empty_list_indexes(self, mock_collection, mock_conn_and_context):
        conn, ctx = mock_conn_and_context
        mock_collection._get_connection.return_value = (conn, ctx)
        conn.list_indexes.return_value = []

        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "HNSW"},
        )

        assert idx._index_name == Config.IndexName


class TestIndexProperties:
    """Tests for Index properties (using construct_only=True)."""

    @pytest.fixture
    def index(self, mock_collection):
        return Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}},
            construct_only=True,
            index_name="my_idx",
        )

    def test_params_returns_deep_copy(self, index):
        params = index.params
        assert params == {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        # Modifying returned params should not affect the index
        params["index_type"] = "HNSW"
        assert index.params["index_type"] == "IVF_FLAT"

    def test_collection_name(self, index, mock_collection):
        assert index.collection_name == "test_collection"

    def test_field_name(self, index):
        assert index.field_name == "vec"

    def test_index_name(self, index):
        assert index.index_name == "my_idx"


class TestIndexEquality:
    """Tests for Index.__eq__."""

    def test_equal_indexes(self, mock_collection):
        idx1 = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
            index_name="idx",
        )
        idx2 = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
            index_name="idx",
        )
        assert idx1 == idx2

    def test_different_field_not_equal(self, mock_collection):
        idx1 = Index(
            mock_collection,
            "vec1",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
        )
        idx2 = Index(
            mock_collection,
            "vec2",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
        )
        assert idx1 != idx2

    def test_different_params_not_equal(self, mock_collection):
        idx1 = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
        )
        idx2 = Index(
            mock_collection,
            "vec",
            {"index_type": "HNSW"},
            construct_only=True,
        )
        assert idx1 != idx2


class TestIndexToDict:
    """Tests for Index.to_dict."""

    def test_to_dict(self, mock_collection):
        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
            index_name="my_idx",
        )
        result = idx.to_dict()
        assert result == {
            "collection": "test_collection",
            "field": "vec",
            "index_name": "my_idx",
            "index_param": {"index_type": "IVF_FLAT"},
        }


class TestIndexDrop:
    """Tests for Index.drop."""

    def test_drop_delegates_to_connection(self, mock_collection, mock_conn_and_context):
        conn, ctx = mock_conn_and_context
        mock_collection._get_connection.return_value = (conn, ctx)

        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
            index_name="my_idx",
        )

        idx.drop(timeout=5.0)

        conn.drop_index.assert_called_once_with(
            collection_name="test_collection",
            field_name="vec",
            index_name="my_idx",
            timeout=5.0,
            context=ctx,
        )

    def test_drop_default_timeout(self, mock_collection, mock_conn_and_context):
        conn, ctx = mock_conn_and_context
        mock_collection._get_connection.return_value = (conn, ctx)

        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
        )

        idx.drop()

        conn.drop_index.assert_called_once_with(
            collection_name="test_collection",
            field_name="vec",
            index_name=Config.IndexName,
            timeout=None,
            context=ctx,
        )


class TestIndexGetConnection:
    """Tests for Index._get_connection."""

    def test_delegates_to_collection(self, mock_collection, mock_conn_and_context):
        conn, ctx = mock_conn_and_context
        mock_collection._get_connection.return_value = (conn, ctx)

        idx = Index(
            mock_collection,
            "vec",
            {"index_type": "IVF_FLAT"},
            construct_only=True,
        )

        result = idx._get_connection()

        mock_collection._get_connection.assert_called_once_with()
        assert result == (conn, ctx)
