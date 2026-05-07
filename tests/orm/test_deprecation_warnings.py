import warnings
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Index,
    Partition,
    PyMilvusDeprecationWarning,
    connections,
    utility,
)
from pymilvus.exceptions import ConnectionNotExistException
from pymilvus.orm import db
from pymilvus.orm.role import Role

from .conftest import GRPC_PREFIX


@pytest.fixture
def schema():
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=4),
        ]
    )


def connect_without_warning(mock_grpc_connect, mock_grpc_close):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyMilvusDeprecationWarning)
        connections.connect(keep_alive=False)


def make_collection(schema, mock_grpc_connect, mock_grpc_close):
    connect_without_warning(mock_grpc_connect, mock_grpc_close)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyMilvusDeprecationWarning)
        with mock.patch(f"{GRPC_PREFIX}.create_collection", return_value=None), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=False
        ):
            return Collection("warn_collection", schema=schema)


def test_connections_connect_warns(mock_grpc_connect, mock_grpc_close):
    with pytest.warns(
        PyMilvusDeprecationWarning,
        match="connections.connect.*removed in PyMilvus 3.1.*MilvusClient",
    ):
        connections.connect("warn_alias", uri="http://localhost:19530", keep_alive=False)


def test_connections_add_connection_warns():
    with pytest.warns(
        PyMilvusDeprecationWarning,
        match="connections.add_connection.*removed in PyMilvus 3.1.*MilvusClient",
    ):
        connections.add_connection(warn_alias={"host": "localhost", "port": "19530"})


@pytest.mark.asyncio
async def test_connections_async_disconnect_warns():
    alias = "warn_async_disconnect"
    handler = MagicMock()
    handler.close = mock.AsyncMock()
    connections._alias_handlers[alias] = handler
    connections._alias_config[alias] = {"address": "localhost:19530"}

    try:
        with pytest.warns(
            PyMilvusDeprecationWarning,
            match="connections.async_disconnect.*removed in PyMilvus 3.1.*MilvusClient",
        ) as records:
            await connections.async_disconnect(alias)

        assert len(records) == 1
        handler.close.assert_awaited_once()
        assert alias not in connections._alias_handlers
        assert alias in connections._alias_config
    finally:
        connections._alias_handlers.pop(alias, None)
        connections._alias_config.pop(alias, None)


@pytest.mark.asyncio
async def test_connections_async_remove_connection_warns():
    alias = "warn_async_remove_connection"
    handler = MagicMock()
    handler.close = mock.AsyncMock()
    connections._alias_handlers[alias] = handler
    connections._alias_config[alias] = {"address": "localhost:19530"}

    try:
        with pytest.warns(
            PyMilvusDeprecationWarning,
            match="connections.async_remove_connection.*removed in PyMilvus 3.1.*MilvusClient",
        ) as records:
            await connections.async_remove_connection(alias)

        assert len(records) == 1
        handler.close.assert_awaited_once()
        assert alias not in connections._alias_handlers
        assert alias not in connections._alias_config
    finally:
        connections._alias_handlers.pop(alias, None)
        connections._alias_config.pop(alias, None)


def test_collection_constructor_warns(schema, mock_grpc_connect, mock_grpc_close):
    connect_without_warning(mock_grpc_connect, mock_grpc_close)

    with pytest.warns(PyMilvusDeprecationWarning, match="Collection.*MilvusClient"):
        with mock.patch(f"{GRPC_PREFIX}.create_collection", return_value=None), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=False
        ):
            Collection("warn_collection", schema=schema)


def test_collection_method_warns(schema, mock_grpc_connect, mock_grpc_close):
    collection = make_collection(schema, mock_grpc_connect, mock_grpc_close)

    with pytest.warns(PyMilvusDeprecationWarning, match="Collection.flush.*MilvusClient"):
        with mock.patch(f"{GRPC_PREFIX}.flush", return_value=None):
            collection.flush()


def test_collection_iterators_do_not_warn(schema, mock_grpc_connect, mock_grpc_close):
    collection = make_collection(schema, mock_grpc_connect, mock_grpc_close)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PyMilvusDeprecationWarning)
        with mock.patch("pymilvus.orm.collection.QueryIterator", return_value=object()):
            collection.query_iterator()
        with mock.patch("pymilvus.orm.collection.SearchIterator", return_value=object()):
            collection.search_iterator(data=[[0.1, 0.2, 0.3, 0.4]], anns_field="vec", param={})


def test_partition_constructor_and_method_warn(schema, mock_grpc_connect, mock_grpc_close):
    collection = make_collection(schema, mock_grpc_connect, mock_grpc_close)

    with pytest.warns(PyMilvusDeprecationWarning, match="Partition.*MilvusClient"):
        with mock.patch.object(collection, "has_partition", return_value=True):
            partition = Partition(collection, "p1")

    with pytest.warns(PyMilvusDeprecationWarning, match="Partition.flush.*MilvusClient"):
        with mock.patch(f"{GRPC_PREFIX}.flush", return_value=None):
            partition.flush()


def test_index_constructor_and_method_warn(schema, mock_grpc_connect, mock_grpc_close):
    collection = make_collection(schema, mock_grpc_connect, mock_grpc_close)

    with pytest.warns(PyMilvusDeprecationWarning, match="Index.*MilvusClient"):
        index = Index(collection, "vec", {"index_type": "FLAT"}, construct_only=True)

    with pytest.warns(PyMilvusDeprecationWarning, match="Index.drop.*MilvusClient"):
        with mock.patch(f"{GRPC_PREFIX}.drop_index", return_value=None):
            index.drop()


def test_role_constructor_and_method_warn():
    with pytest.warns(PyMilvusDeprecationWarning, match="Role.*MilvusClient"):
        role = Role("admin")

    handler = MagicMock()
    with pytest.warns(PyMilvusDeprecationWarning, match="Role.create.*MilvusClient"):
        with mock.patch("pymilvus.orm.role.connections._fetch_handler", return_value=handler):
            role.create()


def test_db_function_warns():
    with pytest.warns(PyMilvusDeprecationWarning, match="db.list_database.*MilvusClient"):
        with pytest.raises(ConnectionNotExistException):
            db.list_database()


def test_connection_bound_utility_warns_and_timestamp_helper_is_silent():
    with pytest.warns(PyMilvusDeprecationWarning, match="utility.list_collections.*MilvusClient"):
        with pytest.raises(ConnectionNotExistException):
            utility.list_collections()

    with warnings.catch_warnings():
        warnings.simplefilter("error", PyMilvusDeprecationWarning)
        assert utility.mkts_from_unixtime(0.0) == 0
