import ast
from pathlib import Path
from unittest.mock import patch

from pymilvus.client import constants as client_constants
from pymilvus.client.call_context import CallContext
from pymilvus.client.constants import COLLECTION_ID, ITERATOR_SESSION_TS_FIELD, OFFSET
from pymilvus.client.iterator import QueryIterator as ClientQueryIterator
from pymilvus.client.iterator import QueryIteratorCursor as ClientQueryIteratorCursor
from pymilvus.client.types import DataType
from pymilvus.milvus_client import milvus_client as milvus_client_module
from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus.orm import collection as collection_module
from pymilvus.orm import constants as orm_constants
from pymilvus.orm.collection import Collection
from pymilvus.orm.iterator import QueryIterator as OrmQueryIterator
from pymilvus.orm.iterator import QueryIteratorCursor as OrmQueryIteratorCursor


def test_query_iterator_compatibility_imports_share_client_owned_classes():
    assert OrmQueryIterator is ClientQueryIterator
    assert OrmQueryIteratorCursor is ClientQueryIteratorCursor
    assert milvus_client_module.QueryIterator is ClientQueryIterator
    assert collection_module.QueryIterator is ClientQueryIterator


def test_client_iterator_package_has_no_orm_imports():
    package = Path(__file__).parents[2] / "pymilvus" / "client" / "iterator"

    for source_path in package.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        assert not any(
            module.startswith("pymilvus.orm") for module in imported_modules
        ), f"{source_path} imports ORM code"


def test_query_iterator_constants_are_canonical_in_client_layer():
    names = {
        "BATCH_SIZE",
        "COLLECTION_ID",
        "FIELDS",
        "GUARANTEE_TIMESTAMP",
        "INT64_MAX",
        "IS_PRIMARY",
        "ITERATOR_FIELD",
        "ITERATOR_SESSION_CP_FILE",
        "ITERATOR_SESSION_TS_FIELD",
        "MAX_BATCH_SIZE",
        "MILVUS_LIMIT",
        "OFFSET",
        "QUERY_ITER_LAST_ELEMENT_OFFSET",
        "QUERY_ITER_LAST_PK",
        "REDUCE_STOP_FOR_BEST",
        "UNLIMITED",
    }
    constants_path = Path(__file__).parents[2] / "pymilvus" / "orm" / "constants.py"
    tree = ast.parse(constants_path.read_text(), filename=str(constants_path))
    literal_assignments = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert names.isdisjoint(literal_assignments)
    for name in names:
        assert getattr(orm_constants, name) == getattr(client_constants, name)


class _QueryResult(list):
    def __init__(self, rows, session_ts=100):
        super().__init__(rows)
        self.extra = {ITERATOR_SESSION_TS_FIELD: session_ts}


_SCHEMA = {
    "fields": [
        {"name": "pk", "type": DataType.INT64, "is_primary": True},
    ]
}


class _SwitchingHandler:
    def __init__(self):
        self.transport = "before"
        self.schema_calls = []
        self.describe_calls = []
        self.query_calls = []
        self.close_calls = 0

    def describe_collection(self, collection_name, **kwargs):
        self.describe_calls.append((collection_name, kwargs))
        return {COLLECTION_ID: 1}

    def _get_schema(self, collection_name, **kwargs):
        self.schema_calls.append((collection_name, kwargs))
        return _SCHEMA, 1

    def query(self, collection_name, **kwargs):
        self.query_calls.append((collection_name, kwargs))
        if len(self.query_calls) == 1:
            return _QueryResult([])
        return _QueryResult([{"pk": len(self.query_calls) - 1, "transport": self.transport}])

    def close(self):
        self.close_calls += 1


def test_query_iterator_keeps_handler_identity_and_call_context_across_pages():
    handler = _SwitchingHandler()
    context = CallContext(db_name="db", client_request_id="request")
    iterator = ClientQueryIterator(
        handler=handler,
        context=context,
        collection_name="collection",
        batch_size=10,
        expr="pk > 0",
        output_fields=["pk"],
        schema=_SCHEMA,
        rpc_options={"cluster_id": "cluster"},
    )

    assert iterator.next() == [{"pk": 1, "transport": "before"}]
    handler.transport = "after"
    assert iterator.next() == [{"pk": 2, "transport": "after"}]
    iterator.close()

    assert handler.describe_calls[0][1] == {
        "context": context,
        "cluster_id": "cluster",
    }
    assert all(call[1]["context"] is context for call in handler.query_calls)
    assert all(call[1]["cluster_id"] == "cluster" for call in handler.query_calls)
    assert all(call[1]["iterator"] == "True" for call in handler.query_calls)
    assert handler.close_calls == 0


def test_query_iterator_forwards_context_during_offset_seek():
    handler = _SwitchingHandler()
    context = CallContext(db_name="db", client_request_id="request")

    ClientQueryIterator(
        handler=handler,
        context=context,
        collection_name="collection",
        batch_size=10,
        expr="pk > 0",
        output_fields=["pk"],
        schema=_SCHEMA,
        rpc_options={OFFSET: 1},
    )

    assert len(handler.query_calls) == 2
    assert all(call[1]["context"] is context for call in handler.query_calls)
    assert handler.query_calls[1][1]["iterator"] == "False"


def test_milvus_client_public_seam_returns_shared_iterator_page():
    handler = _SwitchingHandler()
    context = CallContext(db_name="db", client_request_id="request")
    client = object.__new__(MilvusClient)
    client._handler = handler

    with patch.object(client, "_generate_call_context", return_value=context):
        iterator = client.query_iterator(
            "collection",
            filter="pk > 0",
            client_request_id="request",
            cluster_id="cluster",
        )

    assert isinstance(iterator, ClientQueryIterator)
    assert iterator.next() == [{"pk": 1, "transport": "before"}]
    assert handler.schema_calls[0][1]["context"] is context
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call[1]["context"] is context for call in handler.query_calls)
    assert all(call[1]["cluster_id"] == "cluster" for call in handler.query_calls)


def test_collection_public_seam_returns_shared_iterator_page():
    handler = _SwitchingHandler()
    context = CallContext(db_name="db", client_request_id="request")
    collection = object.__new__(Collection)
    collection._name = "collection"
    collection._schema_dict = _SCHEMA

    with patch.object(collection, "_get_connection", return_value=(handler, context)):
        iterator = collection.query_iterator(expr="pk > 0", client_request_id="request")

    assert isinstance(iterator, ClientQueryIterator)
    assert iterator.next() == [{"pk": 1, "transport": "before"}]
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call[1]["context"] is context for call in handler.query_calls)
    assert all(call[1]["client_request_id"] == "request" for call in handler.query_calls)
