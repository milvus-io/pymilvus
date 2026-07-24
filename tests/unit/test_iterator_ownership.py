import ast
import importlib
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pymilvus import PyMilvusDeprecationWarning
from pymilvus.client import constants as client_constants
from pymilvus.client import iterator as client_iterator
from pymilvus.client import search_iterator as legacy_search_iterator
from pymilvus.client.call_context import CallContext
from pymilvus.client.constants import (
    COLLECTION_ID,
    GUARANTEE_TIMESTAMP,
    ITER_SEARCH_BATCH_SIZE_KEY,
    ITER_SEARCH_ID_KEY,
    ITER_SEARCH_LAST_BOUND_KEY,
    ITER_SEARCH_V2_KEY,
    ITERATOR_FIELD,
    ITERATOR_SESSION_TS_FIELD,
    OFFSET,
)
from pymilvus.client.iterator import QueryIterator as ClientQueryIterator
from pymilvus.client.iterator import QueryIteratorCursor as ClientQueryIteratorCursor
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException, ServerVersionIncompatibleException
from pymilvus.milvus_client import milvus_client as milvus_client_module
from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus.orm import collection as collection_module
from pymilvus.orm import constants as orm_constants
from pymilvus.orm import iterator as orm_iterator
from pymilvus.orm.collection import Collection
from pymilvus.orm.iterator import QueryIterator as OrmQueryIterator
from pymilvus.orm.iterator import QueryIteratorCursor as OrmQueryIteratorCursor

pytestmark = pytest.mark.filterwarnings("error::pymilvus.PyMilvusDeprecationWarning")


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


_QUERY_SCHEMA = {
    "fields": [
        {"name": "pk", "type": DataType.INT64, "is_primary": True},
    ]
}


class _SwitchingQueryHandler:
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
        return _QUERY_SCHEMA, 1

    def query(self, collection_name, **kwargs):
        self.query_calls.append((collection_name, kwargs))
        if len(self.query_calls) == 1:
            return _QueryResult([])
        return _QueryResult([{"pk": len(self.query_calls) - 1, "transport": self.transport}])

    def close(self):
        self.close_calls += 1


def test_query_iterator_keeps_handler_identity_and_call_context_across_pages():
    handler = _SwitchingQueryHandler()
    context = CallContext(db_name="db", client_request_id="request")
    iterator = ClientQueryIterator(
        handler=handler,
        context=context,
        collection_name="collection",
        batch_size=10,
        expr="pk > 0",
        output_fields=["pk"],
        schema=_QUERY_SCHEMA,
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
    handler = _SwitchingQueryHandler()
    context = CallContext(db_name="db", client_request_id="request")

    ClientQueryIterator(
        handler=handler,
        context=context,
        collection_name="collection",
        batch_size=10,
        expr="pk > 0",
        output_fields=["pk"],
        schema=_QUERY_SCHEMA,
        rpc_options={OFFSET: 1},
    )

    assert len(handler.query_calls) == 2
    assert all(call[1]["context"] is context for call in handler.query_calls)
    assert handler.query_calls[1][1]["iterator"] == "False"


def test_milvus_client_public_seam_returns_shared_iterator_page():
    handler = _SwitchingQueryHandler()
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
    handler = _SwitchingQueryHandler()
    context = CallContext(db_name="db", client_request_id="request")
    collection = object.__new__(Collection)
    collection._name = "collection"
    collection._schema_dict = _QUERY_SCHEMA

    with pytest.warns(
        PyMilvusDeprecationWarning, match="Collection.query_iterator.*MilvusClient.query_iterator"
    ) as records:
        with patch.object(collection, "_get_connection", return_value=(handler, context)):
            iterator = collection.query_iterator(expr="pk > 0", client_request_id="request")

    assert len(records) == 1
    assert isinstance(iterator, ClientQueryIterator)
    assert iterator.next() == [{"pk": 1, "transport": "before"}]
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call[1]["context"] is context for call in handler.query_calls)
    assert all(call[1]["client_request_id"] == "request" for call in handler.query_calls)


_SEARCH_SCHEMA = {
    "fields": [
        {"name": "pk", "type": DataType.INT64, "is_primary": True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
    ]
}


class _SearchV1Hit:
    def __init__(self, pk, distance, transport):
        self.id = pk
        self.distance = distance
        self.transport = transport


class _SearchV1Result:
    def __init__(self, hits, session_ts=100):
        self._hits = hits
        self._session_ts = session_ts

    def __getitem__(self, index):
        assert index == 0
        return self._hits

    def get_session_ts(self):
        return self._session_ts


class _SwitchingSearchV1Handler:
    def __init__(self):
        self.transport = "before"
        self.schema_calls = []
        self.describe_calls = []
        self.search_calls = []
        self.close_calls = 0

    def _get_schema(self, collection_name, **kwargs):
        self.schema_calls.append((collection_name, kwargs))
        return _SEARCH_SCHEMA, 1

    def describe_collection(self, collection_name, **kwargs):
        self.describe_calls.append((collection_name, kwargs))
        return {COLLECTION_ID: 1}

    def search(self, collection_name, **kwargs):
        self.search_calls.append((collection_name, kwargs))
        distance = 0.1 * len(self.search_calls)
        return _SearchV1Result([_SearchV1Hit(len(self.search_calls), distance, self.transport)])

    def close(self):
        self.close_calls += 1


def _new_search_v1_iterator(handler, context):
    return client_iterator.SearchIterator(
        handler=handler,
        context=context,
        collection_name="collection",
        data=[[0.1, 0.2]],
        ann_field="vec",
        param={"metric_type": "L2", "params": {}},
        batch_size=1,
        schema=_SEARCH_SCHEMA,
        rpc_options={"cluster_id": "cluster"},
    )


def test_search_v1_compatibility_imports_share_client_owned_classes():
    assert orm_iterator.SearchIterator is client_iterator.SearchIterator
    assert orm_iterator.SearchPage is client_iterator.SearchPage
    assert milvus_client_module.SearchIterator is client_iterator.SearchIterator
    assert collection_module.SearchIterator is client_iterator.SearchIterator


def test_search_v1_compatibility_imports_emit_no_orm_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyMilvusDeprecationWarning)
        reloaded = importlib.reload(orm_iterator)

    assert reloaded.SearchIterator is client_iterator.SearchIterator
    assert reloaded.SearchPage is client_iterator.SearchPage


def test_search_v1_constants_are_canonical_in_client_layer():
    names = {
        "CALC_DIST_BM25",
        "CALC_DIST_COSINE",
        "CALC_DIST_HAMMING",
        "CALC_DIST_IP",
        "CALC_DIST_JACCARD",
        "CALC_DIST_L2",
        "CALC_DIST_TANIMOTO",
        "DEFAULT_SEARCH_EXTENSION_RATE",
        "EF",
        "MAX_FILTERED_IDS_COUNT_ITERATION",
        "MAX_TRY_TIME",
        "METRIC_TYPE",
        "PARAMS",
        "RADIUS",
        "RANGE_FILTER",
    }
    for name in names:
        assert getattr(orm_constants, name) == getattr(client_constants, name)


def test_search_v1_keeps_handler_identity_and_context_across_pages():
    handler = _SwitchingSearchV1Handler()
    context = CallContext(db_name="db", client_request_id="request")
    iterator = _new_search_v1_iterator(handler, context)

    first_page = iterator.next()
    handler.transport = "after"
    second_page = iterator.next()
    iterator.close()

    assert first_page[0].transport == "before"
    assert second_page[0].transport == "after"
    assert handler.describe_calls[0][1] == {
        "context": context,
        "cluster_id": "cluster",
        "iterator": "True",
    }
    assert all(call[1]["context"] is context for call in handler.search_calls)
    assert all(call[1]["cluster_id"] == "cluster" for call in handler.search_calls)
    assert all(call[1]["iterator"] == "True" for call in handler.search_calls)
    assert handler.close_calls == 0


def test_collection_search_iterator_uses_shared_v1_owner():
    handler = _SwitchingSearchV1Handler()
    context = CallContext(db_name="db", client_request_id="request")
    collection = object.__new__(Collection)
    collection._name = "collection"
    collection._schema_dict = _SEARCH_SCHEMA

    with pytest.warns(
        PyMilvusDeprecationWarning,
        match="Collection.search_iterator.*MilvusClient.search_iterator",
    ) as records:
        with patch.object(collection, "_get_connection", return_value=(handler, context)):
            iterator = collection.search_iterator(
                data=[[0.1, 0.2]],
                anns_field="vec",
                param={"metric_type": "L2"},
                batch_size=1,
                client_request_id="request",
            )

    assert len(records) == 1
    assert isinstance(iterator, client_iterator.SearchIterator)
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call[1]["context"] is context for call in handler.search_calls)
    assert all(call[1]["client_request_id"] == "request" for call in handler.search_calls)


def test_milvus_client_v1_fallback_uses_shared_owner_and_one_context():
    handler = _SwitchingSearchV1Handler()
    context = CallContext(db_name="db", client_request_id="request")
    client = object.__new__(MilvusClient)
    client._handler = handler

    with patch.object(client, "_generate_call_context", return_value=context) as generate_context:
        with patch.object(
            milvus_client_module,
            "SearchIteratorV2",
            side_effect=ServerVersionIncompatibleException(),
        ):
            iterator = client.search_iterator(
                "collection",
                data=[[0.1, 0.2]],
                anns_field="vec",
                search_params={"metric_type": "L2"},
                batch_size=1,
                client_request_id="request",
                cluster_id="cluster",
            )

    assert isinstance(iterator, client_iterator.SearchIterator)
    assert generate_context.call_count == 1
    assert handler.schema_calls[0][1]["context"] is context
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call[1]["context"] is context for call in handler.search_calls)
    assert all(call[1]["cluster_id"] == "cluster" for call in handler.search_calls)


def test_milvus_client_does_not_fallback_on_ordinary_v2_errors():
    handler = _SwitchingSearchV1Handler()
    client = object.__new__(MilvusClient)
    client._handler = handler
    error = MilvusException(message="search failed")
    context = CallContext(db_name="db", client_request_id="request")

    with patch.object(client, "_generate_call_context", return_value=context):
        with patch.object(milvus_client_module, "SearchIteratorV2", side_effect=error):
            with patch.object(milvus_client_module, "SearchIterator") as search_v1:
                with pytest.raises(MilvusException) as caught:
                    client.search_iterator("collection", data=[[0.1, 0.2]])

    assert caught.value is error
    search_v1.assert_not_called()


class _SearchV2Hit:
    def __init__(self, pk, transport):
        self.id = pk
        self.distance = float(pk)
        self.transport = transport


class _SearchV2Result:
    def __init__(self, hits, session_ts=100):
        self._hits = hits
        self._session_ts = session_ts
        self._iterator_info = SimpleNamespace(token="token", last_bound=0.5)

    def __getitem__(self, index):
        assert index == 0
        return self._hits

    def get_search_iterator_v2_results_info(self):
        return self._iterator_info

    def get_session_ts(self):
        return self._session_ts


class _SwitchingSearchV2Handler:
    def __init__(self):
        self.transport = "before"
        self.describe_calls = []
        self.search_calls = []
        self.close_calls = 0

    def describe_collection(self, collection_name, **kwargs):
        self.describe_calls.append((collection_name, kwargs))
        return {COLLECTION_ID: 1}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return _SearchV2Result([_SearchV2Hit(len(self.search_calls), self.transport)])

    def close(self):
        self.close_calls += 1


def _new_search_v2_iterator(handler, context):
    return client_iterator.SearchIteratorV2(
        handler=handler,
        context=context,
        collection_name="collection",
        data=[[0.1, 0.2]],
        batch_size=1,
        rpc_options={"cluster_id": "cluster"},
    )


def test_search_v2_compatibility_import_shares_client_owned_class():
    assert legacy_search_iterator.SearchIteratorV2 is client_iterator.SearchIteratorV2
    assert milvus_client_module.SearchIteratorV2 is client_iterator.SearchIteratorV2


def test_search_v2_compatibility_import_emits_no_orm_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyMilvusDeprecationWarning)
        reloaded = importlib.reload(legacy_search_iterator)

    assert reloaded.SearchIteratorV2 is client_iterator.SearchIteratorV2


def test_search_v2_keeps_handler_identity_and_context_across_probe_and_pages():
    handler = _SwitchingSearchV2Handler()
    context = CallContext(db_name="db", client_request_id="request")
    iterator = _new_search_v2_iterator(handler, context)

    handler.transport = "after"
    first_page = iterator.next()
    second_page = iterator.next()
    iterator.close()

    assert isinstance(first_page, client_iterator.SearchPage)
    assert first_page[0].transport == "after"
    assert second_page[0].transport == "after"
    assert handler.describe_calls[0][1] == {
        "context": context,
        "cluster_id": "cluster",
    }
    assert all(call["context"] is context for call in handler.search_calls)
    assert all(call["cluster_id"] == "cluster" for call in handler.search_calls)
    assert all(call[ITERATOR_FIELD] is True for call in handler.search_calls)
    assert all(call[ITER_SEARCH_V2_KEY] is True for call in handler.search_calls)
    assert all(call[COLLECTION_ID] == 1 for call in handler.search_calls)
    assert handler.search_calls[0][ITER_SEARCH_BATCH_SIZE_KEY] == 1
    assert handler.search_calls[2][ITER_SEARCH_ID_KEY] == "token"
    assert handler.search_calls[2][ITER_SEARCH_LAST_BOUND_KEY] == 0.5
    assert handler.close_calls == 0


def test_milvus_client_uses_shared_search_v2_owner_and_one_context():
    handler = _SwitchingSearchV2Handler()
    context = CallContext(db_name="db", client_request_id="request")
    client = object.__new__(MilvusClient)
    client._handler = handler

    with patch.object(client, "_generate_call_context", return_value=context) as generate_context:
        iterator = client.search_iterator(
            "collection",
            data=[[0.1, 0.2]],
            batch_size=1,
            client_request_id="request",
            cluster_id="cluster",
        )

    assert isinstance(iterator, client_iterator.SearchIteratorV2)
    assert generate_context.call_count == 1
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call["context"] is context for call in handler.search_calls)
    assert all(call["cluster_id"] == "cluster" for call in handler.search_calls)


def test_milvus_client_preserves_search_v2_public_preparation():
    handler = _SwitchingSearchV2Handler()
    context = CallContext(db_name="db", client_request_id="request")
    client = object.__new__(MilvusClient)
    client._handler = handler
    data = [[0.1, 0.2]]
    search_params = {"metric_type": "L2", "params": {"nprobe": 3}}

    with patch.object(client, "_generate_call_context", return_value=context):
        client.search_iterator(
            "collection",
            data=data,
            batch_size=7,
            filter="pk > 0",
            limit=11,
            output_fields=["pk"],
            search_params=search_params,
            timeout=2.5,
            partition_names=["partition"],
            anns_field=None,
            round_decimal=4,
            consistency_level="Strong",
            cluster_id="cluster",
        )

    probe = handler.search_calls[0]
    assert probe == {
        "collection_name": "collection",
        "data": data,
        "anns_field": "",
        "param": search_params,
        "limit": 1,
        "expression": "pk > 0",
        "partition_names": ["partition"],
        "output_fields": ["pk"],
        "timeout": 2.5,
        "round_decimal": 4,
        ITERATOR_FIELD: True,
        ITER_SEARCH_V2_KEY: True,
        ITER_SEARCH_BATCH_SIZE_KEY: 1,
        GUARANTEE_TIMESTAMP: 0,
        "consistency_level": "Strong",
        "cluster_id": "cluster",
        COLLECTION_ID: 1,
        "context": context,
    }
