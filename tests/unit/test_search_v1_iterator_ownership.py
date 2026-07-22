import importlib
import warnings
from unittest.mock import patch

import pytest
from pymilvus import PyMilvusDeprecationWarning
from pymilvus.client import constants as client_constants
from pymilvus.client import iterator as client_iterator
from pymilvus.client.call_context import CallContext
from pymilvus.client.constants import COLLECTION_ID
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException, ServerVersionIncompatibleException
from pymilvus.milvus_client import milvus_client as milvus_client_module
from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus.orm import collection as collection_module
from pymilvus.orm import constants as orm_constants
from pymilvus.orm import iterator as orm_iterator
from pymilvus.orm.collection import Collection

_SCHEMA = {
    "fields": [
        {"name": "pk", "type": DataType.INT64, "is_primary": True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
    ]
}


class _Hit:
    def __init__(self, pk, distance, transport):
        self.id = pk
        self.distance = distance
        self.transport = transport


class _SearchResult:
    def __init__(self, hits, session_ts=100):
        self._hits = hits
        self._session_ts = session_ts

    def __getitem__(self, index):
        assert index == 0
        return self._hits

    def get_session_ts(self):
        return self._session_ts


class _SwitchingSearchHandler:
    def __init__(self):
        self.transport = "before"
        self.schema_calls = []
        self.describe_calls = []
        self.search_calls = []
        self.close_calls = 0

    def _get_schema(self, collection_name, **kwargs):
        self.schema_calls.append((collection_name, kwargs))
        return _SCHEMA, 1

    def describe_collection(self, collection_name, **kwargs):
        self.describe_calls.append((collection_name, kwargs))
        return {COLLECTION_ID: 1}

    def search(self, collection_name, **kwargs):
        self.search_calls.append((collection_name, kwargs))
        distance = 0.1 * len(self.search_calls)
        return _SearchResult([_Hit(len(self.search_calls), distance, self.transport)])

    def close(self):
        self.close_calls += 1


def _new_search_iterator(handler, context):
    return client_iterator.SearchIterator(
        handler=handler,
        context=context,
        collection_name="collection",
        data=[[0.1, 0.2]],
        ann_field="vec",
        param={"metric_type": "L2", "params": {}},
        batch_size=1,
        schema=_SCHEMA,
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
    handler = _SwitchingSearchHandler()
    context = CallContext(db_name="db", client_request_id="request")
    iterator = _new_search_iterator(handler, context)

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
    handler = _SwitchingSearchHandler()
    context = CallContext(db_name="db", client_request_id="request")
    collection = object.__new__(Collection)
    collection._name = "collection"
    collection._schema_dict = _SCHEMA

    with patch.object(collection, "_get_connection", return_value=(handler, context)):
        iterator = collection.search_iterator(
            data=[[0.1, 0.2]],
            anns_field="vec",
            param={"metric_type": "L2"},
            batch_size=1,
            client_request_id="request",
        )

    assert isinstance(iterator, client_iterator.SearchIterator)
    assert handler.describe_calls[0][1]["context"] is context
    assert all(call[1]["context"] is context for call in handler.search_calls)
    assert all(call[1]["client_request_id"] == "request" for call in handler.search_calls)


def test_milvus_client_v1_fallback_uses_shared_owner_and_one_context():
    handler = _SwitchingSearchHandler()
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
    handler = _SwitchingSearchHandler()
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
