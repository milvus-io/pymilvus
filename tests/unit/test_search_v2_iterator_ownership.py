import importlib
import warnings
from types import SimpleNamespace
from unittest.mock import patch

from pymilvus import PyMilvusDeprecationWarning
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
)
from pymilvus.milvus_client import milvus_client as milvus_client_module
from pymilvus.milvus_client.milvus_client import MilvusClient


class _Hit:
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
        return _SearchV2Result([_Hit(len(self.search_calls), self.transport)])

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
