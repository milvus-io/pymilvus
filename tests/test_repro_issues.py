"""Regression tests for pymilvus issues."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pymilvus import AnnSearchRequest, WeightedRanker
from pymilvus.client.abstract import CollectionSchema
from pymilvus.client.connection_manager import ConnectionManager
from pymilvus.client.types import ConsistencyLevel
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.milvus_client import MilvusClient

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_connection_manager():
    """Reset ConnectionManager singleton before and after each test."""
    ConnectionManager._reset_instance()
    yield
    ConnectionManager._reset_instance()


def make_client():
    mock_handler = MagicMock()
    mock_handler.get_server_type.return_value = "milvus"
    mock_handler._wait_for_channel_ready = MagicMock()
    with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
        return MilvusClient()


class TestIssue2587:
    """Regression test for #2587

    The error message for search/hybrid_search/flush with non-string collection
    name is not user friendly. Passing an integer (e.g. 1) should raise a clear
    ParamError rather than a cryptic gRPC internal error.
    """

    def test_issue_2587_search_non_string_collection_name(self):
        # Regression test for #2587
        client = make_client()
        with pytest.raises(ParamError, match="collection_name"):
            client.search(
                collection_name=1,
                data=[[0.1, 0.2]],
                limit=10,
            )

    def test_issue_2587_hybrid_search_non_string_collection_name(self):
        # Regression test for #2587
        client = make_client()
        reqs = [AnnSearchRequest([[0.1, 0.2]], "vec", {}, 10)]
        ranker = WeightedRanker(1.0)
        with pytest.raises(ParamError, match="collection_name"):
            client.hybrid_search(
                collection_name=1,
                reqs=reqs,
                ranker=ranker,
                limit=10,
            )

    def test_issue_2587_flush_non_string_collection_name(self):
        # Regression test for #2587
        client = make_client()
        with pytest.raises(ParamError, match="collection_name"):
            client.flush(collection_name=1)


class TestIssue2985:
    """Regression test for #2985

    PR #3409 (commit 07a29c8f) changed CollectionSchema.dict() to emit
    consistency_level as a string (ConsistencyLevel.Name(...)) instead of the
    historical int. This broke public API back-compat: downstream callers
    (including milvus e2e test_milvus_client_search_query_default) that
    indexed dict()['consistency_level'] expecting an int began failing.

    Dual-key fix: keep consistency_level as int (back-compat),
    add consistency_level_name as the human-readable string for callers who
    want the name. Users can migrate at leisure.
    """

    def _make_raw(self, consistency_level_int: int):
        """Build a minimal raw describe-collection response."""
        raw = MagicMock()
        raw.collection_name = "c"
        raw.auto_id = False
        raw.num_shards = 1
        raw.description = ""
        raw.schema = MagicMock()
        raw.schema.fields = []
        raw.schema.struct_array_fields = []
        raw.schema.functions = []
        raw.schema.enable_dynamic_field = False
        raw.schema.enable_namespace = False
        raw.schema.properties = []
        raw.aliases = []
        raw.collection_id = 0
        raw.consistency_level = consistency_level_int
        raw.properties = []
        raw.num_partitions = 1
        raw.created_timestamp = 0
        raw.update_timestamp = 0
        raw.external_source = None
        raw.external_spec = None
        raw.db_name = ""
        return raw

    def test_issue_2985_consistency_level_is_int_back_compat(self):
        # Regression test for #2985
        for level in (
            ConsistencyLevel.Strong,
            ConsistencyLevel.Session,
            ConsistencyLevel.Bounded,
            ConsistencyLevel.Eventually,
            ConsistencyLevel.Customized,
        ):
            raw = self._make_raw(int(level))
            d = CollectionSchema(raw).dict()
            # Back-compat: consistency_level MUST remain an int (the pre-#3409 shape).
            assert d["consistency_level"] == int(
                level
            ), f"Expected int {int(level)} for level {level}, got {d['consistency_level']!r}"

    def test_issue_2985_consistency_level_name_is_string(self):
        # Regression test for #2985
        expected = {
            ConsistencyLevel.Strong: "Strong",
            ConsistencyLevel.Session: "Session",
            ConsistencyLevel.Bounded: "Bounded",
            ConsistencyLevel.Eventually: "Eventually",
            ConsistencyLevel.Customized: "Customized",
        }
        for level, name in expected.items():
            raw = self._make_raw(int(level))
            d = CollectionSchema(raw).dict()
            # New key: consistency_level_name carries the human-readable string.
            assert d["consistency_level_name"] == name, (
                f"Expected name '{name}' for level {level}, got "
                f"{d.get('consistency_level_name')!r}"
            )
