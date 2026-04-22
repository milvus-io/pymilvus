"""Regression tests for pymilvus issues."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pymilvus import AnnSearchRequest, WeightedRanker
from pymilvus.client.connection_manager import ConnectionManager
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
