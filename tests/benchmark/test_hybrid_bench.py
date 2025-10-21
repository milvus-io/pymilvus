from unittest.mock import MagicMock

import pytest
from pymilvus import AnnSearchRequest, WeightedRanker

from . import mock_responses


class TestHybridBench:
    def test_hybrid_search_basic(self, benchmark, mocked_milvus_client) -> None:
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_hybrid_search_results(
                num_requests=2,
                top_k=10,
                output_fields=["id", "score"]
            )
        mocked_milvus_client._get_connection()._stub.HybridSearch = MagicMock(side_effect=custom_search)

        req1 = AnnSearchRequest([[0.1] * 128], "vector_field", {"metric_type": "L2"}, limit=10)
        req2 = AnnSearchRequest([[0.2] * 128], "vector_field", {"metric_type": "L2"}, limit=10)
        ranker = WeightedRanker(0.5, 0.5)

        result = benchmark(
            mocked_milvus_client.hybrid_search,
            collection_name="test_collection",
            reqs=[req1, req2],
            ranker=ranker,
            limit=10,
            output_fields=["id", "score"]
        )
        assert len(result) == 1


    @pytest.mark.parametrize("num_requests", [1, 10, 100, 1000, 10000])
    def test_hybrid_search_multiple_requests(self, benchmark, mocked_milvus_client, num_requests: int) -> None:

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_hybrid_search_results(
                num_requests=num_requests,
                top_k=10,
                output_fields=["id", "score"]
            )
        mocked_milvus_client._get_connection()._stub.HybridSearch = MagicMock(side_effect=custom_search)

        reqs = [
            AnnSearchRequest([[0.1] * 128], "vector_field", {"metric_type": "L2"}, limit=10)
            for _ in range(num_requests)
        ]
        weights = [1.0 / num_requests] * num_requests
        ranker = WeightedRanker(*weights)

        result = benchmark(
            mocked_milvus_client.hybrid_search,
            collection_name="test_collection",
            reqs=reqs,
            ranker=ranker,
            limit=10,
            output_fields=["id", "score"]
        )
        assert len(result) == 1


    @pytest.mark.parametrize("top_k", [1, 10, 100, 1000, 10000])
    def test_hybrid_search_varying_topk(self, benchmark, mocked_milvus_client, top_k: int) -> None:

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_hybrid_search_results(
                num_requests=2,
                top_k=top_k,
                output_fields=["id", "score"]
            )
        mocked_milvus_client._get_connection()._stub.HybridSearch = MagicMock(side_effect=custom_search)

        req1 = AnnSearchRequest([[0.1] * 128], "vector_field", {"metric_type": "L2"}, limit=top_k)
        req2 = AnnSearchRequest([[0.2] * 128], "vector_field", {"metric_type": "L2"}, limit=top_k)
        ranker = WeightedRanker(0.5, 0.5)

        result = benchmark(
            mocked_milvus_client.hybrid_search,
            collection_name="test_collection",
            reqs=[req1, req2],
            ranker=ranker,
            limit=top_k,
            output_fields=["id", "score"]
        )
        assert len(result) == 1
