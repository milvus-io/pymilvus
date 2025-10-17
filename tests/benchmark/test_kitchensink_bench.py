from unittest.mock import MagicMock

import pytest

from . import mock_responses


class TestKitchensinkBench:
    @pytest.mark.parametrize("top_k", [1, 10, 100, 1000, 10000])
    def test_search_all_output_types(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_all_types(
                num_queries=1,
                top_k=top_k,
                dim=128
            )
        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]
        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=top_k,
            output_fields=["*"]
        )
        assert len(result) == 1
        assert len(result[0]) == top_k


    @pytest.mark.parametrize("num_rows", [1, 10, 100, 1000, 10000])
    def test_query_all_output_types(self, benchmark, mocked_milvus_client, num_rows: int) -> None:
        def custom_query(request, timeout=None, metadata=None):
            return mock_responses.create_query_results_all_types(
                num_rows=num_rows,
                dim=128
            )
        mocked_milvus_client._get_connection()._stub.Query = MagicMock(side_effect=custom_query)
        result = benchmark(
            mocked_milvus_client.query,
            collection_name="test_collection",
            filter="id > 0",
            output_fields=["*"]
        )
        assert len(result) == num_rows
