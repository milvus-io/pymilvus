import pytest

from . import mock_responses
from .conftest import (
    get_default_test_schema,
    setup_search_mock,
)


class TestSearchBench:
    @pytest.mark.parametrize("output_fields", [
        None,
        ["id"],
        ["id", "age"],
        ["id", "age", "score"],
        ["id", "age", "score", "name"]
    ])
    def test_search_float32_varying_output_fields(self, benchmark, mocked_milvus_client, output_fields):
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128]

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_from_schema(
                schema=schema,
                num_queries=len(query_vectors),
                top_k=10,
                output_fields=output_fields
            )

        setup_search_mock(mocked_milvus_client, custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=output_fields
        )

        assert len(result) == len(query_vectors)
        assert len(result[0]) == 10


    @pytest.mark.parametrize("top_k", [10, 100, 1000, 10000, 65536])
    def test_search_float32_varying_topk(self, benchmark, mocked_milvus_client, top_k):
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128]

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_from_schema(
                schema=schema,
                num_queries=1,
                top_k=top_k,
                output_fields=["id", "age", "score"]
            )

        setup_search_mock(mocked_milvus_client, custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=top_k,
            output_fields=["id", "age", "score"]
        )

        assert len(result) == 1
        assert len(result[0]) == top_k


    @pytest.mark.parametrize("num_queries", [1, 10, 100, 1000, 10000])
    def test_search_float32_varying_num_queries(self, benchmark, mocked_milvus_client, num_queries):
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128] * num_queries

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_from_schema(
                schema=schema,
                num_queries=num_queries,
                top_k=10,
                output_fields=["id", "score"]
            )

        setup_search_mock(mocked_milvus_client, custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "score"]
        )

        assert len(result) == num_queries


    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_iterate_all(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128]

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_from_schema(
                schema=schema,
                num_queries=1,
                top_k=top_k,
                output_fields=["*"]
            )

        setup_search_mock(mocked_milvus_client, custom_search)

        def run_and_iterate_all():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["*"]
            )
            # Iterate all - materializes everything
            count = 0
            for hits in result:
                for hit in hits:
                    count += 1
            return count

        count = benchmark(run_and_iterate_all)
        assert count == top_k
