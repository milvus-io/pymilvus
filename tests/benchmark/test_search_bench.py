import pytest
from pymilvus import DataType, MilvusClient

from . import mock_responses
from .conftest import (
    get_default_test_schema,
    setup_search_mock,
)


class TestSearchBench:
    @pytest.mark.benchmark(group="nullable_large_topk_varchar")
    def test_search_nullable_varchar_256_topk_100000(self, benchmark, mocked_milvus_client):
        schema = MilvusClient.create_schema()
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=128)
        schema.add_field(
            field_name="varchar_field",
            datatype=DataType.VARCHAR,
            max_length=256,
            nullable=True,
        )

        query_vectors = [[0.1] * 128]
        top_k = 100_000
        output_fields = ["varchar_field"]
        precomputed_results = mock_responses.create_search_results_from_schema(
            schema=schema,
            num_queries=1,
            top_k=top_k,
            output_fields=output_fields,
        )
        precomputed_results.results.fields_data[0].valid_data.extend([True] * top_k)

        def custom_search(request, timeout=None, metadata=None):
            return precomputed_results

        setup_search_mock(mocked_milvus_client, custom_search)

        # One measured round bounds the cost of allocating 100,000 result objects.
        result = benchmark.pedantic(
            mocked_milvus_client.search,
            kwargs={
                "collection_name": "test_collection",
                "data": query_vectors,
                "limit": top_k,
                "output_fields": output_fields,
            },
            rounds=1,
            iterations=1,
        )

        assert len(result) == 1
        assert len(result[0]) == top_k
        assert len(result[0].get_raw_item(top_k - 1).entity["varchar_field"]) == 256

    @pytest.mark.parametrize(
        "output_fields",
        [None, ["id"], ["id", "age"], ["id", "age", "score"], ["id", "age", "score", "name"]],
    )
    def test_search_float32_varying_output_fields(
        self, benchmark, mocked_milvus_client, output_fields
    ):
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128]

        precomputed_results = mock_responses.create_search_results_from_schema(
            schema=schema, num_queries=len(query_vectors), top_k=10, output_fields=output_fields
        )

        def custom_search(request, timeout=None, metadata=None):
            return precomputed_results

        setup_search_mock(mocked_milvus_client, custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=output_fields,
        )

        assert len(result) == len(query_vectors)
        assert len(result[0]) == 10

    @pytest.mark.parametrize("top_k", [10, 100, 1000, 10000, 65536])
    def test_search_float32_varying_topk(self, benchmark, mocked_milvus_client, top_k):
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128]

        precomputed_results = mock_responses.create_search_results_from_schema(
            schema=schema, num_queries=1, top_k=top_k, output_fields=["id", "age", "score"]
        )

        def custom_search(request, timeout=None, metadata=None):
            return precomputed_results

        setup_search_mock(mocked_milvus_client, custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=top_k,
            output_fields=["id", "age", "score"],
        )

        assert len(result) == 1
        assert len(result[0]) == top_k

    @pytest.mark.parametrize("num_queries", [1, 10, 100, 1000, 10000])
    def test_search_float32_varying_num_queries(self, benchmark, mocked_milvus_client, num_queries):
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128] * num_queries

        precomputed_results = mock_responses.create_search_results_from_schema(
            schema=schema, num_queries=num_queries, top_k=10, output_fields=["id", "score"]
        )

        def custom_search(request, timeout=None, metadata=None):
            return precomputed_results

        setup_search_mock(mocked_milvus_client, custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "score"],
        )

        assert len(result) == num_queries

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_iterate_all(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        schema = get_default_test_schema()
        query_vectors = [[0.1] * 128]

        precomputed_results = mock_responses.create_search_results_from_schema(
            schema=schema, num_queries=1, top_k=top_k, output_fields=["*"]
        )

        def custom_search(request, timeout=None, metadata=None):
            return precomputed_results

        setup_search_mock(mocked_milvus_client, custom_search)

        def run_and_iterate_all():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["*"],
            )
            # Iterate all - materializes everything
            count = 0
            for hits in result:
                for hit in hits:
                    assert "id" in hit
                    count += 1
            return count

        count = benchmark(run_and_iterate_all)
        assert count == top_k
