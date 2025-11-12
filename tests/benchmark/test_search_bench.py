from unittest.mock import MagicMock

import pytest

from . import mock_responses


class TestSearchBench:
    def test_search_float32_no_output_fields(self, benchmark, mocked_milvus_client):
        query_vectors = [[0.1] * 128]

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=len(query_vectors),
                top_k=10
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10
        )

        assert len(result) == len(query_vectors)

    def test_search_float32_basic_scalars(self, benchmark, mocked_milvus_client):
        query_vectors = [[0.1] * 128]

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=len(query_vectors),
                top_k=10,
                output_fields=["id", "age", "score", "name"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "age", "score", "name"]
        )

        assert len(result) == len(query_vectors)
        assert len(result[0]) == 10


    @pytest.mark.parametrize("top_k", [10, 100, 1000, 10000, 65536])
    def test_search_float32_varying_topk(self, benchmark, mocked_milvus_client, top_k):

        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=1,
                top_k=top_k,
                output_fields=["id", "age", "score"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

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
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=num_queries,
                top_k=10,
                output_fields=["id", "score"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128] * num_queries

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "score"]
        )

        assert len(result) == num_queries


    @pytest.mark.parametrize("dim", [128, 768, 1536])
    def test_search_float32_varying_dimensions(self, benchmark, mocked_milvus_client, dim):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=1,
                top_k=10,
                output_fields=["id"],
                include_vectors=True,
                dim=dim
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * dim]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "embedding"]
        )

        assert len(result) == 1


    def test_search_float16_vector(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_float16_vector(
                num_queries=1,
                top_k=10,
                output_fields=["id", "embedding"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "embedding"]
        )

        assert len(result) == 1


    def test_search_bfloat16_vector(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_bfloat16_vector(
                num_queries=1,
                top_k=10,
                output_fields=["id", "embedding"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "embedding"]
        )

        assert len(result) == 1


    def test_search_binary_vector(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_binary_vector(
                num_queries=1,
                top_k=10,
                output_fields=["id", "embedding"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [b'\x00' * 16]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "embedding"]
        )

        assert len(result) == 1


    def test_search_int8_vector(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_int8_vector(
                num_queries=1,
                top_k=10,
                output_fields=["id", "embedding"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "embedding"]
        )

        assert len(result) == 1


    def test_search_sparse_vector(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_sparse_vector(
                num_queries=1,
                top_k=10
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [{1: 0.5, 10: 0.3}]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10
        )

        assert len(result) == 1


    def test_search_with_json_output(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_json(
                num_queries=1,
                top_k=10,
                output_fields=["id", "metadata"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "metadata"]
        )

        assert len(result) == 1


    def test_search_with_array_output(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_array(
                num_queries=1,
                top_k=10,
                output_fields=["id", "tags"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "tags"]
        )

        assert len(result) == 1


    def test_search_with_geojson_output(self, benchmark, mocked_milvus_client):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_geojson(
                num_queries=1,
                top_k=10,
                output_fields=["id", "location"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "location"]
        )

        assert len(result) == 1


    @pytest.mark.parametrize("varchar_length", [10, 100, 1000, 10000, 65536])
    def test_search_with_varchar_sizes(self, benchmark, mocked_milvus_client, varchar_length):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_varchar(
                num_queries=1,
                top_k=10,
                varchar_length=varchar_length,
                output_fields=["id", "text"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "text"]
        )

        assert len(result) == 1


    @pytest.mark.parametrize("json_size", ["small", "medium", "large", "huge"])
    def test_search_with_json_sizes(self, benchmark, mocked_milvus_client, json_size):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_json(
                num_queries=1,
                top_k=10,
                json_size=json_size,
                output_fields=["id", "metadata"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)

        query_vectors = [[0.1] * 128]

        result = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "metadata"]
        )

        assert len(result) == 1


    @pytest.mark.parametrize("json_size", ["small", "medium", "large", "huge"])
    def test_search_with_json_sizes_materialized(self, benchmark, mocked_milvus_client, json_size):
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_with_json(
                num_queries=1,
                top_k=10,
                json_size=json_size,
                output_fields=["id", "metadata"]
            )
        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]
        res = benchmark(
            mocked_milvus_client.search,
            collection_name="test_collection",
            data=query_vectors,
            limit=10,
            output_fields=["id", "metadata"]
        )
        # Force materialization to include JSON parsing
        res.materialize()


    @pytest.mark.parametrize("top_k", [10, 100, 1000, 10000, 65536])
    def test_search_struct_field(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Benchmark struct field (ArrayOfStruct) parsing.
        
        Struct fields require column-to-row conversion, which is complex.
        This measures the overhead of struct field extraction.
        """
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
            output_fields=["id", "struct_field"]
        )

        assert len(result) == 1
        assert len(result[0]) == top_k


    @pytest.mark.parametrize("top_k", [10, 100, 1000, 10000, 65536])
    def test_search_struct_field_materialized(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Benchmark struct field with forced materialization.
        
        Forces full struct field conversion by iterating results.
        """
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_all_types(
                num_queries=1,
                top_k=top_k,
                dim=128
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]

        def run_and_materialize():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["id", "struct_field"]
            )
            # Force materialization
            count = 0
            for hits in result:
                for hit in hits:
                    count += 1
            return count

        count = benchmark(run_and_materialize)
        assert count == top_k
