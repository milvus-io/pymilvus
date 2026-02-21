import pytest

from . import mock_responses
from .conftest import (
    get_default_test_schema,
    setup_search_mock,
)


class TestSearchBench:
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


class TestGetColumnBench:
    """Benchmark tests for get_column() API with different return types."""

    def _setup_mock(self, mocked_milvus_client, top_k: int, output_fields: list) -> None:
        """Helper to setup search mock with specified top_k and output fields."""
        schema = get_default_test_schema()
        precomputed_results = mock_responses.create_search_results_from_schema(
            schema=schema, num_queries=1, top_k=top_k, output_fields=output_fields
        )

        def custom_search(request, timeout=None, metadata=None):
            return precomputed_results

        setup_search_mock(mocked_milvus_client, custom_search)

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_get_column_list(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Benchmark get_column with return_type='list'."""
        self._setup_mock(mocked_milvus_client, top_k, ["id", "age", "score"])
        query_vectors = [[0.1] * 128]

        def run_get_column_list():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["id", "age", "score"],
            )
            hits = result[0]
            ids = hits.get_column("id", return_type="list")
            ages = hits.get_column("age", return_type="list")
            scores = hits.get_column("score", return_type="list")
            return len(ids), len(ages), len(scores)

        counts = benchmark(run_get_column_list)
        assert counts == (top_k, top_k, top_k)

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_get_column_numpy(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Benchmark get_column with return_type='numpy'."""
        pytest.importorskip("numpy")
        self._setup_mock(mocked_milvus_client, top_k, ["id", "age", "score"])
        query_vectors = [[0.1] * 128]

        def run_get_column_numpy():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["id", "age", "score"],
            )
            hits = result[0]
            ids = hits.get_column("id", return_type="numpy")
            ages = hits.get_column("age", return_type="numpy")
            scores = hits.get_column("score", return_type="numpy")
            return len(ids), len(ages), len(scores)

        counts = benchmark(run_get_column_numpy)
        assert counts == (top_k, top_k, top_k)

    @pytest.mark.parametrize("return_type", ["list", "numpy"])
    def test_get_column_float_vector(
        self, benchmark, mocked_milvus_client, return_type: str
    ) -> None:
        """Benchmark get_column for FLOAT_VECTOR field with different return types."""
        if return_type == "numpy":
            pytest.importorskip("numpy")

        top_k = 10000
        self._setup_mock(mocked_milvus_client, top_k, ["embedding"])
        query_vectors = [[0.1] * 128]

        def run_get_column_vector():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["embedding"],
            )
            hits = result[0]
            vectors = hits.get_column("embedding", return_type=return_type)
            return len(vectors)

        count = benchmark(run_get_column_vector)
        assert count == top_k

    def test_get_column_vs_iteration(self, benchmark, mocked_milvus_client) -> None:
        """Compare get_column performance vs manual iteration."""
        pytest.importorskip("numpy")
        top_k = 10000
        self._setup_mock(mocked_milvus_client, top_k, ["id", "score"])
        query_vectors = [[0.1] * 128]

        def run_get_column():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["id", "score"],
            )
            hits = result[0]
            ids = hits.get_column("id", return_type="numpy")
            scores = hits.get_column("score", return_type="numpy")
            return ids.mean(), scores.mean()

        result = benchmark(run_get_column)
        assert result is not None
