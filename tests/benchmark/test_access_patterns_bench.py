from unittest.mock import MagicMock

import pytest

from . import mock_responses


class TestAccessPatternsBench:
    """Benchmark different access patterns for search results.
    
    Real-world usage varies:
    - UI display: Access first page only
    - Export/analysis: Iterate all results
    - Pagination: Random access to specific pages
    """

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_no_materialization(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Measure overhead of search without accessing results.
        
        This establishes baseline for result construction without materialization.
        Lazy fields (vectors, JSON) are not parsed.
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
            output_fields=["*"]
        )

        assert len(result) == 1
        assert len(result[0]) == top_k

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_access_first_only(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Measure cost of accessing only the first result.
        
        Simulates UI display of first page. Should materialize minimal data.
        """
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_all_types(
                num_queries=1,
                top_k=top_k,
                dim=128
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]

        def run_and_access_first():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["*"]
            )
            # Access first result - triggers materialization
            first = result[0][0]
            return first

        first_result = benchmark(run_and_access_first)
        assert first_result is not None

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_iterate_all(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Measure cost of iterating all results.
        
        Simulates export/analysis workload. Materializes all lazy fields.
        """
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_all_types(
                num_queries=1,
                top_k=top_k,
                dim=128
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]

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

    @pytest.mark.parametrize("top_k", [1000, 10000, 65536])
    def test_search_random_access_pattern(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Measure cost of random access to specific indices.
        
        Simulates pagination where user jumps to different pages.
        """
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results_all_types(
                num_queries=1,
                top_k=top_k,
                dim=128
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]

        def run_and_random_access():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["*"]
            )
            # Access different pages (indices 0, 50, 25, 75)
            page_indices = [0, 50, 25, 75]
            accessed = []
            for idx in page_indices:
                if idx < len(result[0]):
                    accessed.append(result[0][idx])
            return accessed

        accessed = benchmark(run_and_random_access)
        assert len(accessed) > 0

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_materialize_scalars_only(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Measure iteration over scalar fields only (no vectors).
        
        Scalars are eagerly loaded, so this should be faster than all-field iteration.
        """
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=1,
                top_k=top_k,
                output_fields=["id", "age", "score", "name"]
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]

        def run_and_iterate_scalars():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["id", "age", "score", "name"]
            )
            count = 0
            for hits in result:
                for hit in hits:
                    count += 1
            return count

        count = benchmark(run_and_iterate_scalars)
        assert count == top_k

    @pytest.mark.parametrize("top_k", [100, 1000, 10000, 65536])
    def test_search_materialize_vectors_only(self, benchmark, mocked_milvus_client, top_k: int) -> None:
        """Measure iteration with vector fields.
        
        Vectors are lazily loaded, should be slower than scalars.
        """
        def custom_search(request, timeout=None, metadata=None):
            return mock_responses.create_search_results(
                num_queries=1,
                top_k=top_k,
                output_fields=["id", "embedding"],
                include_vectors=True,
                dim=128
            )

        mocked_milvus_client._get_connection()._stub.Search = MagicMock(side_effect=custom_search)
        query_vectors = [[0.1] * 128]

        def run_and_iterate_vectors():
            result = mocked_milvus_client.search(
                collection_name="test_collection",
                data=query_vectors,
                limit=top_k,
                output_fields=["id", "embedding"]
            )
            count = 0
            for hits in result:
                for hit in hits:
                    count += 1
            return count

        count = benchmark(run_and_iterate_vectors)
        assert count == top_k
