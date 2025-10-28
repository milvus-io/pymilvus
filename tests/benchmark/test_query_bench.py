from unittest.mock import MagicMock

import pytest

from . import mock_responses


class TestQueryBench:
    @pytest.mark.parametrize("num_rows", [1, 10, 100, 1000, 10000, 65536])
    def test_query_basic_scalars(self, benchmark, mocked_milvus_client, num_rows: int) -> None:

        def custom_query(request, timeout=None, metadata=None):
            return mock_responses.create_query_results(
                num_rows=num_rows,
                output_fields=["id", "age", "score", "name"]
            )
        mocked_milvus_client._get_connection()._stub.Query = MagicMock(side_effect=custom_query)
        result = benchmark(
            mocked_milvus_client.query,
            collection_name="test_collection",
            filter="age > 25",
            output_fields=["id", "age", "score", "name"]
        )
        assert len(result) == num_rows


    @pytest.mark.parametrize("num_rows", [1, 100, 1000, 10000, 65536])
    def test_query_with_json_field(self, benchmark, mocked_milvus_client, num_rows: int) -> None:

        def custom_query(request, timeout=None, metadata=None):
            return mock_responses.create_query_results(
                num_rows=num_rows,
                output_fields=["id", "metadata"]
            )
        mocked_milvus_client._get_connection()._stub.Query = MagicMock(side_effect=custom_query)
        result = benchmark(
            mocked_milvus_client.query,
            collection_name="test_collection",
            filter="id > 0",
            output_fields=["id", "metadata"]
        )
        assert len(result) == num_rows


    @pytest.mark.parametrize("num_rows", [1, 100, 1000, 10000, 65536])
    def test_query_all_fields(self, benchmark, mocked_milvus_client, num_rows: int) -> None:
        def custom_query(request, timeout=None, metadata=None):
            return mock_responses.create_query_results(
                num_rows=num_rows,
                output_fields=["id", "age", "score", "name", "active", "metadata"]
            )
        mocked_milvus_client._get_connection()._stub.Query = MagicMock(side_effect=custom_query)
        result = benchmark(
            mocked_milvus_client.query,
            collection_name="test_collection",
            filter="id > 0",
            output_fields=["*"]
        )
        assert len(result) == num_rows
