from unittest.mock import Mock, patch

import numpy as np
import pytest
from pymilvus.client.search_iterator import SearchIteratorV2
from pymilvus.client.search_result import SearchResult
from pymilvus.exceptions import ParamError, ServerVersionIncompatibleException
from pymilvus.grpc_gen import schema_pb2


class TestSearchIteratorV2:
    @pytest.fixture
    def mock_connection(self):
        connection = Mock()
        connection.describe_collection.return_value = {"collection_id": "test_id"}
        return connection

    @pytest.fixture
    def search_data(self):
        rng = np.random.default_rng(seed=19530)
        return rng.random((1, 8)).tolist()

    def create_mock_search_result(self, num_results=10):
        # Create mock search results
        mock_ids = schema_pb2.IDs(
            int_id=schema_pb2.LongArray(data=list(range(num_results)))
        )
        result = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=num_results,
            scores=[1.0 * i for i in range(num_results)],
            ids=mock_ids,
            topks=[num_results],
        )

        # Create mock iterator info
        result.search_iterator_v2_results.token = "test_token"
        result.search_iterator_v2_results.last_bound = 0.5

        return SearchResult(result)

    def test_init_basic(self, mock_connection, search_data):
        iterator = SearchIteratorV2(
            connection=mock_connection,
            collection_name="test_collection",
            data=search_data,
            batch_size=100
        )

        assert iterator._batch_size == 100
        assert iterator._left_res_cnt is None
        assert iterator._collection_id == "test_id"

    def test_init_with_limit(self, mock_connection, search_data):
        iterator = SearchIteratorV2(
            connection=mock_connection,
            collection_name="test_collection",
            data=search_data,
            batch_size=100,
            limit=50
        )

        assert iterator._left_res_cnt == 50

    def test_invalid_batch_size(self, mock_connection, search_data):
        with pytest.raises(ParamError):
            SearchIteratorV2(
                connection=mock_connection,
                collection_name="test_collection",
                data=search_data,
                batch_size=-1
            )

    def test_invalid_offset(self, mock_connection, search_data):
        with pytest.raises(ParamError):
            SearchIteratorV2(
                connection=mock_connection,
                collection_name="test_collection",
                data=search_data,
                batch_size=100,
                offset=10
            )

    def test_multiple_vectors_error(self, mock_connection):
        with pytest.raises(ParamError):
            SearchIteratorV2(
                connection=mock_connection,
                collection_name="test_collection",
                data=[[1, 2], [3, 4]],  # Multiple vectors
                batch_size=100
            )

    @patch('pymilvus.client.search_iterator.SearchIteratorV2._probe_for_compability')
    def test_next_without_external_filter(self, mock_probe, mock_connection, search_data):
        mock_connection.search.return_value = self.create_mock_search_result()
        iterator = SearchIteratorV2(
            connection=mock_connection,
            collection_name="test_collection",
            data=search_data,
            batch_size=100
        )

        result = iterator.next()
        assert result is not None
        assert len(result) == 10  # Number of results from mock

    @patch('pymilvus.client.search_iterator.SearchIteratorV2._probe_for_compability')
    def test_next_with_limit(self, mock_probe, mock_connection, search_data):
        mock_connection.search.return_value = self.create_mock_search_result()
        iterator = SearchIteratorV2(
            connection=mock_connection,
            collection_name="test_collection",
            data=search_data,
            batch_size=100,
            limit=5
        )

        result = iterator.next()
        assert result is not None
        assert len(result) == 5  # Limited to 5 results

    def test_server_incompatible(self, mock_connection, search_data):
        # Mock search result with empty token
        mock_result = self.create_mock_search_result()
        mock_result._search_iterator_v2_results.token = ""
        mock_connection.search.return_value = mock_result

        with pytest.raises(ServerVersionIncompatibleException):
            SearchIteratorV2(
                connection=mock_connection,
                collection_name="test_collection",
                data=search_data,
                batch_size=100
            )

    @patch('pymilvus.client.search_iterator.SearchIteratorV2._probe_for_compability')
    def test_external_filter(self, mock_probe, mock_connection, search_data):
        mock_connection.search.return_value = self.create_mock_search_result()

        def filter_func(hits):
            return [hit for hit in hits if hit["distance"] < 5.0]

        iterator = SearchIteratorV2(
            connection=mock_connection,
            collection_name="test_collection",
            data=search_data,
            batch_size=100,
            external_filter_func=filter_func
        )

        result = iterator.next()
        assert result is not None
        assert all(hit["distance"] < 5.0 for hit in result)

    @patch('pymilvus.client.search_iterator.SearchIteratorV2._probe_for_compability')
    def test_filter_and_external_filter(self, mock_probe, mock_connection, search_data):
        # Create mock search result with field values
        mock_result = self.create_mock_search_result()
        for hit in mock_result[0]:
            hit["entity"]["field_1"] = hit["id"] % 2
        mock_result[0] = list(filter(lambda x: x["entity"]["field_1"] < 5, mock_result[0]))
        mock_connection.search.return_value = mock_result

        expr_filter = "field_1 < 5"

        def filter_func(hits):
            return [hit for hit in hits if hit["distance"] < 5.0]  # Only hits with distance < 5.0 should pass

        iterator = SearchIteratorV2(
            connection=mock_connection,
            collection_name="test_collection",
            data=search_data,
            batch_size=100,
            filter=expr_filter,
            external_filter_func=filter_func
        )

        result = iterator.next()
        assert result is not None
        assert all(hit["distance"] < 5.0 and hit["entity"]["field_1"] < 5 for hit in result)
