"""Tests for GrpcHandler data operations."""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.client.types import DataType
from pymilvus.exceptions import ParamError

from .conftest import make_mutation_response


class TestGrpcHandlerDataOps:
    """Tests for data operations."""

    def test_insert_rows(self, handler, mock_schema):
        mock_resp = make_mutation_response(insert_cnt=2, ids=[1, 2])
        handler._stub.Insert.return_value = mock_resp

        with patch.object(handler, "_get_schema", return_value=(mock_schema, 0)):
            entities = [
                {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
                {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8]},
            ]
            result = handler.insert_rows("coll", entities)
            assert result.insert_count == 2

    def test_upsert_rows(self, handler, mock_schema):
        mock_resp = make_mutation_response(insert_cnt=1, ids=[1], upsert_cnt=1)
        handler._stub.Upsert.return_value = mock_resp

        with patch.object(handler, "_get_schema", return_value=(mock_schema, 0)):
            result = handler.upsert_rows("coll", [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}])
            assert result.insert_count == 1

    def test_delete_sync(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.delete_cnt = 3
        mock_resp.timestamp = 123456
        mock_resp.succ_index = []
        mock_resp.err_index = []
        mock_resp.IDs.WhichOneof.return_value = "int_id"
        mock_resp.IDs.int_id.data = []
        handler._stub.Delete.future.return_value.result.return_value = mock_resp
        result = handler.delete("coll", "id in [1,2,3]")
        assert result is not None

    def test_delete_async(self, handler):
        handler._stub.Delete.future.return_value = MagicMock()
        result = handler.delete("coll", "id > 0", _async=True)
        assert result is not None


class TestGrpcHandlerSearchOps:
    """Tests for search operations."""

    def test_search_sync(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = MagicMock()
        mock_resp.results.top_k = 10
        mock_resp.results.scores = []
        mock_resp.results.ids.WhichOneof.return_value = "int_id"
        mock_resp.results.ids.int_id.data = []
        mock_resp.results.output_fields = []
        mock_resp.results.fields_data = []
        mock_resp.session_ts = 123
        handler._stub.Search.return_value = mock_resp

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            handler.search("coll", "vec", {"metric_type": "L2"}, 10, data=[[0.1, 0.2, 0.3, 0.4]])
            handler._stub.Search.assert_called_once()

    def test_search_async(self, handler):
        mock_future = MagicMock()
        handler._stub.Search.future.return_value = mock_future

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            result = handler.search(
                "coll", "vec", {"metric_type": "L2"}, 10, data=[[0.1, 0.2, 0.3, 0.4]], _async=True
            )
            assert result is not None

    def test_hybrid_search_sync(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = MagicMock()
        mock_resp.results.top_k = 10
        mock_resp.results.scores = []
        mock_resp.results.ids.WhichOneof.return_value = "int_id"
        mock_resp.results.ids.int_id.data = []
        mock_resp.results.output_fields = []
        mock_resp.results.fields_data = []
        handler._stub.HybridSearch.return_value = mock_resp

        req = AnnSearchRequest(
            data=[[0.1, 0.2, 0.3, 0.4]], anns_field="vec", param={"metric_type": "L2"}, limit=10
        )
        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            handler.hybrid_search("coll", [req], RRFRanker(), 10)
            handler._stub.HybridSearch.assert_called_once()

    def test_hybrid_search_async(self, handler):
        mock_future = MagicMock()
        handler._stub.HybridSearch.future.return_value = mock_future

        req = AnnSearchRequest(
            data=[[0.1, 0.2, 0.3, 0.4]], anns_field="vec", param={"metric_type": "L2"}, limit=10
        )
        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            result = handler.hybrid_search("coll", [req], RRFRanker(), 10, _async=True)
            assert result is not None

    def test_search_with_ids(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = MagicMock()
        mock_resp.results.top_k = 10
        mock_resp.results.scores = []
        mock_resp.results.ids.WhichOneof.return_value = "int_id"
        mock_resp.results.ids.int_id.data = []
        mock_resp.results.output_fields = []
        mock_resp.results.fields_data = []
        mock_resp.session_ts = 123
        handler._stub.Search.return_value = mock_resp

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            handler.search("coll", "vec", {"metric_type": "L2"}, 10, ids=123)
            handler._stub.Search.assert_called_once()


class TestGrpcHandlerQueryOps:
    """Tests for query operations."""

    def test_query_success(self, handler):
        mock_field = MagicMock()
        mock_field.field_name = "id"
        mock_field.type = DataType.INT64
        mock_field.scalars.long_data.data = [1, 2, 3]
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.fields_data = [mock_field]
        mock_resp.session_ts = 123
        handler._stub.Query.return_value = mock_resp

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            with patch(
                "pymilvus.client.grpc_handler.entity_helper.extract_dynamic_field_from_result",
                return_value=([], []),
            ):
                with patch(
                    "pymilvus.client.grpc_handler.entity_helper.extract_row_data_from_fields_data_v2",
                    return_value=False,
                ):
                    with patch("pymilvus.client.grpc_handler.len_of", return_value=3):
                        handler.query("coll", "id > 0", ["id"])

    def test_query_invalid_output_fields(self, handler):
        with pytest.raises(ParamError):
            handler.query("coll", "id > 0", output_fields="invalid")


class TestGrpcHandlerBatchInsert:
    """Tests for batch insert operations."""

    def test_prepare_batch_insert_request(self, handler, mock_schema):
        """Test _prepare_batch_insert_request method."""
        entities = [[1, 2], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]
        with patch.object(handler, "describe_collection", return_value=mock_schema):
            with patch("pymilvus.client.grpc_handler.Prepare.batch_insert_param") as mock_prepare:
                mock_prepare.return_value = MagicMock()
                handler._prepare_batch_insert_request("coll", entities)
                mock_prepare.assert_called_once()

    def test_prepare_batch_insert_invalid_param(self, handler):
        """Test _prepare_batch_insert_request with invalid insert_param."""
        with pytest.raises(ParamError):
            handler._prepare_batch_insert_request("coll", [[1]], insert_param="invalid")
