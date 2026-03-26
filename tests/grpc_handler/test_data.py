"""Tests for GrpcHandler data operations."""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.client.call_context import CallContext
from pymilvus.client.types import DataType
from pymilvus.exceptions import (
    DataNotMatchException,
    MilvusException,
    ParamError,
    SchemaMismatchRetryableException,
)
from pymilvus.grpc_gen import common_pb2

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

    def test_search_stub_returns_none_raises_milvus_exception(self, handler):
        handler._stub.Search.return_value = None

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            with pytest.raises(
                MilvusException, match="Received None response from server during search"
            ):
                handler.search(
                    "coll", "vec", {"metric_type": "L2"}, 10, data=[[0.1, 0.2, 0.3, 0.4]]
                )

    def test_hybrid_search_stub_returns_none_raises_milvus_exception(self, handler):
        handler._stub.HybridSearch.return_value = None

        req = AnnSearchRequest(
            data=[[0.1, 0.2, 0.3, 0.4]], anns_field="vec", param={"metric_type": "L2"}, limit=10
        )
        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            with pytest.raises(
                MilvusException, match="Received None response from server during search"
            ):
                handler.hybrid_search("coll", [req], RRFRanker(), 10)

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


class TestSchemaMismatchRetry:
    """Tests for the schema mismatch retry flow during insert/upsert.

    Exercises the full path: insert_rows → server returns SchemaMismatch →
    retry_on_schema_mismatch catches it → schema cache invalidated → retry
    with fresh schema.

    This simulates the scenario from milvus-io/milvus#48522 where concurrent
    add_field changes the collection schema mid-insert.
    """

    @pytest.fixture
    def old_schema(self):
        """Schema before add_field (2 fields)."""
        return {
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
                {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
            ],
            "enable_dynamic_field": False,
            "update_timestamp": 100,
        }

    @pytest.fixture
    def new_schema(self):
        """Schema after add_field (3 fields — new nullable field added)."""
        return {
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
                {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
                {"name": "extra", "type": DataType.INT64, "nullable": True},
            ],
            "enable_dynamic_field": False,
            "update_timestamp": 200,
        }

    @staticmethod
    def _make_schema_mismatch_response():
        """Create a response with SchemaMismatch error status."""
        resp = MagicMock()
        resp.status.code = common_pb2.SchemaMismatch
        resp.status.error_code = common_pb2.SchemaMismatch
        resp.status.reason = "collection schema mismatch"
        return resp

    @staticmethod
    def _make_ok_response(ids=None):
        return make_mutation_response(insert_cnt=len(ids or [1]), ids=ids or [1])

    def test_insert_retries_on_server_schema_mismatch(self, handler, old_schema, new_schema):
        """insert_rows: server returns SchemaMismatch → cache invalidated → retry succeeds.

        Simulates: add_field changes the schema between the first insert attempt
        and the retry. The retry re-fetches the schema and succeeds.
        """
        context = CallContext(db_name="test_db")
        call_count = 0

        def stub_insert_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._make_schema_mismatch_response()
            return self._make_ok_response()

        handler._stub.Insert.side_effect = stub_insert_side_effect

        # First call returns old schema; after invalidation, returns new schema
        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        with patch.object(handler, "_get_schema", side_effect=lambda *a, **kw: next(schema_iter)):
            result = handler.insert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        assert call_count == 2
        assert result.insert_count == 1

    def test_upsert_retries_on_server_schema_mismatch(self, handler, old_schema, new_schema):
        """upsert_rows: server returns SchemaMismatch → cache invalidated → retry succeeds."""
        context = CallContext(db_name="test_db")
        call_count = 0

        def stub_upsert_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._make_schema_mismatch_response()
            return make_mutation_response(upsert_cnt=1, ids=[1])

        handler._stub.Upsert.side_effect = stub_upsert_side_effect

        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        with patch.object(handler, "_get_schema", side_effect=lambda *a, **kw: next(schema_iter)):
            result = handler.upsert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        assert call_count == 2
        assert result.upsert_count == 1

    def test_insert_schema_mismatch_invalidates_cache(self, handler, old_schema, new_schema):
        """Verify _invalidate_schema is called with correct collection and db_name."""
        context = CallContext(db_name="mydb")
        call_count = 0

        def stub_insert(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._make_schema_mismatch_response()
            return self._make_ok_response()

        handler._stub.Insert.side_effect = stub_insert

        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        with patch.object(
            handler, "_get_schema", side_effect=lambda *a, **kw: next(schema_iter)
        ), patch.object(handler, "_invalidate_schema") as mock_invalidate:
            handler.insert_rows(
                "test_coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        mock_invalidate.assert_called_once_with("test_coll", db_name="mydb")

    def test_insert_schema_mismatch_still_fails_after_retry(self, handler, old_schema):
        """If schema mismatch persists after retry, the exception propagates."""
        context = CallContext(db_name="test_db")

        # Server always returns SchemaMismatch
        handler._stub.Insert.return_value = self._make_schema_mismatch_response()

        with patch.object(handler, "_get_schema", return_value=(old_schema, 100)):
            with pytest.raises(SchemaMismatchRetryableException, match="schema mismatch"):
                handler.insert_rows(
                    "coll",
                    [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                    context=context,
                )

        # Insert called twice: original + one retry
        assert handler._stub.Insert.call_count == 2

    def test_insert_client_side_data_mismatch_triggers_retry(self, handler, old_schema, new_schema):
        """DataNotMatchException (client-side validation) also triggers schema retry.

        This happens when cached schema has fewer fields than the data, or
        vice versa — the client detects the mismatch before sending the RPC.
        """
        context = CallContext(db_name="test_db")
        call_count = 0

        def mock_prepare(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise DataNotMatchException(message="field count mismatch")
            # Second call uses fresh schema, builds request successfully
            return MagicMock()

        handler._stub.Insert.return_value = self._make_ok_response()

        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        with patch.object(
            handler, "_get_schema", side_effect=lambda *a, **kw: next(schema_iter)
        ), patch.object(handler, "_prepare_row_insert_request", side_effect=mock_prepare):
            result = handler.insert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        assert call_count == 2
        assert result.insert_count == 1

    def test_insert_schema_mismatch_no_context_raises_param_error(self, handler, old_schema):
        """Schema mismatch without context kwarg raises ParamError (not retried)."""
        handler._stub.Insert.return_value = self._make_schema_mismatch_response()

        with patch.object(handler, "_get_schema", return_value=(old_schema, 100)):
            with pytest.raises(Exception) as exc_info:
                handler.insert_rows(
                    "coll",
                    [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                    # no context= kwarg
                )

        # ParamError from the decorator or SchemaMismatchRetryableException
        # depends on how check_status raises it; the decorator catches it
        # and tries to get context, which is None → ParamError
        assert "context is required" in str(exc_info.value)

    def test_insert_schema_timestamp_updated_on_retry(self, handler, old_schema, new_schema):
        """After schema invalidation, the retried request uses the new schema_timestamp.

        The schema_timestamp is embedded in the insert request and checked by the
        Milvus proxy. A stale timestamp causes SchemaMismatch rejection.
        """
        context = CallContext(db_name="test_db")
        captured_timestamps = []

        original_prepare = handler._prepare_row_insert_request

        def tracking_prepare(*args, **kwargs):
            # Call the real _prepare_row_insert_request but capture schema timestamps
            result = original_prepare(*args, **kwargs)
            # The schema_timestamp is set on the request protobuf
            if hasattr(result, "schema_timestamp"):
                captured_timestamps.append(result.schema_timestamp)
            return result

        call_count = 0

        def stub_insert(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._make_schema_mismatch_response()
            return self._make_ok_response()

        handler._stub.Insert.side_effect = stub_insert

        get_schema_calls = []
        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        def mock_get_schema(*args, **kwargs):
            result = next(schema_iter)
            get_schema_calls.append(result[1])  # capture timestamp
            return result

        with patch.object(handler, "_get_schema", side_effect=mock_get_schema):
            handler.insert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        # Schema fetched twice: first with old ts (100), then with new ts (200)
        assert get_schema_calls == [100, 200]

    def test_upsert_schema_mismatch_still_fails_after_retry(self, handler, old_schema):
        """upsert_rows: persistent SchemaMismatch propagates after retry exhaustion."""
        context = CallContext(db_name="test_db")

        handler._stub.Upsert.return_value = self._make_schema_mismatch_response()

        with patch.object(handler, "_get_schema", return_value=(old_schema, 100)):
            with pytest.raises(SchemaMismatchRetryableException, match="schema mismatch"):
                handler.upsert_rows(
                    "coll",
                    [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                    context=context,
                )

    def test_insert_re_prepares_request_with_fresh_schema_on_retry(
        self, handler, old_schema, new_schema
    ):
        """End-to-end: retry re-prepares the request through _parse_row_request.

        Simulates the milvus-io/milvus#48522 scenario:
        1. Rows have 2 fields (id, vector) — matching old schema
        2. Concurrent add_field adds a 3rd nullable field
        3. Server returns SchemaMismatch on first attempt
        4. retry_on_schema_mismatch invalidates cache, retries
        5. Retry fetches new schema (3 fields), _parse_row_request auto-fills
           None for the missing nullable field
        6. New request carries updated schema_timestamp → server accepts

        This test does NOT mock _prepare_row_insert_request — it exercises
        the real Prepare.row_insert_param → _parse_row_request path to
        prove nullable auto-fill works.
        """
        context = CallContext(db_name="test_db")
        call_count = 0
        captured_requests = []

        def stub_insert_capture(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Capture the request protobuf for inspection
            request = args[0] if args else kwargs.get("request")
            captured_requests.append(request)
            if call_count == 1:
                return self._make_schema_mismatch_response()
            return self._make_ok_response()

        handler._stub.Insert.side_effect = stub_insert_capture

        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        # Only mock _get_schema — let _prepare_row_insert_request and
        # Prepare.row_insert_param run for real
        with patch.object(handler, "_get_schema", side_effect=lambda *a, **kw: next(schema_iter)):
            result = handler.insert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        assert call_count == 2
        assert result.insert_count == 1

        # First request: built with old schema (2 fields), schema_timestamp=100
        req1 = captured_requests[0]
        assert req1.schema_timestamp == 100
        req1_field_names = {fd.field_name for fd in req1.fields_data}
        assert "extra" not in req1_field_names

        # Second request: built with new schema (3 fields), schema_timestamp=200
        # _parse_row_request auto-filled None for the nullable "extra" field
        req2 = captured_requests[1]
        assert req2.schema_timestamp == 200
        req2_field_names = {fd.field_name for fd in req2.fields_data}
        assert "extra" in req2_field_names

    def test_upsert_re_prepares_request_with_fresh_schema_on_retry(
        self, handler, old_schema, new_schema
    ):
        """End-to-end: upsert retry re-prepares with fresh schema, auto-fills nullable."""
        context = CallContext(db_name="test_db")
        call_count = 0
        captured_requests = []

        def stub_upsert_capture(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            request = args[0] if args else kwargs.get("request")
            captured_requests.append(request)
            if call_count == 1:
                return self._make_schema_mismatch_response()
            return make_mutation_response(upsert_cnt=1, ids=[1])

        handler._stub.Upsert.side_effect = stub_upsert_capture

        schema_sequence = [(old_schema, 100), (new_schema, 200)]
        schema_iter = iter(schema_sequence)

        with patch.object(handler, "_get_schema", side_effect=lambda *a, **kw: next(schema_iter)):
            result = handler.upsert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]}],
                context=context,
            )

        assert call_count == 2
        assert result.upsert_count == 1

        req1 = captured_requests[0]
        assert req1.schema_timestamp == 100
        assert "extra" not in {fd.field_name for fd in req1.fields_data}

        req2 = captured_requests[1]
        assert req2.schema_timestamp == 200
        assert "extra" in {fd.field_name for fd in req2.fields_data}
