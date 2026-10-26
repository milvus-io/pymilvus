"""Tests for AsyncGrpcHandler data operations.

Coverage: Insert, delete, upsert, query, search, flush, compaction.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from pymilvus import FieldOp
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.call_context import CallContext
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, milvus_pb2, schema_pb2


class TestAsyncGrpcHandlerDataOps:
    """Tests for data operations (insert, delete, upsert)."""

    @pytest.mark.asyncio
    async def test_insert_rows(self) -> None:
        """Test insert_rows async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()
        handler._prepare_row_insert_request = AsyncMock(return_value=MagicMock())

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.IDs = MagicMock()
        mock_response.IDs.int_id = MagicMock()
        mock_response.IDs.int_id.data = [1]
        mock_response.IDs.str_id = MagicMock()
        mock_response.IDs.str_id.data = []
        mock_response.timestamp = 123456
        mock_response.succ_index = [0]
        mock_response.err_index = []
        mock_response.insert_count = 1
        mock_stub.Insert = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.update_collection_ts"
        ):
            result = await handler.insert_rows("test_coll", [{"id": 1, "vector": [0.1, 0.2]}])
            assert result is not None

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test delete async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.IDs = MagicMock()
        mock_response.IDs.int_id = MagicMock()
        mock_response.IDs.int_id.data = [1]
        mock_response.IDs.str_id = MagicMock()
        mock_response.IDs.str_id.data = []
        mock_response.timestamp = 123456
        mock_response.delete_count = 1
        mock_stub.Delete = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.ts_utils.update_collection_ts"):
            mock_prepare.delete_request.return_value = MagicMock()
            result = await handler.delete("test_coll", "id in [1]")
            assert result is not None

    @pytest.mark.asyncio
    async def test_upsert(self) -> None:
        """Test upsert async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()
        handler._prepare_batch_upsert_request = AsyncMock(return_value=MagicMock())

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.IDs = MagicMock()
        mock_response.IDs.int_id = MagicMock()
        mock_response.IDs.int_id.data = [1]
        mock_response.IDs.str_id = MagicMock()
        mock_response.IDs.str_id.data = []
        mock_response.timestamp = 123456
        mock_response.upsert_count = 1
        mock_response.succ_index = [0]
        mock_response.err_index = []
        mock_stub.Upsert = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch(
            "pymilvus.client.async_grpc_handler.check_invalid_binary_vector", return_value=True
        ), patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.update_collection_ts"
        ):
            result = await handler.upsert("test_coll", [[1, [0.1, 0.2]]])
            assert result is not None

    @pytest.mark.asyncio
    async def test_upsert_rows(self) -> None:
        """Test upsert_rows async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()
        handler._prepare_row_upsert_request = AsyncMock(return_value=MagicMock())

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.IDs = MagicMock()
        mock_response.IDs.int_id = MagicMock()
        mock_response.IDs.int_id.data = [1]
        mock_response.IDs.str_id = MagicMock()
        mock_response.IDs.str_id.data = []
        mock_response.timestamp = 123456
        mock_response.upsert_count = 1
        mock_response.succ_index = [0]
        mock_response.err_index = []
        mock_stub.Upsert = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.update_collection_ts"
        ):
            result = await handler.upsert_rows("test_coll", [{"id": 1, "vector": [0.1, 0.2]}])
            assert result is not None

    @pytest.mark.asyncio
    async def test_get_persistent_segment_infos(self) -> None:
        """Test get_persistent_segment_infos async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.infos = []
        mock_stub.GetPersistentSegmentInfo = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.get_persistent_segment_info_request.return_value = MagicMock()
            result = await handler.get_persistent_segment_infos("test_coll")
            assert result == []


class TestAsyncGrpcHandlerQuery:
    """Tests for query operations."""

    @pytest.mark.asyncio
    async def test_query(self) -> None:
        """Test query async API - verifies query stub is called"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_field = MagicMock()
        mock_field.field_name = "id"
        mock_field.scalars = MagicMock()
        mock_field.scalars.long_data = MagicMock()
        mock_field.scalars.long_data.data = [1, 2, 3]
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.fields_data = [mock_field]
        mock_response.collection_name = "test_coll"
        mock_response.session_ts = 0
        mock_response.element_indices = []
        mock_stub.Query = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ), patch(
            "pymilvus.client.async_grpc_handler.len_of", return_value=3
        ), patch(
            "pymilvus.client.async_grpc_handler.entity_helper.extract_dynamic_field_from_result",
            return_value=([], []),
        ), patch(
            "pymilvus.client.async_grpc_handler.entity_helper.extract_row_data_from_fields_data_v2",
            return_value=None,
        ), patch(
            "pymilvus.client.async_grpc_handler.get_extra_info", return_value={}
        ), patch(
            "pymilvus.client.async_grpc_handler.HybridExtraList"
        ) as mock_hybrid:
            mock_hybrid.return_value = MagicMock()
            mock_prepare.query_request.return_value = MagicMock()
            await handler.query("test_coll", expr="id > 0", output_fields=["id"])
            mock_stub.Query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_element_indices(self) -> None:
        """Test that element_indices are parsed and rows are expanded correctly."""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_field = MagicMock()
        mock_field.field_name = "id"
        mock_field.scalars = MagicMock()
        mock_field.scalars.long_data = MagicMock()
        mock_field.scalars.long_data.data = [1, 2]

        ei_0 = MagicMock()
        ei_0.indices.data = [0, 2]
        ei_1 = MagicMock()
        ei_1.indices.data = [1]

        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.fields_data = [mock_field]
        mock_response.collection_name = "test_coll"
        mock_response.session_ts = 0
        mock_response.element_indices = [ei_0, ei_1]
        mock_stub.Query = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ), patch(
            "pymilvus.client.async_grpc_handler.len_of", return_value=2
        ), patch(
            "pymilvus.client.async_grpc_handler.entity_helper.extract_dynamic_field_from_result",
            return_value=([], []),
        ), patch(
            "pymilvus.client.async_grpc_handler.entity_helper.extract_row_data_from_fields_data_v2",
            return_value=False,
        ), patch(
            "pymilvus.client.async_grpc_handler.get_extra_info", return_value={}
        ):
            mock_prepare.query_request.return_value = MagicMock()
            result = await handler.query("test_coll", expr="id > 0", output_fields=["id"])

        assert len(result) == 3
        assert result[0]["offset"] == 0
        assert result[1]["offset"] == 2
        assert result[2]["offset"] == 1

    @pytest.mark.asyncio
    async def test_query_element_indices_length_mismatch(self) -> None:
        """Test that mismatched element_indices length raises MilvusException."""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_field = MagicMock()
        mock_field.field_name = "id"
        mock_field.scalars = MagicMock()
        mock_field.scalars.long_data = MagicMock()
        mock_field.scalars.long_data.data = [1, 2, 3]

        ei_0 = MagicMock()
        ei_0.indices.data = [0]

        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.fields_data = [mock_field]
        mock_response.collection_name = "test_coll"
        mock_response.session_ts = 0
        mock_response.element_indices = [ei_0]  # length 1 != num_entities 3
        mock_stub.Query = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ), patch(
            "pymilvus.client.async_grpc_handler.len_of", return_value=3
        ), patch(
            "pymilvus.client.async_grpc_handler.entity_helper.extract_dynamic_field_from_result",
            return_value=([], []),
        ), patch(
            "pymilvus.client.async_grpc_handler.entity_helper.extract_row_data_from_fields_data_v2",
            return_value=False,
        ), patch(
            "pymilvus.client.async_grpc_handler.get_extra_info", return_value={}
        ):
            mock_prepare.query_request.return_value = MagicMock()
            with pytest.raises(MilvusException, match="element_indices length"):
                await handler.query("test_coll", expr="id > 0", output_fields=["id"])


class TestAsyncGrpcHandlerSearch:
    """Tests for search operations."""

    @pytest.mark.asyncio
    async def test_search_with_embedding_list(self) -> None:
        """Test search with EmbeddingList data type"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.results = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=10,
            scores=[0.9] * 10,
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(10)))),
            topks=[10],
        )
        mock_response.session_ts = 0
        mock_stub.Search = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        # EmbeddingList takes embeddings as initial arg or via add() method
        embedding_list = EmbeddingList()
        embedding_list.add([0.1, 0.2, 0.3, 0.4])
        embedding_list.add([0.5, 0.6, 0.7, 0.8])

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.construct_guarantee_ts", return_value=False
        ):
            mock_prepare.search_requests_with_expr.return_value = MagicMock()
            await handler.search(
                collection_name="test_collection",
                anns_field="embedding",
                param={"metric_type": "L2"},
                limit=10,
                data=[embedding_list],
            )

            call_kwargs = mock_prepare.search_requests_with_expr.call_args[1]
            assert call_kwargs.get("is_embedding_list") is True

    @pytest.mark.asyncio
    async def test_hybrid_search_with_embedding_list(self) -> None:
        """Test hybrid_search with EmbeddingList data type"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.results = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=10,
            scores=[0.9] * 10,
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(10)))),
            topks=[10],
        )
        mock_stub.HybridSearch = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        # EmbeddingList takes embeddings as initial arg or via add() method
        embedding_list = EmbeddingList()
        embedding_list.add([0.1, 0.2, 0.3, 0.4])

        mock_ranker = MagicMock()
        ann_req = AnnSearchRequest(
            data=[embedding_list],
            anns_field="embedding",
            param={"metric_type": "L2"},
            limit=10,
        )

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.construct_guarantee_ts", return_value=False
        ):
            mock_prepare.search_requests_with_expr.return_value = MagicMock()
            mock_prepare.hybrid_search_request_with_ranker.return_value = MagicMock()

            await handler.hybrid_search(
                collection_name="test_collection",
                reqs=[ann_req],
                rerank=mock_ranker,
                limit=10,
            )


class TestAsyncGrpcHandlerFlush:
    """Tests for flush operations."""

    @pytest.mark.asyncio
    async def test_flush(self) -> None:
        """Test flush async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_seg_ids = MagicMock()
        mock_seg_ids.data = [1, 2]
        mock_response.coll_segIDs = {"test_coll": mock_seg_ids}
        mock_response.coll_flush_ts = {"test_coll": 123456}
        mock_stub.Flush = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        handler._wait_for_flushed = AsyncMock()

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_pass_param"
        ), patch("pymilvus.client.async_grpc_handler.check_status"):
            mock_prepare.flush_param.return_value = MagicMock()
            await handler.flush(["test_coll"])
            mock_stub.Flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_flush_state(self) -> None:
        """Test get_flush_state async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.flushed = True
        mock_stub.GetFlushState = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.get_flush_state_request.return_value = MagicMock()
            result = await handler.get_flush_state([1, 2, 3], "test_coll", 123456)
            assert result is True

    @pytest.mark.asyncio
    async def test_alloc_timestamp(self) -> None:
        """Test alloc_timestamp async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.timestamp = 123456789
        mock_stub.AllocTimestamp = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.alloc_timestamp()
            assert result == 123456789


class TestAsyncGrpcHandlerCompaction:
    """Tests for compaction operations."""

    @pytest.mark.asyncio
    async def test_compact(self) -> None:
        """Test compact async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_desc_resp = MagicMock()
        mock_desc_resp.status.code = 0
        mock_desc_resp.collectionID = 12345
        mock_stub.DescribeCollection = AsyncMock(return_value=mock_desc_resp)
        mock_comp_resp = MagicMock()
        mock_comp_resp.status.code = 0
        mock_comp_resp.compactionID = 67890
        mock_stub.ManualCompaction = AsyncMock(return_value=mock_comp_resp)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.describe_collection_request.return_value = MagicMock()
            mock_prepare.manual_compaction.return_value = MagicMock()
            result = await handler.compact("test_coll")
            assert result == 67890

    @pytest.mark.asyncio
    async def test_get_compaction_state(self) -> None:
        """Test get_compaction_state async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.status.error_code = 0
        mock_response.status.reason = ""
        mock_response.state = 2
        mock_stub.GetCompactionState = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"):
            result = await handler.get_compaction_state(12345, timeout=30)
            assert result is not None

    @pytest.mark.asyncio
    async def test_upsert_rows_forwards_field_ops_and_auto_promotes(self) -> None:
        """Async counterpart: FieldPartialUpdateOp threads into the outgoing
        request and auto-promotes partial_update=True."""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        fields_info = [
            {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
            {
                "name": "tags",
                "type": DataType.ARRAY,
                "element_type": DataType.INT64,
                "params": {"max_capacity": 16},
            },
        ]
        handler._get_schema = AsyncMock(
            return_value=(
                {
                    "fields": fields_info,
                    "struct_array_fields": [],
                    "enable_dynamic_field": False,
                },
                0,
            )
        )

        captured = []

        async def capture(req, **_):
            captured.append(req)
            resp = MagicMock()
            resp.status.code = 0
            resp.IDs.int_id.data = [1]
            resp.IDs.str_id.data = []
            resp.timestamp = 1
            resp.upsert_count = 1
            resp.succ_index = [0]
            resp.err_index = []
            return resp

        mock_stub = AsyncMock()
        mock_stub.Upsert = AsyncMock(side_effect=capture)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.check_status"), patch(
            "pymilvus.client.async_grpc_handler.ts_utils.update_collection_ts"
        ):
            await handler.upsert_rows(
                "coll",
                [{"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "tags": [7]}],
                field_ops={"tags": FieldOp.array_append()},
            )

        assert len(captured) == 1
        req = captured[0]
        assert len(req.field_ops) == 1
        assert req.field_ops[0].field_name == "tags"
        assert req.partial_update is True

    @pytest.mark.asyncio
    async def test_run_analyzer(self) -> None:
        """Test run_analyzer async API"""
        mock_channel = MagicMock()
        mock_channel._unary_unary_interceptors = []
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True
        handler.ensure_channel_ready = AsyncMock()

        mock_stub = AsyncMock()
        mock_response = MagicMock()
        mock_response.status.code = 0
        mock_response.token_results = []
        mock_stub.RunAnalyzer = AsyncMock(return_value=mock_response)
        handler._async_stub = mock_stub

        with patch("pymilvus.client.async_grpc_handler.Prepare") as mock_prepare, patch(
            "pymilvus.client.async_grpc_handler.check_status"
        ):
            mock_prepare.run_analyzer_request.return_value = MagicMock()
            result = await handler.run_analyzer("hello world")
            assert result is not None


@pytest.mark.asyncio
async def test_conditional_async_transport_failure_is_not_replayed():

    class TransportError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "response lost"

    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel)
    handler._is_channel_ready = True
    handler.ensure_channel_ready = AsyncMock()
    handler._prepare_row_upsert_request = AsyncMock(
        return_value=milvus_pb2.UpsertRequest(
            write_condition=milvus_pb2.WriteCondition(field="version", op=1)
        )
    )
    handler._async_stub = MagicMock()
    handler._async_stub.Upsert = AsyncMock(side_effect=TransportError())
    with pytest.raises(MilvusException) as exc:
        await handler.upsert_rows("coll", [], write_condition={"field": "version", "op": "gt"})
    assert exc.value.code == 112
    assert handler._async_stub.Upsert.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("code,expected_calls", [(8, 2), (109, 2), (5, 1), (112, 1), (111, 1)])
async def test_partial_async_upsert_server_retry_contract(code, expected_calls, conditional):
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel)
    handler._is_channel_ready = True
    handler.ensure_channel_ready = AsyncMock()
    handler._prepare_row_upsert_request = AsyncMock(
        return_value=milvus_pb2.UpsertRequest(
            write_condition=milvus_pb2.WriteCondition(field="version", op=1)
        )
    )
    if not conditional:
        handler._prepare_row_upsert_request.return_value.ClearField("write_condition")
    handler._invalidate_schema = MagicMock()
    failed = milvus_pb2.MutationResult(
        status=common_pb2.Status(
            code=code,
            error_code=common_pb2.SchemaMismatch if code == 109 else common_pb2.UnexpectedError,
            reason="STREAMING_CODE_REPLICATE_VIOLATION: SchemaMismatch RateLimit CAS diagnostic cause",
        )
    )
    succeeded = milvus_pb2.MutationResult(
        status=common_pb2.Status(),
        upsert_cnt=1,
        IDs=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[42])),
        succ_index=[0],
    )
    handler._async_stub = MagicMock()
    handler._async_stub.Upsert = AsyncMock(side_effect=[failed, succeeded])
    handler._on_rpc_error = AsyncMock(return_value=True)
    if expected_calls == 1:
        with pytest.raises(MilvusException) as exc:
            await handler.upsert_rows(
                "coll",
                [],
                context=CallContext(db_name="test_db"),
                write_condition={"field": "version", "op": "gt"} if conditional else None,
            )
        assert exc.value.code == code
        assert exc.value.retriable is False
    else:
        result = await handler.upsert_rows(
            "coll",
            [],
            context=CallContext(db_name="test_db"),
            write_condition={"field": "version", "op": "gt"} if conditional else None,
        )
        assert result.succ_index == [0]
    assert handler._async_stub.Upsert.await_count == expected_calls

    handler._on_rpc_error.assert_not_called()


@pytest.fixture
def async_upsert_timeout_handler():
    channel = MagicMock()
    channel._unary_unary_interceptors = []
    handler = AsyncGrpcHandler(channel=channel)
    handler._is_channel_ready = True
    handler.ensure_channel_ready = AsyncMock()
    handler._prepare_row_upsert_request = AsyncMock(
        return_value=milvus_pb2.UpsertRequest(
            write_condition=milvus_pb2.WriteCondition(field="version", op=1)
        )
    )
    handler._async_stub = MagicMock()
    handler._async_stub.Upsert = AsyncMock()
    handler._on_rpc_error = AsyncMock(return_value=True)
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preparation", "write"])
@pytest.mark.parametrize("conditional", [False, True])
async def test_async_upsert_overall_timeout_classifies_write_phase(
    async_upsert_timeout_handler, phase, conditional
):
    handler = async_upsert_timeout_handler
    if not conditional:
        handler._prepare_row_upsert_request.return_value.ClearField("write_condition")

    async def block(*_args, **_kwargs):
        await asyncio.Event().wait()

    if phase == "preparation":
        handler._prepare_row_upsert_request.side_effect = block
    else:
        handler._async_stub.Upsert.side_effect = block

    with pytest.raises(MilvusException) as exc:
        await handler.upsert_rows(
            "coll",
            [],
            timeout=0.02,
            write_condition={"field": "version", "op": "gt"} if conditional else None,
        )
    if conditional and phase == "write":
        assert exc.value.code == 112
        assert "outcome unknown" in exc.value.message
        assert exc.value.retriable is False
    else:
        assert exc.value.code == 1
        assert "Retry timeout" in exc.value.message
    assert handler._prepare_row_upsert_request.await_count == 1
    assert handler._async_stub.Upsert.await_count == (phase == "write")
    handler._on_rpc_error.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preparation", "write"])
@pytest.mark.parametrize("timeout", [None, 10])
async def test_conditional_async_upsert_preserves_explicit_task_cancellation(
    async_upsert_timeout_handler, phase, timeout
):
    handler = async_upsert_timeout_handler
    started = asyncio.Event()

    async def block(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    if phase == "preparation":
        handler._prepare_row_upsert_request.side_effect = block
    else:
        handler._async_stub.Upsert.side_effect = block

    task = asyncio.create_task(
        handler.upsert_rows(
            "coll", [], timeout=timeout, write_condition={"field": "version", "op": "gt"}
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert handler._async_stub.Upsert.await_count == (phase == "write")
        handler._on_rpc_error.assert_not_called()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [8, 109])
async def test_conditional_async_admission_response_clears_pending_write(
    async_upsert_timeout_handler, code
):
    handler = async_upsert_timeout_handler
    request = handler._prepare_row_upsert_request.return_value

    async def prepare(*_args, **_kwargs):
        if handler._prepare_row_upsert_request.await_count > 1:
            await asyncio.Event().wait()
        return request

    handler._prepare_row_upsert_request.side_effect = prepare
    handler._invalidate_schema = MagicMock()
    handler._async_stub.Upsert.return_value = milvus_pb2.MutationResult(
        status=common_pb2.Status(
            code=code,
            error_code=common_pb2.SchemaMismatch if code == 109 else common_pb2.UnexpectedError,
        )
    )
    with pytest.raises(MilvusException) as exc:
        await handler.upsert_rows(
            "coll",
            [],
            timeout=0.05,
            context=CallContext(db_name="test_db"),
            write_condition={"field": "version", "op": "gt"},
        )
    assert exc.value.code == 1
    assert "Retry timeout" in exc.value.message
    assert handler._prepare_row_upsert_request.await_count == 2
    handler._async_stub.Upsert.assert_awaited_once()
    handler._on_rpc_error.assert_not_called()


@pytest.mark.asyncio
async def test_conditional_async_timeout_state_is_isolated_between_calls(
    async_upsert_timeout_handler,
):
    handler = async_upsert_timeout_handler
    write_started = asyncio.Event()
    request = handler._prepare_row_upsert_request.return_value

    async def prepare(collection_name, *_args, **_kwargs):
        if collection_name == "preparing":
            await asyncio.Event().wait()
        return request

    async def write(*_args, **_kwargs):
        write_started.set()
        await asyncio.Event().wait()

    handler._prepare_row_upsert_request.side_effect = prepare
    handler._async_stub.Upsert.side_effect = write
    writing = asyncio.create_task(
        handler.upsert_rows(
            "writing", [], timeout=1, write_condition={"field": "version", "op": "gt"}
        )
    )
    try:
        await asyncio.wait_for(write_started.wait(), timeout=1)
        with pytest.raises(MilvusException) as exc:
            await handler.upsert_rows(
                "preparing", [], timeout=0.02, write_condition={"field": "version", "op": "gt"}
            )
        assert exc.value.code == 1
        with pytest.raises(MilvusException) as exc:
            await writing
        assert exc.value.code == 112
        handler._async_stub.Upsert.assert_awaited_once()
    finally:
        if not writing.done():
            writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)
