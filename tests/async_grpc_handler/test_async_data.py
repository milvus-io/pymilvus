"""Tests for AsyncGrpcHandler data operations.

Coverage: Insert, delete, upsert, query, search, flush, compaction.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.grpc_gen import schema_pb2


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
