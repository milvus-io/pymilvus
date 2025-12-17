from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import schema_pb2


class TestAsyncGrpcHandler:
    """Test cases for AsyncGrpcHandler class"""

    @pytest.mark.asyncio
    async def test_load_partitions_refresh_attribute(self) -> None:
        """
        Test that load_partitions correctly accesses request.refresh instead of request.is_refresh.
        This test verifies the fix for issue #2796.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock the async stub
        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create a mock response for LoadPartitions
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        # Mock wait_for_loading_partitions to avoid actual waiting
        handler.wait_for_loading_partitions = AsyncMock()

        # Mock Prepare.load_partitions to return a request with refresh attribute
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            # Create mock request with refresh attribute (not is_refresh)
            mock_request = MagicMock()
            mock_request.refresh = True  # This is the correct attribute name
            mock_prepare.load_partitions.return_value = mock_request

            # Call load_partitions
            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                replica_number=1,
                timeout=30,
                refresh=True
            )

            # Verify that Prepare.load_partitions was called correctly
            mock_prepare.load_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                replica_number=1,
                refresh=True
            )

            # Verify that wait_for_loading_partitions was called with is_refresh parameter
            # correctly set from request.refresh (not request.is_refresh)
            handler.wait_for_loading_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1", "partition2"],
                is_refresh=True,  # Should be the value from request.refresh
                timeout=30,
                refresh=True
            )

    @pytest.mark.asyncio
    async def test_load_partitions_without_refresh(self) -> None:
        """Test load_partitions when refresh parameter is not provided"""
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock the async stub
        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create a mock response for LoadPartitions
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        # Mock wait_for_loading_partitions
        handler.wait_for_loading_partitions = AsyncMock()

        # Mock Prepare.load_partitions
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            # Create mock request with default refresh value
            mock_request = MagicMock()
            mock_request.refresh = False  # Default value when not specified
            mock_prepare.load_partitions.return_value = mock_request

            # Call load_partitions without refresh parameter
            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                timeout=30
            )

            # Verify that wait_for_loading_partitions was called with is_refresh=False
            handler.wait_for_loading_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1"],
                is_refresh=False,  # Should be False when not specified
                timeout=30
            )

    @pytest.mark.asyncio
    async def test_wait_for_loading_partitions(self) -> None:
        """Test wait_for_loading_partitions method"""
        # Setup mock channel
        mock_channel = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock get_loading_progress to return 100 immediately
        handler.get_loading_progress = AsyncMock(return_value=100)

        # Call wait_for_loading_partitions
        await handler.wait_for_loading_partitions(
            collection_name="test_collection",
            partition_names=["partition1", "partition2"],
            is_refresh=True,
            timeout=30
        )

        # Verify that get_loading_progress was called
        handler.get_loading_progress.assert_called_once_with(
            "test_collection",
            ["partition1", "partition2"],
            timeout=30,
            is_refresh=True
        )

    @pytest.mark.asyncio
    async def test_wait_for_loading_partitions_timeout(self) -> None:
        """Test that wait_for_loading_partitions raises exception on timeout"""

        # Setup mock channel
        mock_channel = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock get_loading_progress to always return less than 100
        handler.get_loading_progress = AsyncMock(return_value=50)

        # Call wait_for_loading_partitions with very short timeout
        with pytest.raises(MilvusException) as exc_info:
            await handler.wait_for_loading_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                is_refresh=False,
                timeout=0.001  # Very short timeout to trigger timeout error
            )

        assert "wait for loading partition timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_partitions_with_resource_groups(self) -> None:
        """Test load_partitions with additional parameters like resource_groups"""
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel.close = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        # Create handler with mocked channel
        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        # Mock the async stub
        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Create a mock response for LoadPartitions
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status
        mock_stub.LoadPartitions = AsyncMock(return_value=mock_response)

        # Mock wait_for_loading_partitions
        handler.wait_for_loading_partitions = AsyncMock()

        # Mock Prepare.load_partitions
        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            # Create mock request
            mock_request = MagicMock()
            mock_request.refresh = False
            mock_prepare.load_partitions.return_value = mock_request

            # Call load_partitions with resource_groups
            await handler.load_partitions(
                collection_name="test_collection",
                partition_names=["partition1"],
                replica_number=2,
                resource_groups=["rg1", "rg2"],
                timeout=30
            )

            # Verify that Prepare.load_partitions was called with resource_groups
            mock_prepare.load_partitions.assert_called_once_with(
                collection_name="test_collection",
                partition_names=["partition1"],
                replica_number=2,
                resource_groups=["rg1", "rg2"]
            )

    @pytest.mark.asyncio
    async def test_create_index_with_nested_field(self) -> None:
        """
        Test that create_index works with nested field names (e.g., "chunks[text_vector]").
        This test verifies the fix for issue where AsyncMilvusClient.create_index 
        failed for nested fields in Array of Struct.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Mock wait_for_creating_index to return success
        handler.wait_for_creating_index = AsyncMock(return_value=(True, ""))
        handler.alloc_timestamp = AsyncMock(return_value=12345)

        # Mock CreateIndex response
        mock_create_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.reason = ""
        mock_create_response.status = mock_status
        mock_stub.CreateIndex = AsyncMock(return_value=mock_create_response)

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            
            # Create mock index request
            mock_index_request = MagicMock()
            mock_prepare.create_index_request.return_value = mock_index_request

            # Call create_index with a nested field name (Array of Struct field path)
            nested_field_name = "chunks[text_vector]"
            index_params = {
                "metric_type": "MAX_SIM_COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            
            await handler.create_index(
                collection_name="test_collection",
                field_name=nested_field_name,
                params=index_params,
                index_name="test_index"
            )

            # Verify that Prepare.create_index_request was called with the nested field name
            mock_prepare.create_index_request.assert_called_once_with(
                "test_collection",
                nested_field_name,
                index_params,
                index_name="test_index"
            )

            # Verify that CreateIndex was called on the stub
            # The key point is that no MilvusException was raised before this call
            # (which would have happened with the old client-side validation)
            mock_stub.CreateIndex.assert_called_once()

            # Verify wait_for_creating_index was called
            handler.wait_for_creating_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_embedding_list(self) -> None:
        """
        Test that search works with EmbeddingList input data.
        This test verifies the fix for issue where AsyncMilvusClient.search 
        failed when using EmbeddingList for array-of-vector searches.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Mock Search response with proper SearchResultData structure
        mock_search_result_data = schema_pb2.SearchResultData(
            num_queries=2,
            top_k=0,
            scores=[],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[])),
            topks=[],
            primary_field_name="id"
        )
        mock_search_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.reason = ""
        mock_search_response.status = mock_status
        mock_search_response.results = mock_search_result_data
        mock_search_response.session_ts = 0
        mock_stub.Search = AsyncMock(return_value=mock_search_response)

        # Create EmbeddingList data
        from pymilvus.client.embedding_list import EmbeddingList
        emb_list1 = EmbeddingList()
        emb_list1.add([0.1, 0.2, 0.3, 0.4, 0.5])
        emb_list2 = EmbeddingList()
        emb_list2.add([0.5, 0.4, 0.3, 0.2, 0.1])
        data = [emb_list1, emb_list2]

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            
            # Mock search_requests_with_expr to return a request
            mock_request = MagicMock()
            mock_prepare.search_requests_with_expr.return_value = mock_request

            await handler.search(
                collection_name="test_collection",
                data=data,
                anns_field="vector",
                param={"metric_type": "COSINE"},
                limit=10
            )

            # Verify that Prepare.search_requests_with_expr was called
            mock_prepare.search_requests_with_expr.assert_called_once()
            call_args = mock_prepare.search_requests_with_expr.call_args
            
            # Verify that is_embedding_list was passed as True in kwargs
            assert call_args.kwargs.get("is_embedding_list") is True
            
            # Verify data was converted (not EmbeddingList objects anymore)
            passed_data = call_args.kwargs.get("data")
            assert isinstance(passed_data, list)
            assert not isinstance(passed_data[0], EmbeddingList)
            # The data should be converted to flat arrays
            assert isinstance(passed_data[0], (list, np.ndarray))

            # Verify Search was called
            mock_stub.Search.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_with_embedding_list(self) -> None:
        """
        Test that hybrid_search works with EmbeddingList input data.
        """
        # Setup mock channel and stub
        mock_channel = AsyncMock()
        mock_channel.channel_ready = AsyncMock()
        mock_channel._unary_unary_interceptors = []

        handler = AsyncGrpcHandler(channel=mock_channel)
        handler._is_channel_ready = True

        mock_stub = AsyncMock()
        handler._async_stub = mock_stub

        # Mock HybridSearch response with proper SearchResultData structure
        mock_hybrid_result_data = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=0,
            scores=[],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[])),
            topks=[],
            primary_field_name="id"
        )
        mock_hybrid_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.reason = ""
        mock_hybrid_response.status = mock_status
        mock_hybrid_response.results = mock_hybrid_result_data
        mock_stub.HybridSearch = AsyncMock(return_value=mock_hybrid_response)

        # Create AnnSearchRequest with EmbeddingList
        from pymilvus.client.embedding_list import EmbeddingList
        from pymilvus.client.abstract import AnnSearchRequest
        import numpy as np
        
        emb_list = EmbeddingList()
        emb_list.add([0.1, 0.2, 0.3])
        req = AnnSearchRequest(
            data=[emb_list],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=10
        )

        with patch('pymilvus.client.async_grpc_handler.Prepare') as mock_prepare, \
             patch('pymilvus.client.async_grpc_handler.check_pass_param'), \
             patch('pymilvus.client.async_grpc_handler.check_status'), \
             patch('pymilvus.client.async_grpc_handler._api_level_md', return_value={}):
            
            # Mock search_requests_with_expr and hybrid_search_request_with_ranker
            mock_search_request = MagicMock()
            mock_hybrid_request = MagicMock()
            mock_prepare.search_requests_with_expr.return_value = mock_search_request
            mock_prepare.hybrid_search_request_with_ranker.return_value = mock_hybrid_request

            # Mock rerank (BaseRanker)
            mock_ranker = MagicMock()

            await handler.hybrid_search(
                collection_name="test_collection",
                reqs=[req],
                rerank=mock_ranker,
                limit=10
            )

            # Verify that search_requests_with_expr was called with converted data
            mock_prepare.search_requests_with_expr.assert_called_once()
            call_args = mock_prepare.search_requests_with_expr.call_args
            
            # Verify is_embedding_list flag was set
            assert call_args.kwargs.get("is_embedding_list") is True
            
            # Verify data was converted
            passed_data = call_args.kwargs.get("data")
            assert isinstance(passed_data, list)
            assert not isinstance(passed_data[0], EmbeddingList)

            # Verify HybridSearch was called
            mock_stub.HybridSearch.assert_called_once()
