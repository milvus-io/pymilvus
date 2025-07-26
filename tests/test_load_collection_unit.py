"""
Unit tests for load_collection partition loading fix.
These tests focus on the logic without requiring grpc_testing dependencies.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.types import LoadState
from pymilvus.exceptions import MilvusException


class TestSyncLoadCollectionLogic:
    """Test the synchronous load_collection logic changes"""

    def test_wait_for_loading_collection_checks_partition_states_when_progress_100(self):
        """Test that wait_for_loading_collection checks partition states when progress is 100%"""
        handler = GrpcHandler()
        
        def mock_get_loading_progress(*args, **kwargs):
            return 100
            
        # First call returns Loading for partA, then all return Loaded to allow completion
        call_count = 0
        def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if partition_names and 'partA' in partition_names and call_count == 1:
                return LoadState.Loading  # First check: partA is loading
            return LoadState.Loaded  # Subsequent checks: all loaded
        
        with patch.object(handler, 'get_loading_progress', side_effect=mock_get_loading_progress):
            with patch.object(handler, 'get_load_state', side_effect=mock_get_load_state) as mock_load_state:
                # Should eventually succeed when partA becomes loaded
                handler.wait_for_loading_collection(
                    collection_name="test_collection",
                    partition_names=['partA', 'partB', 'partC'],
                    timeout=5
                )
                
                # Verify that get_load_state was called
                assert mock_load_state.call_count >= 3

    def test_wait_for_loading_collection_returns_when_all_partitions_loaded(self):
        """Test that wait_for_loading_collection returns successfully when all partitions are loaded"""
        handler = GrpcHandler()
        
        def mock_get_loading_progress(*args, **kwargs):
            return 100  # Collection progress is always 100%
            
        def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            return LoadState.Loaded  # All partitions are loaded
        
        with patch.object(handler, 'get_loading_progress', side_effect=mock_get_loading_progress):
            with patch.object(handler, 'get_load_state', side_effect=mock_get_load_state) as mock_load_state:
                # Should return successfully
                handler.wait_for_loading_collection(
                    collection_name="test_collection",
                    partition_names=['partA', 'partB', 'partC'],
                    timeout=5
                )
                
                # Verify that get_load_state was called for each partition
                assert mock_load_state.call_count == 3

    def test_wait_for_loading_collection_backward_compatibility(self):
        """Test backward compatibility when partition_names is None"""
        handler = GrpcHandler()
        
        def mock_get_loading_progress(*args, **kwargs):
            return 100  # Collection progress is always 100%
        
        with patch.object(handler, 'get_loading_progress', side_effect=mock_get_loading_progress):
            with patch.object(handler, 'get_load_state') as mock_load_state:
                # Should return successfully using original behavior
                handler.wait_for_loading_collection(
                    collection_name="test_collection",
                    partition_names=None,  # No partition checking
                    timeout=5
                )
                
                # get_load_state should not be called when partition_names is None
                mock_load_state.assert_not_called()

    def test_wait_for_loading_collection_with_less_than_100_progress(self):
        """Test that the method waits when collection progress is less than 100%"""
        handler = GrpcHandler()
        
        call_count = 0
        def mock_get_loading_progress(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return 99 if call_count < 3 else 100  # Progress increases to 100% after 3 calls
            
        def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            return LoadState.Loaded  # All partitions are loaded when checked
        
        with patch.object(handler, 'get_loading_progress', side_effect=mock_get_loading_progress) as mock_progress:
            with patch.object(handler, 'get_load_state', side_effect=mock_get_load_state) as mock_load_state:
                handler.wait_for_loading_collection(
                    collection_name="test_collection",
                    partition_names=['partA', 'partB', 'partC'],
                    timeout=5
                )
                
                # Should have called get_loading_progress multiple times
                assert mock_progress.call_count >= 3
                # Should have called get_load_state only when progress reached 100%
                assert mock_load_state.call_count == 3

    def test_timeout_when_partition_still_loading(self):
        """Test timeout when a partition remains in loading state"""
        handler = GrpcHandler()
        
        def mock_get_loading_progress(*args, **kwargs):
            return 100
            
        def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            # Always return loading for partA to force timeout
            if partition_names and 'partA' in partition_names:
                return LoadState.Loading
            return LoadState.Loaded
        
        # Mock time to advance quickly
        start_time = 0
        def mock_time():
            nonlocal start_time
            start_time += 1
            return start_time
        
        with patch('pymilvus.client.grpc_handler.time.time', side_effect=mock_time):
            with patch('pymilvus.client.grpc_handler.time.sleep'):
                with patch.object(handler, 'get_loading_progress', side_effect=mock_get_loading_progress):
                    with patch.object(handler, 'get_load_state', side_effect=mock_get_load_state):
                        with pytest.raises(MilvusException, match="wait for loading collection timeout"):
                            handler.wait_for_loading_collection(
                                collection_name="test_collection",
                                partition_names=['partA', 'partB', 'partC'],
                                timeout=1  # 1 second timeout
                            )


class TestAsyncLoadCollectionLogic:
    """Test the asynchronous load_collection logic changes"""

    @pytest.mark.asyncio
    async def test_async_wait_for_loading_collection_checks_partition_states(self):
        """Test that async wait_for_loading_collection checks partition states"""
        handler = AsyncGrpcHandler()
        
        async def mock_get_loading_progress(*args, **kwargs):
            return 100
            
        # First call returns Loading, then Loaded to allow completion
        call_count = 0
        async def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if partition_names and 'partA' in partition_names and call_count == 1:
                return LoadState.Loading
            return LoadState.Loaded
        
        handler.get_loading_progress = mock_get_loading_progress
        handler.get_load_state = AsyncMock(side_effect=mock_get_load_state)
        
        # Should complete successfully
        await handler.wait_for_loading_collection(
            collection_name="test_collection",
            partition_names=['partA', 'partB', 'partC'],
            timeout=5
        )
        
        # Verify that get_load_state was called
        assert handler.get_load_state.call_count >= 3

    @pytest.mark.asyncio
    async def test_async_wait_for_loading_collection_returns_when_all_loaded(self):
        """Test that async wait_for_loading_collection returns when all partitions are loaded"""
        handler = AsyncGrpcHandler()
        
        async def mock_get_loading_progress(*args, **kwargs):
            return 100
            
        async def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            return LoadState.Loaded
        
        handler.get_loading_progress = mock_get_loading_progress
        handler.get_load_state = AsyncMock(side_effect=mock_get_load_state)
        
        # Should return successfully
        await handler.wait_for_loading_collection(
            collection_name="test_collection",
            partition_names=['partA', 'partB', 'partC'],
            timeout=5
        )
        
        # Verify calls were made
        assert handler.get_load_state.call_count == 3

    @pytest.mark.asyncio
    async def test_async_wait_for_loading_collection_backward_compatibility(self):
        """Test async backward compatibility when partition_names is None"""
        handler = AsyncGrpcHandler()
        
        async def mock_get_loading_progress(*args, **kwargs):
            return 100
        
        handler.get_loading_progress = mock_get_loading_progress
        handler.get_load_state = AsyncMock()
        
        # Should return successfully using original behavior
        await handler.wait_for_loading_collection(
            collection_name="test_collection",
            partition_names=None,
            timeout=5
        )
        
        # get_load_state should not be called
        handler.get_load_state.assert_not_called()

    @pytest.mark.asyncio 
    async def test_async_timeout_when_partition_still_loading(self):
        """Test async timeout when partition remains loading"""
        handler = AsyncGrpcHandler()
        
        async def mock_get_loading_progress(*args, **kwargs):
            return 100
            
        async def mock_get_load_state(collection_name, partition_names=None, **kwargs):
            # Always return loading for partA
            if partition_names and 'partA' in partition_names:
                return LoadState.Loading
            return LoadState.Loaded
        
        # Mock advancing time
        start_time = 0
        def mock_time():
            nonlocal start_time
            start_time += 2  # Advance by 2 seconds each call
            return start_time
        
        with patch('pymilvus.client.async_grpc_handler.time.time', side_effect=mock_time):
            with patch('pymilvus.client.async_grpc_handler.asyncio.sleep', new_callable=AsyncMock):
                handler.get_loading_progress = mock_get_loading_progress
                handler.get_load_state = AsyncMock(side_effect=mock_get_load_state)
                
                with pytest.raises(MilvusException, match="wait for loading collection timeout"):
                    await handler.wait_for_loading_collection(
                        collection_name="test_collection",
                        partition_names=['partA', 'partB', 'partC'],
                        timeout=1  # 1 second timeout
                    )


class TestLoadCollectionIntegration:
    """Integration-style tests for the load_collection method changes"""

    def test_load_collection_calls_list_partitions_before_load_call(self):
        """Test that load_collection calls list_partitions before making the LoadCollection RPC"""
        handler = GrpcHandler()
        
        # Track the order of calls
        call_order = []
        
        def mock_list_partitions(*args, **kwargs):
            call_order.append('list_partitions')
            return ['partA', 'partB', 'partC']
        
        def mock_load_collection_rpc(*args, **kwargs):
            call_order.append('LoadCollection')
            # Mock successful response - LoadCollection returns Status directly
            from pymilvus.grpc_gen import common_pb2
            return common_pb2.Status(code=0, reason="")
        
        def mock_wait_for_loading_collection(*args, **kwargs):
            call_order.append('wait_for_loading_collection')
        
        with patch.object(handler, 'list_partitions', side_effect=mock_list_partitions):
            with patch.object(handler, '_stub') as mock_stub:
                mock_stub.LoadCollection = mock_load_collection_rpc
                with patch.object(handler, 'wait_for_loading_collection', side_effect=mock_wait_for_loading_collection):
                    handler.load_collection("test_collection")
        
        # Verify the order of calls
        assert call_order == ['list_partitions', 'LoadCollection', 'wait_for_loading_collection']

    def test_load_collection_passes_partition_names_to_wait_method(self):
        """Test that load_collection passes the retrieved partition_names to wait_for_loading_collection"""
        handler = GrpcHandler()
        
        expected_partitions = ['partA', 'partB', 'partC']
        
        with patch.object(handler, 'list_partitions', return_value=expected_partitions):
            with patch.object(handler, '_stub') as mock_stub:
                from pymilvus.grpc_gen import common_pb2
                mock_stub.LoadCollection.return_value = common_pb2.Status(code=0, reason="")
                with patch.object(handler, 'wait_for_loading_collection') as mock_wait:
                    handler.load_collection("test_collection")
                    
                    # Verify wait_for_loading_collection was called with correct partition_names
                    mock_wait.assert_called_once()
                    args, kwargs = mock_wait.call_args
                    assert kwargs['collection_name'] == "test_collection"
                    assert kwargs['partition_names'] == expected_partitions

    @pytest.mark.asyncio
    async def test_async_load_collection_calls_list_partitions_before_load(self):
        """Test that async load_collection calls list_partitions before LoadCollection"""
        handler = AsyncGrpcHandler()
        
        # Track call order
        call_order = []
        
        async def mock_ensure_channel_ready():
            call_order.append('ensure_channel_ready')
        
        async def mock_list_partitions(*args, **kwargs):
            call_order.append('list_partitions')
            return ['partA', 'partB', 'partC']
        
        async def mock_load_collection_rpc(*args, **kwargs):
            call_order.append('LoadCollection')
            from pymilvus.grpc_gen import common_pb2
            return common_pb2.Status(code=0, reason="")
        
        async def mock_wait_for_loading_collection(*args, **kwargs):
            call_order.append('wait_for_loading_collection')
        
        handler.ensure_channel_ready = mock_ensure_channel_ready
        handler.list_partitions = mock_list_partitions
        handler._async_stub = MagicMock()
        handler._async_stub.LoadCollection = mock_load_collection_rpc
        handler.wait_for_loading_collection = mock_wait_for_loading_collection
        
        await handler.load_collection("test_collection")
        
        # Verify the order of calls
        expected_order = ['ensure_channel_ready', 'list_partitions', 'LoadCollection', 'wait_for_loading_collection']
        assert call_order == expected_order