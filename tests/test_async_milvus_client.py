"""Test cases for AsyncMilvusClient new features"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import AsyncMilvusClient
from pymilvus.client.types import LoadState
from pymilvus.orm.collection import Function, FunctionScore
from pymilvus.orm.schema import StructFieldSchema


class TestAsyncMilvusClientNewFeatures:
    """Test cases for newly added features in AsyncMilvusClient"""

    def test_get_server_type(self):
        """Test get_server_type returns correct value"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = MagicMock()
            mock_handler.get_server_type.return_value = "milvus"
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            # get_server_type is called during __init__ for is_self_hosted, and again here
            assert mock_handler.get_server_type.call_count >= 1
            result = client.get_server_type()
            assert result == "milvus"
            # Verify it was called at least once more (during the explicit call)
            assert mock_handler.get_server_type.call_count >= 2

    @pytest.mark.asyncio
    async def test_get_load_state_with_progress_when_loading(self):
        """Test get_load_state returns progress when state is Loading"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.get_load_state = AsyncMock(return_value=LoadState.Loading)
            mock_handler.get_loading_progress = AsyncMock(return_value=75)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.get_load_state("test_collection")
            
            assert result["state"] == LoadState.Loading
            assert result["progress"] == 75
            mock_handler.get_load_state.assert_called_once()
            mock_handler.get_loading_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_load_state_without_progress_when_not_loading(self):
        """Test get_load_state does not return progress when state is not Loading"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.get_load_state = AsyncMock(return_value=LoadState.Loaded)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.get_load_state("test_collection")
            
            assert result["state"] == LoadState.Loaded
            assert "progress" not in result
            mock_handler.get_load_state.assert_called_once()
            mock_handler.get_loading_progress.assert_not_called()

    @pytest.mark.asyncio
    async def test_describe_collection_with_struct_array_fields(self):
        """Test describe_collection converts struct_array_fields to user-friendly format"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_result = {
                "fields": [{"name": "field1", "type": "INT64"}],
                "struct_array_fields": [
                    {"name": "struct_field", "type": "STRUCT", "element_type": "INT64"}
                ]
            }
            mock_handler.describe_collection = AsyncMock(return_value=mock_result)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.describe_collection("test_collection")
            
            # Verify struct_array_fields is converted and added to fields
            assert "struct_array_fields" not in result
            assert len(result["fields"]) == 2  # original field + converted struct field
            mock_handler.describe_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_describe_collection_without_struct_array_fields(self):
        """Test describe_collection works normally when no struct_array_fields"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_result = {
                "fields": [{"name": "field1", "type": "INT64"}]
            }
            mock_handler.describe_collection = AsyncMock(return_value=mock_result)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.describe_collection("test_collection")
            
            assert result == mock_result
            assert "struct_array_fields" not in result
            mock_handler.describe_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_ranker(self):
        """Test search method accepts ranker parameter"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.search = AsyncMock(return_value=[[{"id": 1, "distance": 0.1}]])
            mock_fetch.return_value = mock_handler
            
            mock_ranker = MagicMock(spec=Function)
            
            client = AsyncMilvusClient()
            result = await client.search(
                "test_collection",
                data=[[0.1, 0.2]],
                ranker=mock_ranker
            )
            
            assert result == [[{"id": 1, "distance": 0.1}]]
            # Verify ranker was passed to handler
            call_args = mock_handler.search.call_args
            # ranker is passed as keyword argument
            assert call_args.kwargs.get("ranker") == mock_ranker

    @pytest.mark.asyncio
    async def test_hybrid_search_with_function_ranker(self):
        """Test hybrid_search accepts Function type ranker"""
        from pymilvus.client.abstract import AnnSearchRequest
        
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.hybrid_search = AsyncMock(return_value=[[{"id": 1}]])
            mock_fetch.return_value = mock_handler
            
            mock_function_ranker = MagicMock(spec=Function)
            mock_req = MagicMock(spec=AnnSearchRequest)
            
            client = AsyncMilvusClient()
            result = await client.hybrid_search(
                "test_collection",
                reqs=[mock_req],
                ranker=mock_function_ranker
            )
            
            assert result == [[{"id": 1}]]
            # ranker is passed as positional argument (3rd arg), check args[2]
            call_args = mock_handler.hybrid_search.call_args
            assert call_args[0][2] == mock_function_ranker  # ranker is 3rd positional arg

    @pytest.mark.asyncio
    async def test_compact_with_is_l0(self):
        """Test compact method accepts is_l0 parameter"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.compact = AsyncMock(return_value=123)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.compact("test_collection", is_l0=True)
            
            assert result == 123
            call_args = mock_handler.compact.call_args
            assert call_args[1]["is_l0"] is True

    @pytest.mark.asyncio
    async def test_flush_all(self):
        """Test flush_all method"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.flush_all = AsyncMock()
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            await client.flush_all(timeout=10)
            
            mock_handler.flush_all.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_get_flush_all_state(self):
        """Test get_flush_all_state method"""
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.get_flush_all_state = AsyncMock(return_value=True)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.get_flush_all_state(timeout=10)
            
            assert result is True
            mock_handler.get_flush_all_state.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_get_compaction_plans(self):
        """Test get_compaction_plans method"""
        from pymilvus.client.types import CompactionPlans
        
        mock_plans = MagicMock(spec=CompactionPlans)
        
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_handler.get_compaction_plans = AsyncMock(return_value=mock_plans)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.get_compaction_plans(job_id=123, timeout=10)
            
            assert result == mock_plans
            mock_handler.get_compaction_plans.assert_called_once_with(123, timeout=10)

    @pytest.mark.asyncio
    async def test_update_replicate_configuration(self):
        """Test update_replicate_configuration method"""
        clusters = [
            {
                "cluster_id": "cluster1",
                "connection_param": {"uri": "http://localhost:19530", "token": "token1"}
            }
        ]
        cross_cluster_topology = [
            {"source_cluster_id": "cluster1", "target_cluster_id": "cluster2"}
        ]
        
        with patch('pymilvus.milvus_client.async_milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler') as mock_fetch:
            mock_handler = AsyncMock()
            mock_status = MagicMock()
            mock_handler.update_replicate_configuration = AsyncMock(return_value=mock_status)
            mock_fetch.return_value = mock_handler
            
            client = AsyncMilvusClient()
            result = await client.update_replicate_configuration(
                clusters=clusters,
                cross_cluster_topology=cross_cluster_topology,
                timeout=10
            )
            
            assert result == mock_status
            mock_handler.update_replicate_configuration.assert_called_once_with(
                clusters=clusters,
                cross_cluster_topology=cross_cluster_topology,
                timeout=10
            )

    def test_create_struct_field_schema(self):
        """Test create_struct_field_schema class method"""
        result = AsyncMilvusClient.create_struct_field_schema()
        assert isinstance(result, StructFieldSchema)

