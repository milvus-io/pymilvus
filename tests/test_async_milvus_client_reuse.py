import asyncio
import pytest
from unittest.mock import patch, MagicMock
from pymilvus import AsyncMilvusClient

class TestAsyncConnectionReuse:
    def test_async_client_alias_different_loops(self):
        """
        Test that AsyncMilvusClient generates different connection aliases
        for different event loops to avoid reusing closed connections.
        """
        uri = "http://localhost:19530"
        
        async def create_client():
            mock_handler = MagicMock()
            mock_handler.get_server_type.return_value = "milvus"
            
            with patch("pymilvus.orm.connections.Connections.connect") as mock_connect, \
                 patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
                
                client = AsyncMilvusClient(uri=uri)
                # Return the alias used and the current loop id
                return client._using, id(asyncio.get_running_loop())

        # Manually create loops to ensure we have distinct objects
        loop1 = asyncio.new_event_loop()
        loop2 = asyncio.new_event_loop()
        
        try:
            # Run in loop 1
            alias1, loop1_id = loop1.run_until_complete(create_client())
            
            # Run in loop 2
            alias2, loop2_id = loop2.run_until_complete(create_client())
            
            print(f"Loop 1 ID: {loop1_id}, Alias 1: {alias1}")
            print(f"Loop 2 ID: {loop2_id}, Alias 2: {alias2}")

            # Since loop1 and loop2 are both alive (referenced), they must have different IDs
            assert loop1_id != loop2_id, "Two active loops must have different IDs"
            assert alias1 != alias2, "Aliases should be different for different loops"
            
            # Check if loop ID is part of the alias
            assert f"loop{loop1_id}" in alias1
            assert f"loop{loop2_id}" in alias2
            
        finally:
            loop1.close()
            loop2.close()
