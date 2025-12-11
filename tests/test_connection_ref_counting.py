"""Test cases for connection reference counting fix.

This tests the fix for the bug where closing one MilvusClient disconnects
all other clients sharing the same connection.
"""

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from pymilvus.orm.connections import Connections


class TestConnectionRefCounting:
    """Test reference counting in Connections class."""

    def test_incr_ref_new_alias(self):
        """Test incrementing ref count for new alias."""
        conn = Connections.__new__(Connections)
        conn._alias_ref_counts = {}
        
        conn._incr_ref("test_alias")
        
        assert conn._alias_ref_counts["test_alias"] == 1

    def test_incr_ref_existing_alias(self):
        """Test incrementing ref count for existing alias."""
        conn = Connections.__new__(Connections)
        conn._alias_ref_counts = {"test_alias": 2}
        
        conn._incr_ref("test_alias")
        
        assert conn._alias_ref_counts["test_alias"] == 3

    def test_decr_ref_and_check_should_close(self):
        """Test decrementing to zero returns True (should close)."""
        conn = Connections.__new__(Connections)
        conn._alias_ref_counts = {"test_alias": 1}
        
        result = conn._decr_ref_and_check("test_alias")
        
        assert result is True
        assert "test_alias" not in conn._alias_ref_counts

    def test_decr_ref_and_check_should_not_close(self):
        """Test decrementing with remaining refs returns False."""
        conn = Connections.__new__(Connections)
        conn._alias_ref_counts = {"test_alias": 2}
        
        result = conn._decr_ref_and_check("test_alias")
        
        assert result is False
        assert conn._alias_ref_counts["test_alias"] == 1

    def test_decr_ref_and_check_unknown_alias(self):
        """Test decrementing unknown alias returns True."""
        conn = Connections.__new__(Connections)
        conn._alias_ref_counts = {}
        
        result = conn._decr_ref_and_check("unknown")
        
        assert result is True


class TestMilvusClientConnectionSharing:
    """Test that MilvusClient instances properly share connections."""

    def test_multiple_clients_same_uri_share_connection(self):
        """Test multiple clients with same URI share connection."""
        from pymilvus.milvus_client.milvus_client import MilvusClient
        from pymilvus.orm.connections import connections

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch("pymilvus.orm.connections.Connections.connect"), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            
            client1 = MilvusClient(uri="http://localhost:19530")
            client2 = MilvusClient(uri="http://localhost:19530")

            # Same alias means same connection
            assert client1._using == client2._using
            # Ref count should be 2
            assert connections._alias_ref_counts.get(client1._using, 0) == 2

    def test_close_one_client_does_not_affect_others(self):
        """Test that closing one client doesn't affect others."""
        from pymilvus.milvus_client.milvus_client import MilvusClient
        from pymilvus.orm.connections import connections

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        # Clean up any existing state for this alias
        test_alias = "http://localhost:19530"
        connections._alias_ref_counts.pop(test_alias, None)
        connections._alias_handlers.pop(test_alias, None)
        connections._alias_config.pop(test_alias, None)

        with patch("pymilvus.orm.connections.Connections.connect"), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):

            client1 = MilvusClient(uri="http://localhost:19530")
            client2 = MilvusClient(uri="http://localhost:19530")
            alias = client1._using

            # Close first client
            client1.close()

            # Ref count should be 1, connection still exists
            assert connections._alias_ref_counts.get(alias, 0) == 1

    def test_close_all_clients_removes_connection(self):
        """Test closing all clients removes the connection."""
        from pymilvus.milvus_client.milvus_client import MilvusClient
        from pymilvus.orm.connections import connections

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.close = MagicMock()

        with patch("pymilvus.orm.connections.Connections.connect"), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):

            client1 = MilvusClient(uri="http://test:19530")
            client2 = MilvusClient(uri="http://test:19530")
            alias = client1._using

            # Add handler to simulate real connection
            connections._alias_handlers[alias] = mock_handler

            # Close both clients
            client1.close()
            client2.close()

            # Ref count should be 0/removed
            assert alias not in connections._alias_ref_counts
            # Handler should be removed and close called
            assert alias not in connections._alias_handlers
            mock_handler.close.assert_called_once()


class TestAsyncMilvusClientConnectionSharing:
    """Test that AsyncMilvusClient handles reference counting."""

    @pytest.mark.asyncio
    async def test_async_close_respects_ref_count(self):
        """Test async close respects reference count."""
        from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClient
        from pymilvus.orm.connections import connections

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.close = AsyncMock()

        with patch("pymilvus.orm.connections.Connections.connect"), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):

            # Simulate two async clients sharing connection
            alias = "async-http://localhost:19530"
            connections._alias_ref_counts[alias] = 2
            connections._alias_handlers[alias] = mock_handler

            # Decrement once
            await connections.async_disconnect(alias)

            # Should not close yet
            assert connections._alias_ref_counts.get(alias) == 1
            mock_handler.close.assert_not_called()

            # Decrement again (last client)
            await connections.async_disconnect(alias)

            # Now should close
            assert alias not in connections._alias_ref_counts
            mock_handler.close.assert_called_once()
