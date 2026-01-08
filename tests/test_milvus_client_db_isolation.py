"""
Unit tests for database name isolation fix.

Tests that multiple MilvusClient instances sharing the same connection
maintain separate database contexts.
"""
import pytest
from unittest import mock

from pymilvus import MilvusClient
from pymilvus.milvus_client.async_milvus_client import AsyncMilvusClient


class TestDatabaseIsolation:
    """Test database isolation for MilvusClient instances sharing connections."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        yield
        # Cleanup connections after each test
        from pymilvus.orm.connections import connections
        for alias in list(connections._alias_handlers.keys()):
            try:
                connections.remove_connection(alias)
            except:
                pass

    def test_shared_connection_db_isolation(self):
        """Test that two clients sharing a connection maintain separate database contexts."""
        mock_prefix = "pymilvus.client.grpc_handler.GrpcHandler"
        with mock.patch(f"{mock_prefix}._setup_grpc_channel"), \
             mock.patch(f"{mock_prefix}._wait_for_channel_ready"):
            
            # Create two clients with same URI and db_name
            client_a = MilvusClient(uri="http://localhost:19530", db_name="default")
            client_b = MilvusClient(uri="http://localhost:19530", db_name="default")
            
            # Verify they share the same connection
            assert client_a._using == client_b._using, "Clients should share the same connection"
            
            # Verify initial db_name
            assert client_a._db_name == "default"
            assert client_b._db_name == "default"
            
            # Client A switches database
            client_a.use_database("db1")
            
            # Verify isolation: each client should use its own db_name
            assert client_a._db_name == "db1", "Client A should use db1"
            assert client_b._db_name == "default", "Client B should still use default"
            
            # Mock the list_collections call to verify db_name in kwargs
            conn = client_a._get_connection()
            with mock.patch.object(conn, 'list_collections', return_value=[]) as mock_list:
                client_a.list_collections()
                client_b.list_collections()
                
                # Verify each client passes its own db_name
                assert mock_list.call_count == 2, "list_collections should be called twice"
                
                # Check first call (client_a)
                first_call_kwargs = mock_list.call_args_list[0].kwargs
                assert first_call_kwargs.get('db_name') == 'db1', \
                    f"Client A should pass db1 in kwargs, got {first_call_kwargs.get('db_name')}"
                
                # Check second call (client_b)
                second_call_kwargs = mock_list.call_args_list[1].kwargs
                assert second_call_kwargs.get('db_name') == 'default', \
                    f"Client B should pass default in kwargs, got {second_call_kwargs.get('db_name')}"

    def test_different_db_names_share_connection(self):
        """Test that clients with different db_names share the same connection."""
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"), \
             mock.patch("pymilvus.client.grpc_handler.GrpcHandler._wait_for_channel_ready"):
            
            client_a = MilvusClient(uri="http://localhost:19530", db_name="db1")
            client_b = MilvusClient(uri="http://localhost:19530", db_name="db2")
            
            # After fix: should share connection (same URI)
            assert client_a._using == client_b._using, \
                "Clients with different db_names should share the same connection"
            
            # But maintain separate db contexts
            assert client_a._db_name == "db1"
            assert client_b._db_name == "db2"

    def test_empty_db_name_defaults_to_default(self):
        """Test that empty db_name defaults to 'default'."""
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"), \
             mock.patch("pymilvus.client.grpc_handler.GrpcHandler._wait_for_channel_ready"):
            
            client = MilvusClient(uri="http://localhost:19530", db_name="")
            assert client._db_name == "default", "Empty db_name should default to 'default'"

    def test_none_db_name_defaults_to_default(self):
        """Test that None db_name defaults to 'default'."""
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"), \
             mock.patch("pymilvus.client.grpc_handler.GrpcHandler._wait_for_channel_ready"):
            
            # Pass db_name as None explicitly
            client = MilvusClient(uri="http://localhost:19530")
            assert client._db_name == "default", "None db_name should default to 'default'"

    def test_decorator_injects_db_name(self):
        """Test that @_inject_db_name decorator properly injects db_name into kwargs."""
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"), \
             mock.patch("pymilvus.client.grpc_handler.GrpcHandler._wait_for_channel_ready"):
            
            client = MilvusClient(uri="http://localhost:19530", db_name="test_db")
            
            # Mock a method call
            with mock.patch.object(client._get_connection(), 'list_collections', return_value=[]) as mock_list:
                client.list_collections()
                
                # Verify db_name was injected
                assert mock_list.call_args.kwargs.get('db_name') == 'test_db'

    def test_explicit_db_name_not_overridden(self):
        """Test that explicitly passed db_name in kwargs is not overridden."""
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"), \
             mock.patch("pymilvus.client.grpc_handler.GrpcHandler._wait_for_channel_ready"):
            
            client = MilvusClient(uri="http://localhost:19530", db_name="default")
            
            # Mock a method call with explicit db_name
            with mock.patch.object(client._get_connection(), 'list_collections', return_value=[]) as mock_list:
                # Explicitly pass a different db_name
                client.list_collections(db_name="override_db")
                
                # Verify explicit db_name is preserved
                assert mock_list.call_args.kwargs.get('db_name') == 'override_db', \
                    "Explicit db_name should not be overridden"

    def test_reset_db_name_deprecation_warning(self):
        """Test that reset_db_name() emits a deprecation warning."""
        mock_prefix = "pymilvus.client.grpc_handler.GrpcHandler"
        with mock.patch(f"{mock_prefix}._setup_grpc_channel"), \
             mock.patch(f"{mock_prefix}._wait_for_channel_ready"), \
             mock.patch(f"{mock_prefix}._setup_identifier_interceptor"):
            
            client = MilvusClient(uri="http://localhost:19530")
            conn = client._get_connection()
            
            # Call reset_db_name and expect a deprecation warning
            with pytest.warns(DeprecationWarning, match="reset_db_name.*deprecated"):
                conn.reset_db_name("new_db")

    def test_using_database_calls_use_database(self):
        """Test that using_database() calls use_database()."""
        with mock.patch("pymilvus.client.grpc_handler.GrpcHandler._setup_grpc_channel"), \
             mock.patch("pymilvus.client.grpc_handler.GrpcHandler._wait_for_channel_ready"):
            
            client = MilvusClient(uri="http://localhost:19530", db_name="default")
            
            # Call using_database
            client.using_database("new_db")
            
            # Verify it updated _db_name
            assert client._db_name == "new_db"


class TestAsyncDatabaseIsolation:
    """Test database isolation for AsyncMilvusClient instances."""

    @pytest.mark.asyncio
    async def test_async_shared_connection_db_isolation(self):
        """Test that two async clients sharing a connection maintain separate database contexts."""
        mock_prefix = "pymilvus.client.async_grpc_handler.AsyncGrpcHandler"
        with mock.patch(f"{mock_prefix}._setup_grpc_channel"), \
             mock.patch(f"{mock_prefix}.ensure_channel_ready"):
            
            # Create two async clients with same URI and db_name
            client_a = AsyncMilvusClient(uri="http://localhost:19530", db_name="default")
            client_b = AsyncMilvusClient(uri="http://localhost:19530", db_name="default")
            
            # Verify they share the same connection
            assert client_a._using == client_b._using
            
            # Client A switches database
            client_a.use_database("db1")
            
            # Verify isolation
            assert client_a._db_name == "db1"
            assert client_b._db_name == "default"

    @pytest.mark.asyncio
    async def test_async_different_db_names_share_connection(self):
        """Test that async clients with different db_names share the same connection."""
        mock_prefix = "pymilvus.client.async_grpc_handler.AsyncGrpcHandler"
        with mock.patch(f"{mock_prefix}._setup_grpc_channel"), \
             mock.patch(f"{mock_prefix}.ensure_channel_ready"):
            
            client_a = AsyncMilvusClient(uri="http://localhost:19530", db_name="db1")
            client_b = AsyncMilvusClient(uri="http://localhost:19530", db_name="db2")
            
            # Should share connection
            assert client_a._using == client_b._using
            
            # But maintain separate db contexts
            assert client_a._db_name == "db1"
            assert client_b._db_name == "db2"
