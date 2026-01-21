
import pytest
from unittest.mock import MagicMock, patch
from pymilvus import MilvusClient
from pymilvus.orm.connections import connections

class TestMilvusClientDBIsolation:
    def test_client_db_isolation(self):
        """
        Test that two clients sharing the same connection but using different databases
        remain isolated when one switches database.
        """
        # Mock the handler
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        
        # We need to mock create_connection to return a fixed alias
        # and connections._fetch_handler to return our mock handler
        with patch("pymilvus.milvus_client._utils.create_connection", return_value="shared_alias"), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler), \
             patch("pymilvus.orm.connections.Connections.has_connection", return_value=True): # Simulate second connect finding existing connection

            # 1. Client A connects to default database
            client_a = MilvusClient(uri="http://localhost:19530", db_name="default")
            
            # 2. Client B connects to "testdb", reusing the same connection alias
            client_b = MilvusClient(uri="http://localhost:19530", db_name="testdb")

            # Verify initial state
            assert client_a._db_name == "default"
            assert client_b._db_name == "testdb"

            # 3. Client A switches database to "db1"
            # This simulates the user scenario where Client A modifies shared state
            client_a.use_database("db1")
            
            assert client_a._db_name == "db1"
            assert client_b._db_name == "testdb" # Client B should remain on testdb

            # 4. Client B lists collections
            # This triggered the bug where it would query "db1" instead of "testdb"
            client_b.list_collections()

            # Verify that list_collections was called with the correct context
            assert mock_handler.list_collections.called
            _, kwargs = mock_handler.list_collections.call_args
            context = kwargs.get('context')
            
            assert context is not None
            # Context must contain "testdb", NOT "db1"
            assert context.get_db_name() == "testdb"
