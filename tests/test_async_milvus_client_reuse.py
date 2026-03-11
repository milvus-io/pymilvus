import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from pymilvus import AsyncMilvusClient
from pymilvus.client.connection_manager import AsyncConnectionManager

LOGGER = logging.getLogger(__name__)


class TestAsyncConnectionReuse:
    def test_async_client_alias_different_loops(self):
        """
        Test that AsyncMilvusClient can be used in different event loops
        and each client gets its own handler reference.
        """
        uri = "http://localhost:19530"

        async def create_client():
            # Reset the singleton to ensure clean state for each loop
            AsyncConnectionManager._reset_instance()

            mock_handler = MagicMock()
            mock_handler.get_server_type.return_value = "milvus"
            mock_handler.ensure_channel_ready = AsyncMock()

            with patch(
                "pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=mock_handler
            ):
                client = AsyncMilvusClient(uri=uri)
                await client._connect()
                # Return the alias used and the current loop id
                return client._using, id(asyncio.get_running_loop()), id(client._handler)

        # Manually create loops to ensure we have distinct objects
        loop1 = asyncio.new_event_loop()
        loop2 = asyncio.new_event_loop()

        try:
            # Run in loop 1
            alias1, loop1_id, handler1_id = loop1.run_until_complete(create_client())

            # Run in loop 2
            alias2, loop2_id, handler2_id = loop2.run_until_complete(create_client())

            LOGGER.info(f"Loop 1 ID: {loop1_id}, Alias 1: {alias1}, Handler ID: {handler1_id}")
            LOGGER.info(f"Loop 2 ID: {loop2_id}, Alias 2: {alias2}, Handler ID: {handler2_id}")

            # Since loop1 and loop2 are both alive (referenced), they must have different IDs
            assert loop1_id != loop2_id, "Two active loops must have different IDs"

            # Each client should have a valid alias (not None)
            assert alias1 is not None, "Client 1 should have a valid alias after connect"
            assert alias2 is not None, "Client 2 should have a valid alias after connect"

            # The aliases are based on handler IDs, which should be different since
            # we reset the manager and create new mock handlers for each loop
            assert alias1 != alias2, "Aliases should be different for different handlers"

        finally:
            # Reset singleton state
            AsyncConnectionManager._reset_instance()
            loop1.close()
            loop2.close()
