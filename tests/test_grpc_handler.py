from typing import Any
from unittest.mock import MagicMock, patch

import grpc
import pytest
from pymilvus import CollectionSchema, FieldSchema, MilvusException, Function, FunctionType
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import common_pb2, milvus_pb2
from pymilvus.orm.types import DataType

descriptor = milvus_pb2.DESCRIPTOR.services_by_name["MilvusService"]


class TestGrpcHandler:
    @pytest.mark.parametrize("has", [True, False])
    def test_has_collection_no_error(self, channel, client_thread, has):
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(
            handler.has_collection, "fake")

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DescribeCollection"]
        )
        rpc.send_initial_metadata(())

        reason = "" if has else "can't find collection"
        code = 0 if has else 100

        expected_result = milvus_pb2.DescribeCollectionResponse(
            status=common_pb2.Status(code=code, reason=reason),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        got_result = has_collection_future.result()
        assert got_result is has

    def test_has_collection_error(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(
            handler.has_collection, "fake")

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DescribeCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.DescribeCollectionResponse(
            status=common_pb2.Status(code=1, reason="other reason"),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            has_collection_future.result()

    def test_has_collection_Unavailable_exception(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)
        channel.close()

        # Retry is unable to test
        has_collection_future = client_thread.submit(
            handler.has_collection, "fake", timeout=0)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DescribeCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.DescribeCollectionResponse()

        rpc.terminate(expected_result, (),
                      grpc.StatusCode.UNAVAILABLE, "server Unavailable")

        with pytest.raises(MilvusException):
            has_collection_future.result()

    def test_get_server_version_error(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)

        get_version_future = client_thread.submit(handler.get_server_version)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["GetVersion"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.GetVersionResponse(
            status=common_pb2.Status(code=1, reason="unexpected error"),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            get_version_future.result()

    def test_get_server_version(self, channel, client_thread):
        version = "2.2.0"
        handler = GrpcHandler(channel=channel)

        get_version_future = client_thread.submit(handler.get_server_version)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["GetVersion"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.GetVersionResponse(
            status=common_pb2.Status(code=0),
            version=version,
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        got_result = get_version_future.result()
        assert got_result == version

    @pytest.mark.parametrize("_async", [True])
    def test_flush_all(self, channel, client_thread, _async):
        handler = GrpcHandler(channel=channel)

        flush_all_future = client_thread.submit(
            handler.flush_all, _async=_async, timeout=10)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["FlushAll"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.FlushAllResponse(
            status=common_pb2.Status(code=0),
            flush_all_ts=100,
        )

        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")
        assert flush_all_future is not None

    def test_get_flush_all_state(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)

        flushed = client_thread.submit(
            handler.get_flush_all_state, flush_all_ts=100)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["GetFlushAllState"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.GetFlushStateResponse(
            status=common_pb2.Status(code=0),
            flushed=True,
        )

        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")
        assert flushed.result() is True


class TestGrpcHandlerInitialization:
    def test_init_with_uri(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(uri="http://localhost:19530")
            assert handler.server_address == "localhost:19530"

    def test_init_with_host_port(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(host="localhost", port="19530")
            assert handler.server_address == "localhost:19530"

    def test_init_with_secure_connection(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.secure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(uri="http://localhost:19530", secure=True)
            assert handler.server_address == "localhost:19530"

    def test_init_with_invalid_secure_param(self) -> None:
        with pytest.raises(ParamError, match="secure must be bool type"):
            GrpcHandler(uri="http://localhost:19530", secure="not_bool")

    def test_init_with_authorization(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(
                uri="http://localhost:19530",
                user="test_user",
                password="test_password"
            )
            assert handler.server_address == "localhost:19530"

    def test_init_with_token(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(
                uri="http://localhost:19530",
                token="test_token"
            )
            assert handler.server_address == "localhost:19530"

    def test_init_with_db_name(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(
                uri="http://localhost:19530",
                db_name="test_db"
            )
            assert handler.server_address == "localhost:19530"

    def test_get_server_type(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(uri="http://localhost:19530")
            # get_server_type will return 'milvus' for localhost
            server_type = handler.get_server_type()
            assert server_type == "milvus"


class TestGrpcHandlerStateManagement:
    def test_register_state_change_callback(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch
            handler = GrpcHandler(uri="http://localhost:19530")

            def callback(state: Any) -> None:
                pass

            handler.register_state_change_callback(callback)
            assert callback in handler.callbacks
            mock_ch.subscribe.assert_called_once_with(
                callback, try_to_connect=True)

    def test_deregister_state_change_callbacks(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch
            handler = GrpcHandler(uri="http://localhost:19530")

            def callback(state: Any) -> None:
                pass

            handler.register_state_change_callback(callback)
            handler.deregister_state_change_callbacks()

            assert len(handler.callbacks) == 0
            mock_ch.unsubscribe.assert_called_once_with(callback)

    def test_close(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch
            handler = GrpcHandler(uri="http://localhost:19530")

            # Register a callback first
            def callback(state: Any) -> None:
                pass
            handler.register_state_change_callback(callback)

            handler.close()
            # Verify close was called on the channel
            mock_ch.close.assert_called_once()

    def test_reset_db_name(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(uri="http://localhost:19530")

            # Add some dummy data to schema_cache
            handler.schema_cache["test_collection"] = {"field": "value"}

            with patch.object(handler, '_setup_identifier_interceptor'):
                handler.reset_db_name("new_db")

            assert len(handler.schema_cache) == 0

    def test_set_onetime_loglevel(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            handler = GrpcHandler(uri="http://localhost:19530")

            # Test passes if no exception is raised
            handler.set_onetime_loglevel("DEBUG")

    def test_wait_for_channel_ready_success(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch
            handler = GrpcHandler(uri="http://localhost:19530")

            with patch('pymilvus.client.grpc_handler.grpc.channel_ready_future') as mock_future:
                mock_result = MagicMock()
                mock_result.result.return_value = None
                mock_future.return_value = mock_result

                with patch.object(handler, '_setup_identifier_interceptor'):
                    handler._wait_for_channel_ready(timeout=10)

                mock_future.assert_called_once_with(mock_ch)

    def test_wait_for_channel_ready_timeout(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_channel.return_value = mock_ch
            handler = GrpcHandler(uri="http://localhost:19530")

            with patch('pymilvus.client.grpc_handler.grpc.channel_ready_future') as mock_future:
                mock_result = MagicMock()
                mock_result.result.side_effect = grpc.FutureTimeoutError()
                mock_future.return_value = mock_result

                with pytest.raises(MilvusException) as exc_info:
                    handler._wait_for_channel_ready(timeout=10)

                assert "Fail connecting to server" in str(exc_info.value)

    def test_wait_for_channel_ready_no_channel(self) -> None:
        handler = GrpcHandler(channel=None)
        # Manually set channel to None to test the error case
        handler._channel = None

        with pytest.raises(MilvusException) as exc_info:
            handler._wait_for_channel_ready()

        assert "No channel in handler" in str(exc_info.value)


class TestGrpcHandlerCollectionOperations:
    def test_create_collection_sync(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ])

        create_future = client_thread.submit(
            handler.create_collection,
            collection_name="test_collection",
            fields=schema,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["CreateCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = create_future.result()
        assert result is None

    def test_create_collection_async(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ])

        create_future = client_thread.submit(
            handler.create_collection,
            collection_name="test_collection",
            fields=schema,
            timeout=10,
            _async=True
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["CreateCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = create_future.result()
        assert result is not None  # Should return a future object

    def test_drop_collection(self, channel: Any, client_thread: Any) -> None:
        """Test drop_collection"""
        handler = GrpcHandler(channel=channel)

        drop_future = client_thread.submit(
            handler.drop_collection,
            collection_name="test_collection",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DropCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = drop_future.result()
        assert result is None

    def test_add_collection_field(self, channel: Any, client_thread: Any) -> None:
        """Test add_collection_field"""
        handler = GrpcHandler(channel=channel)

        field_schema = FieldSchema(name="new_field", dtype=DataType.INT64)

        add_field_future = client_thread.submit(
            handler.add_collection_field,
            collection_name="test_collection",
            field_schema=field_schema,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["AddCollectionField"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = add_field_future.result()
        assert result is None

    def test_alter_collection_properties(self, channel: Any, client_thread: Any) -> None:
        """Test alter_collection_properties"""
        handler = GrpcHandler(channel=channel)

        properties = {"prop1": "value1", "prop2": "value2"}

        alter_future = client_thread.submit(
            handler.alter_collection_properties,
            collection_name="test_collection",
            properties=properties,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["AlterCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = alter_future.result()
        assert result is None

    def test_alter_collection_field_properties(self, channel: Any, client_thread: Any) -> None:
        """Test alter_collection_field_properties"""
        handler = GrpcHandler(channel=channel)

        field_params = {"param1": "value1"}

        alter_field_future = client_thread.submit(
            handler.alter_collection_field_properties,
            collection_name="test_collection",
            field_name="test_field",
            field_params=field_params,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["AlterCollectionField"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = alter_field_future.result()
        assert result is None

    def test_drop_collection_properties(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        property_keys = ["prop1", "prop2"]

        drop_props_future = client_thread.submit(
            handler.drop_collection_properties,
            collection_name="test_collection",
            property_keys=property_keys,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["AlterCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = drop_props_future.result()
        assert result is None

    def test_add_collection_function(self, channel: Any, client_thread: Any) -> None:
        """Test add_collection_function"""
        handler = GrpcHandler(channel=channel)

        function = Function("test", FunctionType.TEXTEMBEDDING, input_field_names=[
                            "text"], output_field_names=["embedding"])

        add_function_future = client_thread.submit(
            handler.add_collection_function,
            collection_name="test_collection",
            function=function,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["AddCollectionFunction"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = add_function_future.result()
        assert result is None

    def test_alter_collection_function(self, channel: Any, client_thread: Any) -> None:
        """Test alter_collection_function"""
        handler = GrpcHandler(channel=channel)

        function = Function("test", FunctionType.TEXTEMBEDDING, input_field_names=[
                            "text"], output_field_names=["embedding"])

        alter_function_future = client_thread.submit(
            handler.alter_collection_function,
            collection_name="test_collection",
            function_name="test",
            function=function,
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["AlterCollectionFunction"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = alter_function_future.result()
        assert result is None

    def test_drop_collection_function(self, channel: Any, client_thread: Any) -> None:
        """Test drop_collection_function"""
        handler = GrpcHandler(channel=channel)

        drop_function_future = client_thread.submit(
            handler.drop_collection_function,
            collection_name="test_collection",
            function_name="test",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DropCollectionFunction"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = drop_function_future.result()
        assert result is None

    def test_has_collection_compatibility(self, channel: Any, client_thread: Any) -> None:
        """Test has_collection with Milvus < 2.3.2 compatibility"""
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(
            handler.has_collection, "fake")

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DescribeCollection"]
        )
        rpc.send_initial_metadata(())

        # Test compatibility with older Milvus versions
        expected_result = milvus_pb2.DescribeCollectionResponse(
            status=common_pb2.Status(
                error_code=common_pb2.UnexpectedError,
                reason="can't find collection fake"
            ),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        got_result = has_collection_future.result()
        assert got_result is False


class TestGrpcHandlerPasswordReset:
    def test_reset_password(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        with patch.object(handler, 'update_password') as mock_update:
            with patch.object(handler, '_setup_authorization_interceptor') as mock_auth:
                with patch.object(handler, '_setup_grpc_channel') as mock_setup:
                    handler.reset_password(
                        user="test_user",
                        old_password="old_pass",
                        new_password="new_pass",
                        timeout=10
                    )

                    mock_update.assert_called_once_with(
                        "test_user", "old_pass", "new_pass", timeout=10
                    )
                    mock_auth.assert_called_once_with(
                        "test_user", "new_pass", None)
                    mock_setup.assert_called_once()


class TestGrpcHandlerSecureConnection:
    def test_setup_grpc_channel_with_tls(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.secure_channel') as mock_secure:
            with patch('pymilvus.client.grpc_handler.grpc.ssl_channel_credentials') as mock_creds:
                with patch('pymilvus.client.grpc_handler.Path') as mock_path:
                    # Mock file reading
                    mock_file = MagicMock()
                    mock_file.read.return_value = b"cert_content"
                    mock_path.return_value.open.return_value.__enter__.return_value = mock_file

                    mock_secure.return_value = MagicMock()
                    mock_creds.return_value = MagicMock()

                    GrpcHandler(
                        uri="http://localhost:19530",
                        secure=True,
                        server_pem_path="/path/to/server.pem"
                    )

                    # Verify secure channel was created
                    mock_creds.assert_called_once()
                    mock_secure.assert_called_once()

    def test_setup_grpc_channel_with_client_certs(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.secure_channel') as mock_secure:
            with patch('pymilvus.client.grpc_handler.grpc.ssl_channel_credentials') as mock_creds:
                with patch('pymilvus.client.grpc_handler.Path') as mock_path:
                    # Mock file reading
                    mock_file = MagicMock()
                    mock_file.read.return_value = b"cert_content"
                    mock_path.return_value.open.return_value.__enter__.return_value = mock_file

                    mock_secure.return_value = MagicMock()
                    mock_creds.return_value = MagicMock()

                    GrpcHandler(
                        uri="http://localhost:19530",
                        secure=True,
                        client_pem_path="/path/to/client.pem",
                        client_key_path="/path/to/client.key",
                        ca_pem_path="/path/to/ca.pem"
                    )

                    # Verify secure channel was created
                    mock_creds.assert_called_once()
                    mock_secure.assert_called_once()

    def test_setup_grpc_channel_with_server_name_override(self) -> None:
        with patch('pymilvus.client.grpc_handler.grpc.secure_channel') as mock_secure:
            with patch('pymilvus.client.grpc_handler.grpc.ssl_channel_credentials') as mock_creds:
                mock_secure.return_value = MagicMock()
                mock_creds.return_value = MagicMock()

                GrpcHandler(
                    uri="http://localhost:19530",
                    secure=True,
                    server_name="custom.server.name"
                )

                # Check that the server name override was added to options
                call_args = mock_secure.call_args
                options = call_args[1]['options']
                assert any(
                    opt[0] == "grpc.ssl_target_name_override" and opt[1] == "custom.server.name"
                    for opt in options
                )


class TestGrpcHandlerListAndRenameOperations:
    @pytest.mark.parametrize("result", [["collection1", "collection2", "collection3"], []])
    def test_list_collections(self, channel: grpc.Channel, client_thread: Any, result) -> None:
        handler = GrpcHandler(channel=channel)

        list_future = client_thread.submit(
            handler.list_collections, timeout=10)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["ShowCollections"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.ShowCollectionsResponse(
            status=common_pb2.Status(code=0),
            collection_names=result
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        assert list_future.result() == result

    def test_list_collections_error(self, channel: grpc.Channel, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)
        list_future = client_thread.submit(
            handler.list_collections, timeout=10)

        (_, _, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["ShowCollections"])
        rpc.send_initial_metadata(())

        rpc.terminate(
            milvus_pb2.ShowCollectionsResponse(
                status=common_pb2.Status(code=1, reason="Internal error")),
            (),
            grpc.StatusCode.OK,
            "",
        )

        with pytest.raises(MilvusException):
            list_future.result()

    def test_rename_collections(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        rename_future = client_thread.submit(
            handler.rename_collections,
            old_name="old_collection",
            new_name="new_collection",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["RenameCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = rename_future.result()
        assert result is None

    def test_rename_collections_with_new_db(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        rename_future = client_thread.submit(
            handler.rename_collections,
            old_name="old_collection",
            new_name="new_collection",
            new_db_name="new_database",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["RenameCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = rename_future.result()
        assert result is None

    def test_rename_collections_error(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        rename_future = client_thread.submit(
            handler.rename_collections,
            old_name="old_collection",
            new_name="new_collection",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["RenameCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(
            code=1, reason="Collection already exists")
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            rename_future.result()


class TestGrpcHandlerPartitionOperations:
    def test_create_partition(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        create_future = client_thread.submit(
            handler.create_partition,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["CreatePartition"]
        )
        rpc.send_initial_metadata(())
        rpc.terminate(common_pb2.Status(code=0), (), grpc.StatusCode.OK, "")

        result = create_future.result()
        assert result is None

    def test_create_partition_error(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        create_future = client_thread.submit(
            handler.create_partition,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["CreatePartition"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(
            code=1, reason="Partition already exists")
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            create_future.result()

    def test_drop_partition(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        drop_future = client_thread.submit(
            handler.drop_partition,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DropPartition"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(code=0)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = drop_future.result()
        assert result is None

    def test_drop_partition_error(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        drop_future = client_thread.submit(
            handler.drop_partition,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DropPartition"]
        )
        rpc.send_initial_metadata(())

        expected_result = common_pb2.Status(
            code=1, reason="Partition not found")
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            drop_future.result()

    @pytest.mark.parametrize("has", [True, False])
    def test_has_partition(self, channel: Any, client_thread: Any, has: bool) -> None:
        handler = GrpcHandler(channel=channel)

        has_future = client_thread.submit(
            handler.has_partition,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["HasPartition"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.BoolResponse(
            status=common_pb2.Status(code=0),
            value=has
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = has_future.result()
        assert result == has

    def test_has_partition_error(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        has_future = client_thread.submit(
            handler.has_partition,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["HasPartition"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.BoolResponse(
            status=common_pb2.Status(code=1, reason="Internal error")
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        with pytest.raises(MilvusException):
            has_future.result()

    def test_list_partitions(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        list_future = client_thread.submit(
            handler.list_partitions,
            collection_name="test_collection",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["ShowPartitions"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.ShowPartitionsResponse(
            status=common_pb2.Status(code=0),
            partition_names=["_default", "partition1", "partition2"]
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = list_future.result()
        assert result == ["_default", "partition1", "partition2"]

    def test_list_partitions_empty(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        list_future = client_thread.submit(
            handler.list_partitions,
            collection_name="test_collection",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["ShowPartitions"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.ShowPartitionsResponse(
            status=common_pb2.Status(code=0),
            partition_names=[]
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = list_future.result()
        assert result == []

    def test_get_partition_stats(self, channel: Any, client_thread: Any) -> None:
        handler = GrpcHandler(channel=channel)

        stats_future = client_thread.submit(
            handler.get_partition_stats,
            collection_name="test_collection",
            partition_name="test_partition",
            timeout=10
        )

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["GetPartitionStatistics"]
        )
        rpc.send_initial_metadata(())

        stats = [
            common_pb2.KeyValuePair(key="row_count", value="1000")
        ]
        expected_result = milvus_pb2.GetPartitionStatisticsResponse(
            status=common_pb2.Status(code=0),
            stats=stats
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        result = stats_future.result()
        assert result == stats
