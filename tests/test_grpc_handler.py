import pytest
import grpc
from pymilvus import MilvusException, MilvusUnavailableException
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.grpc_gen import milvus_pb2, common_pb2

descriptor = milvus_pb2.DESCRIPTOR.services_by_name["MilvusService"]


class TestGrpcHandler:
    @pytest.mark.parametrize("ifHas", [True, False])
    def test_has_collection_no_error(self, channel, client_thread, ifHas):
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(handler.has_collection, "fake")

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DescribeCollection"]
        )
        rpc.send_initial_metadata(())

        reason = "" if ifHas else "can't find collection"
        code = 0 if ifHas else 100

        expected_result = milvus_pb2.DescribeCollectionResponse(
            status=common_pb2.Status(code=code, reason=reason),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, "")

        got_result = has_collection_future.result()
        assert got_result is ifHas

    def test_has_collection_error(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(handler.has_collection, "fake")

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
        has_collection_future = client_thread.submit(handler.has_collection, "fake", timeout=0)

        (invocation_metadata, request, rpc) = channel.take_unary_unary(
            descriptor.methods_by_name["DescribeCollection"]
        )
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.DescribeCollectionResponse()

        rpc.terminate(expected_result, (), grpc.StatusCode.UNAVAILABLE, "server Unavailable")

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

        flush_all_future = client_thread.submit(handler.flush_all, _async=_async, timeout=10)

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

        flushed = client_thread.submit(handler.get_flush_all_state, flush_all_ts=100)

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
