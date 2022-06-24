import pytest
import grpc
from pymilvus import MilvusException, MilvusUnavaliableException
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.grpc_gen import milvus_pb2, common_pb2

descriptor = milvus_pb2.DESCRIPTOR.services_by_name['MilvusService']


class TestGrpcHandler:
    @pytest.mark.parametrize("ifHas", [True, False])
    def test_has_collection_no_error(self, channel, client_thread, ifHas):
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(
            handler.has_collection, "fake")

        (invocation_metadata, request, rpc) = (
            channel.take_unary_unary(descriptor.methods_by_name['HasCollection']))
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.BoolResponse(
            status=common_pb2.Status(error_code=common_pb2.Success),
            value=ifHas)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        got_result = has_collection_future.result()
        assert got_result is ifHas

    def test_has_collection_error(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)

        has_collection_future = client_thread.submit(
            handler.has_collection, "fake")

        (invocation_metadata, request, rpc) = (
            channel.take_unary_unary(descriptor.methods_by_name['HasCollection']))
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.BoolResponse(
            status=common_pb2.Status(error_code=common_pb2.UnexpectedError),
            value=False)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        with pytest.raises(MilvusException):
            has_collection_future.result()

    def test_has_collection_UNAVALIABLE_exception(self, channel, client_thread):
        handler = GrpcHandler(channel=channel)
        channel.close()

        # Retry is unable to test
        has_collection_future = client_thread.submit(
            handler.has_collection, "fake", timeout=0)

        (invocation_metadata, request, rpc) = (
            channel.take_unary_unary(descriptor.methods_by_name['HasCollection']))
        rpc.send_initial_metadata(())

        expected_result = milvus_pb2.BoolResponse(
            value=False)

        rpc.terminate(expected_result, (), grpc.StatusCode.UNAVAILABLE, 'server unavaliable')

        with pytest.raises(MilvusUnavaliableException):
            has_collection_future.result()
