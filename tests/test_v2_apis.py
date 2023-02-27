import pytest
import grpc
from pymilvus.v2 import MilvusClient
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, milvus_pb2

descriptor = milvus_pb2.DESCRIPTOR.services_by_name['MilvusService']

def prep_channel(channel, method_name):
    (invocation_metadata, request, rpc) = (
        channel.take_unary_unary(descriptor.methods_by_name[method_name]))

    rpc.send_initial_metadata(())
    return rpc


@pytest.fixture(params=[
    common_pb2.Success,
    common_pb2.UnexpectedError,
    common_pb2.CollectionNotExists,
])
def error_code(request):
    yield request.param


class TestGetServerVersion:

    def test_normal(self, client_channel, client_thread, error_code):
        client, channel = client_channel
        future = client_thread.submit(client.get_server_version)

        rpc = prep_channel(channel, 'GetVersion')

        reason = f"mock error: {error_code}" if error_code != common_pb2.Success else ""
        version = "test.test.test" if error_code == common_pb2.Success else ""

        expected_result = milvus_pb2.GetVersionResponse(
            status=common_pb2.Status(error_code=error_code, reason=reason),
            version=version,
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        if error_code != common_pb2.Success:
            with pytest.raises(MilvusException) as excinfo:
                got_result = future.result()
            assert error_code == excinfo.value.code
        else:
            got_result = future.result()
            assert got_result == version


class TestAlias:
    def test_create_alias(self, client_channel, client_thread, error_code):
        client, channel = client_channel
        future = client_thread.submit(client.create_alias, "alias", "coll")

        rpc = prep_channel(channel, 'CreateAlias')

        reason = f"mock error: {error_code}" if error_code != common_pb2.Success else ""

        expected_result = common_pb2.Status(error_code=error_code, reason=reason)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        if error_code != common_pb2.Success:
            with pytest.raises(MilvusException) as excinfo:
                future.result()
            assert error_code == excinfo.value.code
        else:
            future.result()

    def test_alter_alias(self, client_channel, client_thread, error_code):
        client, channel = client_channel
        future = client_thread.submit(client.alter_alias, "alias", "coll")

        rpc = prep_channel(channel, 'AlterAlias')

        reason = f"mock error: {error_code}" if error_code != common_pb2.Success else ""

        expected_result = common_pb2.Status(error_code=error_code, reason=reason)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        if error_code != common_pb2.Success:
            with pytest.raises(MilvusException) as excinfo:
                future.result()
            assert error_code == excinfo.value.code
        else:
            future.result()

    def test_drop_alias(self, client_channel, client_thread, error_code):
        client, channel = client_channel
        future = client_thread.submit(client.drop_alias, "alias")

        rpc = prep_channel(channel, 'DropAlias')

        reason = f"mock error: {error_code}" if error_code != common_pb2.Success else ""

        expected_result = common_pb2.Status(error_code=error_code, reason=reason)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        if error_code != common_pb2.Success:
            with pytest.raises(MilvusException) as excinfo:
                future.result()
            assert error_code == excinfo.value.code
        else:
            future.result()


class TestCollection:
    def test_drop_collection(self, client_channel, client_thread, error_code):
        client, channel = client_channel
        future = client_thread.submit(client.drop_collection, "cname")

        rpc = prep_channel(channel, 'DropCollection')

        reason = f"mock error: {error_code}" if error_code != common_pb2.Success else ""

        expected_result = common_pb2.Status(error_code=error_code, reason=reason)
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        if error_code != common_pb2.Success:
            with pytest.raises(MilvusException) as excinfo:
                future.result()
            assert error_code == excinfo.value.code
        else:
            future.result()

    def test_has_collection_error(self, client_channel, client_thread):
        client, channel = client_channel
        future = client_thread.submit(client.has_collection, "cname")

        rpc = prep_channel(channel, 'DescribeCollection')

        reason = f"mock describe collection error"

        expected_result = milvus_pb2.DescribeCollectionResponse(
            status= common_pb2.Status(error_code=common_pb2.UnexpectedError, reason=reason),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        with pytest.raises(MilvusException) as excinfo:
            future.result()
        assert excinfo.value.code == common_pb2.UnexpectedError

    def test_has_collection_normal(self, client_channel, client_thread, error_code):
        client, channel = client_channel
        future = client_thread.submit(client.has_collection, "cname")

        rpc = prep_channel(channel, 'DescribeCollection')

        reason = "" if error_code == common_pb2.Success else "can\'t find collection"

        expected_result = milvus_pb2.DescribeCollectionResponse(
            status= common_pb2.Status(error_code=error_code, reason=reason),
        )
        rpc.terminate(expected_result, (), grpc.StatusCode.OK, '')

        got_result = future.result()
        assert got_result == (error_code == common_pb2.Success)

