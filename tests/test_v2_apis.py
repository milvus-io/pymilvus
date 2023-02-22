import pytest
import grpc
from pymilvus.v2 import MilvusClient
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, milvus_pb2


class TestGetServerVersion:
    @pytest.mark.parametrize("error_code", [
        common_pb2.Success,
        common_pb2.UnexpectedError,
        common_pb2.ConnectFailed,
    ])
    def test_normal(self, rpc_future_GetVersion, error_code):
        rpc, future = rpc_future_GetVersion

        reason = f"error: {error_code}" if error_code != common_pb2.Success else ""
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
