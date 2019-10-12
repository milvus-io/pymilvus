import grpc
import pytest
import mock

import sys

sys.path.append(".")
from milvus.client.grpc_client import Prepare, GrpcMilvus, Status
from milvus.client.Abstract import IndexType, TableSchema, TopKQueryResult, MetricType
from milvus.client.Exceptions import *


class TestConnectException:

    client = GrpcMilvus()

    @mock.patch("grpc.channel_ready_future", side_effect=grpc.FutureTimeoutError())
    def test_connect_timeout_exp(self, mock_channel):

        with pytest.raises(NotConnectError):
            self.client.connect()

    @mock.patch("grpc.channel_ready_future", side_effect=grpc.RpcError())
    def test_connect_grpc_exp(self, mock_channel):

        with pytest.raises(NotConnectError):
            self.client.connect()


# class TestDisconnectException:
#     client = GrpcMilvus()
#
#     def