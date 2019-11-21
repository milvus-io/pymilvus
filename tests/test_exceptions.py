import random

import sys

sys.path.append(".")

import grpc
import pytest
import mock

from grpc import FutureTimeoutError as FError

from milvus import Status
from milvus.client.grpc_client import GrpcMilvus
from milvus.client.exceptions import NotConnectError


class RpcTestError(grpc.RpcError):

    def code(self):
        return 10

    def details(self):
        return "test"


class TestConnectException:
    client = GrpcMilvus()

    @mock.patch("grpc.channel_ready_future", side_effect=grpc.FutureTimeoutError())
    def test_connect_timeout_exp(self, _):
        with pytest.raises(NotConnectError):
            self.client.connect()

    @mock.patch("grpc.channel_ready_future", side_effect=grpc.RpcError())
    def test_connect_grpc_exp(self, _):
        with pytest.raises(NotConnectError):
            self.client.connect()

    @mock.patch("grpc.channel_ready_future", side_effect=Exception())
    def test_connect_grpc_exp(self, _):
        with pytest.raises(NotConnectError):
            self.client.connect()


class TestConnectedException:
    client = GrpcMilvus()

    @mock.patch("grpc.channel_ready_future", side_effect=grpc.RpcError())
    def test_connected_exp(self, _):
        assert not self.client.connected()


class TestDisconnectException:
    client = GrpcMilvus()

    @mock.patch.object(GrpcMilvus, "connected")
    def test_disconnect_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.disconnect()


class TestCreateTableException:
    client = GrpcMilvus()

    table_param = {
        "table_name": "test001",
        "dimension": 16
    }

    @mock.patch.object(GrpcMilvus, "connected")
    def test_create_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.create_table({})

    def test_create_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status = self.client.create_table(self.table_param)
        assert not status.OK()


class TestHastableException:
    client = GrpcMilvus()

    def test_has_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.has_table("aaa")

    def test_has_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())

        status, _ = self.client.has_table("a123")
        assert not status.OK()

    def test_has_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status, _ = self.client.has_table("a123")
        assert not status.OK()


class TestDeleteTableException:
    client = GrpcMilvus()

    def test_delete_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.delete_table("aaa")

    def test_delete_table_timeout_exp(self, gip):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())

        status = self.client.delete_table("a123")
        assert not status.OK()

    def test_delete_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status = self.client.delete_table("a123")
        assert not status.OK()


class TestCreateIndexException:
    client = GrpcMilvus()

    def test_create_index_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.create_index("aaa")

    def test_create_index_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())

        status = self.client.create_index("a123", timeout=10)
        assert not status.OK()

    def test_create_index_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status = self.client.create_index("a123", timeout=10)
        assert not status.OK()


class TestAddVectorsException:
    client = GrpcMilvus()

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    def test_add_vectors_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.add_vectors(self.table_name, self.records)

    def test_add_vectors_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())

        status, _ = self.client.add_vectors(self.table_name, self.records)
        assert not status.OK()

    def test_add_vectors_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status, _ = self.client.add_vectors(self.table_name, self.records)
        assert not status.OK()


class TestSearchVectorsException:
    client = GrpcMilvus()

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    def test_search_vectors_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.search_vectors(self.table_name, 1, 1, self.records)

    def test_search_vectors_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status, _ = self.client.search(self.table_name, 1, 1, self.records)
        assert not status.OK()


class TestSearchVectorsInFilesException:
    client = GrpcMilvus()

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    def test_search_vectors_in_files_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.search_vectors_in_files(self.table_name, ["1"], self.records, 1)

    def test_search_vectors_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status, _ = \
            self.client.search_in_files(self.table_name,
                                        ["1"], self.records,
                                        1)
        assert not status.OK()


class TestDescribeTableException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_describe_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.describe_table(self.table_name)

    def test_describe_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())

        status, _ = self.client.describe_table(self.table_name)
        assert not status.OK()

    def test_describe_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status, _ = self.client.describe_table(self.table_name)
        assert not status.OK()


class TestShowTableException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_show_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.show_tables()

    def test_show_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())

        status, _ = self.client.show_tables()
        assert not status.OK()

    def test_show_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())

        status, _ = self.client.show_tables()
        assert not status.OK()


class TestGetTableRowCountException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_get_table_row_count_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.get_table_row_count(self.table_name)

    def test_get_table_row_count_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())
        status, _ = self.client.get_table_row_count(self.table_name)
        assert not status.OK()

    def test_sget_table_row_count_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())
        status, _ = self.client.get_table_row_count(self.table_name)
        assert not status.OK()


class TestCmdException:
    client = GrpcMilvus()
    cmd_str = "OK"

    def test__cmd_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client._cmd(self.cmd_str)

    def test__cmd_count_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())
        status, _ = self.client._cmd(self.cmd_str)
        assert not status.OK()

    def test__cmd_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())
        status, _ = self.client._cmd(self.cmd_str)
        assert not status.OK()


class TestPreloadTableException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_preload_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.preload_table(self.table_name)

    def test_preload_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())
        status = self.client.preload_table(self.table_name)
        assert not status.OK()

    def test_preload_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())
        status = self.client.preload_table(self.table_name)
        assert not status.OK()


class TestDescribeIndexException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_describe_index_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.describe_index(self.table_name)

    def test_desribe_index_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())
        status, _ = self.client.describe_index(self.table_name)
        assert not status.OK()

    def test_describe_index_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())
        status, _ = self.client.describe_index(self.table_name)
        assert not status.OK()


class TestDropIndexException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_drop_index_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.drop_index(self.table_name)

    def test_drop_index_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=FError())
        status = self.client.drop_index(self.table_name)
        assert not status.OK()

    def test_drop_index_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        self.client._stub = mock.Mock(side_effect=RpcTestError())
        status = self.client.drop_index(self.table_name)
        assert not status.OK()
