import random

import sys

sys.path.append(".")

import grpc
import pytest
import mock

from grpc import FutureTimeoutError as FError
from grpc._channel import _UnaryUnaryMultiCallable as FC

from milvus import Status
from milvus.client.grpc_client import GrpcMilvus
from milvus.client.exceptions import NotConnectError

from factorys import get_last_day, get_next_day


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
    def test_connected_exp(self, mock_channel):
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
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_param = {
        "table_name": "test001",
        "dimension": 16
    }

    @mock.patch.object(GrpcMilvus, "connected")
    def test_create_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.disconnect()

    @mock.patch.object(FC, "future")
    def test_create_table_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status = self.client.create_table(self.table_param)
        assert not status.OK()


class TestHastableException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    @mock.patch.object(GrpcMilvus, "connected")
    def test_has_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.has_table("aaa")

    @mock.patch.object(FC, "future")
    def test_has_table_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client.has_table("a123")
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_has_table_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.has_table("a123")
        assert not status.OK()


class TestDeleteTableException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    @mock.patch.object(GrpcMilvus, "connected")
    def test_delete_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.delete_table("aaa")

    @mock.patch.object(FC, "future")
    def test_delete_table_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status = self.client.delete_table("a123")
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_delete_table_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status = self.client.delete_table("a123")
        assert not status.OK()


class TestCreateIndexException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    @mock.patch.object(GrpcMilvus, "connected")
    def test_create_index_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.create_index("aaa")

    @mock.patch.object(FC, "future")
    def test_create_index_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status = self.client.create_index("a123", timeout=10)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_create_index_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status = self.client.create_index("a123", timeout=10)
        assert not status.OK()


class TestAddVectorsException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    @mock.patch.object(GrpcMilvus, "connected")
    def test_add_vectors_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.add_vectors(self.table_name, self.records)

    @mock.patch.object(FC, "future")
    def test_add_vectors_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client.add_vectors(self.table_name, self.records)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_add_vectors_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.add_vectors(self.table_name, self.records)
        assert not status.OK()


class TestSearchVectorsException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    @mock.patch.object(GrpcMilvus, "connected")
    def test_search_vectors_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.search_vectors(self.table_name, 1, 1, self.records)

    @mock.patch.object(FC, "__call__")
    def test_search_vectors_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.search(self.table_name, 1, 1, self.records)
        assert not status.OK()


class TestSearchVectorsInFilesException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    @mock.patch.object(GrpcMilvus, "connected")
    def test_search_vectors_in_files_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.search_vectors_in_files(self.table_name, ["1"], self.records, 1)

    @mock.patch.object(FC, "__call__")
    def test_search_vectors_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = \
            self.client.search_in_files(self.table_name,
                                        ["1"], self.records,
                                        1)
        assert not status.OK()


class TestDescribeTableException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "test_table_name"

    @mock.patch.object(GrpcMilvus, "connected")
    def test_describe_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.describe_table(self.table_name)

    @mock.patch.object(FC, "future")
    def test_describe_table_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client.describe_table(self.table_name)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_describe_table_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.describe_table(self.table_name)
        assert not status.OK()


class TestShowTableException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "test_table_name"

    @mock.patch.object(GrpcMilvus, "connected")
    def test_show_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.show_tables()

    @mock.patch.object(FC, "future")
    def test_show_table_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client.show_tables()
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_show_table_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.show_tables()
        assert not status.OK()


class TestGetTableRowCountException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "test_table_name"

    @mock.patch.object(GrpcMilvus, "connected")
    def test_get_table_row_count_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.get_table_row_count(self.table_name)

    @mock.patch.object(FC, "future")
    def test_get_table_row_count_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client.get_table_row_count(self.table_name)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_sget_table_row_count_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.get_table_row_count(self.table_name)
        assert not status.OK()


class TestCmdException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    cmd_str = "OK"

    @mock.patch.object(GrpcMilvus, "connected")
    def test__cmd_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client._cmd(self.cmd_str)

    @mock.patch.object(FC, "future")
    def test__cmd_count_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client._cmd(self.cmd_str)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test__cmd_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client._cmd(self.cmd_str)
        assert not status.OK()


class TestPreloadTableException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "test_table_name"

    @mock.patch.object(GrpcMilvus, "connected")
    def test_preload_table_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.preload_table(self.table_name)

    @mock.patch.object(FC, "future")
    def test_preload_table_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status = self.client.preload_table(self.table_name)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_preload_table_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status = self.client.preload_table(self.table_name)
        assert not status.OK()


class TestDescribeIndexException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "test_table_name"

    @mock.patch.object(GrpcMilvus, "connected")
    def test_describe_index_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.describe_index(self.table_name)

    @mock.patch.object(FC, "future")
    def test_desribe_index_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status, _ = self.client.describe_index(self.table_name)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_describe_index_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status, _ = self.client.describe_index(self.table_name)
        assert not status.OK()


class TestDropIndexException:
    client = GrpcMilvus()
    client.server_status = mock.Mock(return_value=(Status(), "OK"))

    table_name = "test_table_name"

    @mock.patch.object(GrpcMilvus, "connected")
    def test_drop_index_not_connect_exp(self, mock_client):
        mock_client.return_value = False

        with pytest.raises(NotConnectError):
            self.client.drop_index(self.table_name)

    @mock.patch.object(FC, "future")
    def test_drop_index_timeout_exp(self, mock_callable, gip):
        mock_callable.side_effect = FError()

        self.client.connect(*gip)

        status = self.client.drop_index(self.table_name)
        assert not status.OK()

    @mock.patch.object(FC, "future")
    def test_drop_index_rpcerror_exp(self, mock_callable, gip):
        mock_callable.side_effect = RpcTestError()

        self.client.connect(*gip)

        status = self.client.drop_index(self.table_name)
        assert not status.OK()
