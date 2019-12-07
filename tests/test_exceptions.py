import random

import sys

sys.path.append(".")

import grpc
import pytest
import mock

from grpc import FutureTimeoutError as FError

from milvus.client.grpc_client import GrpcMilvus
from milvus.client.exceptions import NotConnectError


class RpcTestError(grpc.RpcError):

    def code(self):
        return 10

    def details(self):
        return "test"
    
    
class FakeFuture:
    def future(self, request):
        pass
    

class FakeStub:
    def CreateTable(self):
        pass
    
    def HasTable(self):
        pass
    
    def DescribeTable(self):
        pass
    
    def DropTable(self):
        pass
    
    def CountTable(self):
        pass
    
    def ShowTables(self):
        pass
    
    def PreloadTable(self):
        pass

    def Insert(self):
        pass
    
    def CreateIndex(self):
        pass

    def DescribeIndex(self):
        pass

    def DropIndex(self):
        pass

    def CreatePartition(self):
        pass
    
    def ShowPartitions(self):
        pass
    
    def DropPartition(self):
        pass

    def Search(self):
        pass

    def SearchInFiles(self):
        pass

    def DeleteByDate(self):
        pass

    def Cmd(self):
        pass


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

    def test_create_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.CreateTable = future

        self.client._stub = stub

        status = self.client.create_table(self.table_param)
        assert not status.OK()

    def test_create_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.CreateTable = future

        self.client._stub = stub

        status = self.client.create_table(self.table_param)
        assert not status.OK()


class TestHasTableException:
    client = GrpcMilvus()

    def test_has_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.has_table("aaa")

    def test_has_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.HasTable = future
        self.client._stub = stub

        status, _ = self.client.has_table("a123")
        assert not status.OK()

    def test_has_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.HasTable = future

        self.client._stub = stub 

        status, _ = self.client.has_table("a123")
        assert not status.OK()


class TestDropTableException:
    client = GrpcMilvus()

    def test_drop_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.drop_table("aaa")

    def test_drop_table_timeout_exp(self, gip):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.DropTable = future
        self.client._stub = stub

        status = self.client.drop_table("a123")
        assert not status.OK()

    def test_drop_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.DropTable = future
        self.client._stub = stub

        status = self.client.drop_table("a123")
        assert not status.OK()


class TestCreateIndexException:
    client = GrpcMilvus()

    def test_create_index_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.create_index("aaa")

    def test_create_index_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.CreateIndex = future

        self.client._stub = stub

        status = self.client.create_index("a123", timeout=10)
        assert not status.OK()

    def test_create_index_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.CreateIndex = future

        self.client._stub = stub

        status = self.client.create_index("a123", timeout=10)
        assert not status.OK()


class TestInsertException:
    client = GrpcMilvus()

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    def test_insert_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.insert(self.table_name, self.records)

    def test_insert_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.Insert = future

        self.client._stub = stub

        status, _ = self.client.insert(self.table_name, self.records, timeout=1)
        assert not status.OK()

    def test_insert_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.Insert = future

        self.client._stub = stub

        status, _ = self.client.insert(self.table_name, self.records, timeout=1)
        assert not status.OK()


class TestSearchVectorsException:
    client = GrpcMilvus()

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    def test_search_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.search(self.table_name, 1, 1, self.records)

    def test_search_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        stub.Search = mock.Mock(side_effect=RpcTestError())
        self.client._stub = stub

        status, _ = self.client.search(self.table_name, 1, 1, self.records)
        assert not status.OK()


class TestSearchInFilesException:
    client = GrpcMilvus()

    table_name = "aaa"
    records = [[random.random() for _ in range(16)] for _ in range(10)]

    def test_search_in_files_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)

        with pytest.raises(NotConnectError):
            self.client.search_in_files(self.table_name, ["1"], self.records, 1)

    def test_search_in_files_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        stub.SearchInFiles = mock.Mock(side_effect=RpcTestError())
        self.client._stub = stub

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
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.DescribeTable = future
        self.client._stub = stub

        status, _ = self.client.describe_table(self.table_name)
        assert not status.OK()

    def test_describe_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.DescribeTable = future
        self.client._stub = stub

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
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.ShowTables = future
        self.client._stub = stub

        status, _ = self.client.show_tables()
        assert not status.OK()

    def test_show_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.ShowTables = future
        self.client._stub = stub

        status, _ = self.client.show_tables()
        assert not status.OK()


class TestCountTableException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_count_table_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.count_table(self.table_name)

    def test_count_table_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.CountTable = future
        self.client._stub = stub

        status, _ = self.client.count_table(self.table_name)
        assert not status.OK()

    def test_count_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.CountTable = future
        self.client._stub = stub

        status, _ = self.client.count_table(self.table_name)
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
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.Cmd = future
        self.client._stub = stub

        status, _ = self.client._cmd(self.cmd_str)
        assert not status.OK()

    def test__cmd_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.Cmd = future
        self.client._stub = stub

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
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.PreloadTable = future
        self.client._stub = stub

        status = self.client.preload_table(self.table_name)
        assert not status.OK()

    def test_preload_table_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.PreloadTable = future
        self.client._stub = stub

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
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.DescribeIndex = future
        self.client._stub = stub

        status, _ = self.client.describe_index(self.table_name)
        assert not status.OK()

    def test_describe_index_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.DescribeIndex = future
        self.client._stub = stub

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
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.DropIndex = future
        self.client._stub = stub

        status = self.client.drop_index(self.table_name)
        assert not status.OK()

    def test_drop_index_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.DropIndex = future
        self.client._stub = stub

        status = self.client.drop_index(self.table_name)
        assert not status.OK()


# TODO: remove in the future
class TestDeleteByRangeException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_delete_by_range_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client._GrpcMilvus__delete_vectors_by_range(
                self.table_name, start_date="2010-01-01", end_date="2099-12-31")

    def test_delete_by_range_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.DeleteByDate = future
        self.client._stub = stub

        status = self.client._GrpcMilvus__delete_vectors_by_range(
            self.table_name, start_date="2010-01-01", end_date="2099-12-31")
        assert not status.OK()

    def test_delete_by_range_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.DeleteByDate = future
        self.client._stub = stub

        status = self.client._GrpcMilvus__delete_vectors_by_range(
            self.table_name, start_date="2010-01-01", end_date="2099-12-31")
        assert not status.OK()


# partition exception
class TestCreatePartitionException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_create_partition_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.create_partition(self.table_name, "par", "tag")

    def test_create_partition_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.CreatePartition = future
        self.client._stub = stub

        status = self.client.create_partition(self.table_name, "par", 'tag')
        assert not status.OK()

    def test_create_partition_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.CreatePartition = future
        self.client._stub = stub

        status = self.client.create_partition(self.table_name, "par", "tag")
        assert not status.OK()


class TestShowPartitionsException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_show_partition_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.show_partitions(self.table_name, "tag")

    def test_show_partition_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.ShowPartitions = future
        self.client._stub = stub

        status, _ = self.client.show_partitions(self.table_name, 'tag')
        assert not status.OK()

    def test_show_partition_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.ShowPartitions = future
        self.client._stub = stub

        status, _ = self.client.show_partitions(self.table_name, "tag")
        assert not status.OK()


class TestDropPartitionException:
    client = GrpcMilvus()
    table_name = "test_table_name"

    def test_drop_partition_not_connect_exp(self):
        self.client.connected = mock.Mock(return_value=False)
        with pytest.raises(NotConnectError):
            self.client.drop_partition(self.table_name, "tag")

    def test_drop_partition_timeout_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=FError())
        stub.DropPartition = future
        self.client._stub = stub

        status = self.client.drop_partition(self.table_name, 'tag')
        assert not status.OK()

    def test_drop_partition_rpcerror_exp(self):
        self.client.connected = mock.Mock(return_value=True)
        stub = FakeStub()
        future = FakeFuture()
        future.future = mock.Mock(side_effect=RpcTestError())
        stub.DropPartition = future
        self.client._stub = stub

        status = self.client.drop_partition(self.table_name, "tag")
        assert not status.OK()


class TestWithException:

    @mock.patch("grpc.channel_ready_future", side_effect=FError())
    def test_with_timeout_error(self, _):
        with pytest.raises(NotConnectError):
            with GrpcMilvus():
                pass

    @mock.patch("grpc.channel_ready_future", side_effect=RpcTestError())
    def test_with_rpc_error(self, _):
        with pytest.raises(NotConnectError):
            with GrpcMilvus():
                pass

    @mock.patch("grpc.channel_ready_future", side_effect=ValueError())
    def test_with_unkonwn_error(self, _):
        with pytest.raises(NotConnectError):
            with GrpcMilvus():
                pass
