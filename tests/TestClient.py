import logging
import pytest
import mock
import faker
import random
import sys

from faker.providers import BaseProvider
from milvus import Milvus, Prepare, IndexType, Status

from milvus.client.Exceptions import (
    RepeatingConnectError,
    DisconnectNotConnectedClientError)
from milvus.thrift import ttypes, MilvusService

from thrift.transport.TSocket import TSocket
from thrift.transport import TTransport

LOGGER = logging.getLogger(__name__)
sys.path.append('.')


class FakerProvider(BaseProvider):

    def table_name(self):
        return 'table_name' + str(random.randint(1000, 9999))

    def name(self):
        return 'name' + str(random.randint(1000, 9999))

    def dim(self):
        return random.randint(0, 999)


fake = faker.Faker()
fake.add_provider(FakerProvider)


def range_factory():
    param = {
        'start': str(random.randint(1, 10)),
        'end': str(random.randint(11, 20)),
    }
    return Prepare.range(**param)


def ranges_factory():
    return [range_factory() for _ in range(5)]


def table_schema_factory():
    param = {
        'table_name': fake.table_name(),
        'dimension': random.randint(0, 999),
        'index_type': IndexType.IDMAP,
        'store_raw_vector': False
    }
    return Prepare.table_schema(**param)


def records_factory(dimension):
    return Prepare.records([[random.random() for _ in range(dimension)] for _ in range(20)])


class TestConnection:
    param = {'host': 'localhost', 'port': '5000'}

    @mock.patch.object(TSocket, 'open')
    def test_true_connect(self, open):
        open.return_value = None
        cnn = Milvus()

        cnn.connect(**self.param)
        assert cnn.status == Status.SUCCESS
        assert cnn.connected

        with pytest.raises(RepeatingConnectError):
            cnn.connect(**self.param)
            cnn.connect()

    def test_false_connect(self):
        cnn = Milvus()

        cnn.connect(**self.param)
        assert cnn.status != Status.SUCCESS

    @mock.patch.object(TSocket, 'open')
    def test_uri(self, open):
        open.return_value = None
        cnn = Milvus()

        cnn.connect(uri='tcp://127.0.0.1:9090')
        assert cnn.status == Status.SUCCESS

    @mock.patch.object(TSocket, 'open')
    def test_uri_runtime_error(self, open):
        open.return_value = None
        cnn = Milvus()
        with pytest.raises(RuntimeError):
            cnn.connect(uri='http://127.0.0.1:9090')

        cnn.connect()
        assert cnn.status == Status.SUCCESS

    @mock.patch.object(TTransport.TBufferedTransport, 'close')
    @mock.patch.object(TSocket, 'open')
    def test_disconnected(self, close, open):
        close.return_value = None
        open.return_value = None

        cnn = Milvus()
        cnn.connect(**self.param)

        assert cnn.disconnect() == Status.SUCCESS

    def test_disconnected_error(self):
        cnn = Milvus()
        cnn.connect_status = Status(Status.PERMISSION_DENIED)
        with pytest.raises(DisconnectNotConnectedClientError):
            cnn.disconnect()


class TestTable:

    @pytest.fixture
    @mock.patch.object(TSocket, 'open')
    def client(self, open):
        param = {'host': 'localhost', 'port': '5000'}
        open.return_value = None

        cnn = Milvus()
        cnn.connect(**param)
        return cnn

    @mock.patch.object(MilvusService.Client, 'CreateTable')
    def test_create_table(self, CreateTable, client):
        CreateTable.return_value = None

        param = table_schema_factory()
        res = client.create_table(param)
        assert res == Status.SUCCESS

    def test_false_create_table(self, client):
        param = table_schema_factory()
        res = client.create_table(param)
        assert res != Status.SUCCESS

    @mock.patch.object(MilvusService.Client, 'DeleteTable')
    def test_delete_table(self, DeleteTable, client):
        DeleteTable.return_value = None
        table_name = 'fake_table_name'
        res = client.delete_table(table_name)
        assert res == Status.SUCCESS

    def test_false_delete_table(self, client):
        table_name = 'fake_table_name'
        res = client.delete_table(table_name)
        assert res != Status.SUCCESS


class TestVector:

    @pytest.fixture
    @mock.patch.object(TSocket, 'open')
    def client(self, open):
        param = {'host': 'localhost', 'port': '5000'}
        open.return_value = None

        cnn = Milvus()
        cnn.connect(**param)
        return cnn

    @mock.patch.object(MilvusService.Client, 'AddVector')
    def test_add_vector(self, AddVector, client):
        AddVector.return_value = None

        param = {
            'table_name': fake.table_name(),
            'records': records_factory(256)
        }
        res, ids = client.add_vectors(**param)
        assert res == Status.SUCCESS

    def test_false_add_vector(self, client):
        param = {
            'table_name': fake.table_name(),
            'records': records_factory(256)
        }
        res, ids = client.add_vectors(**param)
        assert res != Status.SUCCESS

    @mock.patch.object(MilvusService.Client, 'SearchVector')
    def test_search_vector(self, SearchVector, client):
        SearchVector.return_value = None, None
        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'query_ranges': ranges_factory(),
            'top_k': random.randint(0, 10)
        }
        res, results = client.search_vectors(**param)
        assert res == Status.SUCCESS

    def test_false_vector(self, client):
        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'query_ranges': ranges_factory(),
            'top_k': random.randint(0, 10)
        }
        res, results = client.search_vectors(**param)
        assert res != Status.SUCCESS

    @mock.patch.object(MilvusService.Client, 'DescribeTable')
    def test_describe_table(self, DescribeTable, client):
        DescribeTable.return_value = table_schema_factory()

        table_name = fake.table_name()
        res, table_schema = client.describe_table(table_name)
        assert res == Status.SUCCESS
        assert isinstance(table_schema, ttypes.TableSchema)

    def test_false_decribe_table(self, client):
        table_name = fake.table_name()
        res, table_schema = client.describe_table(table_name)
        assert res != Status.SUCCESS
        assert not table_schema

    @mock.patch.object(MilvusService.Client, 'ShowTables')
    def test_show_tables(self, ShowTables, client):
        ShowTables.return_value = [fake.table_name() for _ in range(10)]
        res, tables = client.show_tables()
        assert res == Status.SUCCESS
        assert isinstance(tables, list)

    def test_false_show_tables(self, client):
        res, tables = client.show_tables()
        assert res != Status.SUCCESS
        assert not tables

    @mock.patch.object(MilvusService.Client, 'GetTableRowCount')
    def test_get_table_row_count(self, GetTableRowCount, client):
        GetTableRowCount.return_value = 22, None
        res, count = client.get_table_row_count('fake_table')
        assert res == Status.SUCCESS

    def test_false_get_table_row_count(self, client):
        res, count = client.get_table_row_count('fake_table')
        assert res != Status.SUCCESS
        assert not count

    def test_client_version(self, client):
        res = client.client_version()
        assert isinstance(res, str)


class TestPrepare:

    def test_table_schema(self):
        param = {
            'table_name': fake.table_name(),
            'dimension': random.randint(0, 999),
            'index_type': IndexType.IDMAP,
            'store_raw_vector': False
        }
        res = Prepare.table_schema(**param)
        assert isinstance(res, ttypes.TableSchema)

    def test_range(self):
        param = {
            'start': '200',
            'end': '1000'
        }

        res = Prepare.range(**param)
        assert isinstance(res, ttypes.Range)
        assert res.start_value == '200'
        assert res.end_value == '1000'

    def test_row_record(self):
        vec = [random.random() + random.randint(0, 9) for _ in range(256)]
        res = Prepare.row_record(vec)
        assert isinstance(res, ttypes.RowRecord)
        assert isinstance(res.vector_data, bytes)

    def test_records(self):
        vecs = [[random.random() for _ in range(256)] for _ in range(20)]
        res = Prepare.records(vecs)
        assert isinstance(res, list)
        assert isinstance(res[0], ttypes.RowRecord)
        assert isinstance(res[0].vector_data, bytes)
