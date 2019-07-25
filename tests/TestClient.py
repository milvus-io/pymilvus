import logging
import pytest
import mock
import faker
import random
import struct
import sys
from faker.providers import BaseProvider
from thrift.transport.TSocket import TSocket
from thrift.transport import TTransport

sys.path.append('.')
from milvus.client.Abstract import TableSchema
from milvus.client.Client import Milvus, IndexType, Status, TopKQueryResult, Prepare
from milvus.client.Exceptions import (
    NotConnectError,
    RepeatingConnectError,
    DisconnectNotConnectedClientError,
    ParamError)
from milvus.thrift import ttypes, MilvusService

LOGGER = logging.getLogger(__name__)


def gen_one_binary(topk):
    ids = [random.randrange(10000000, 99999999) for _ in range(topk)]
    distances = [random.random() for _ in range(topk)]
    return ttypes.TopKQueryBinResult(struct.pack(str(topk) + 'l', *ids), struct.pack(str(topk) + 'd', *distances))


def gen_nq_binaries(nq, topk):
    return [gen_one_binary(topk) for _ in range(nq)]


def fake_query_bin_result(nq, topk):
    return gen_nq_binaries(nq, topk)


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
        'start_date': '2019-06-25',
        'end_date': '2019-06-25'
    }
    return Prepare.range(**param)


def ranges_factory():
    return [range_factory() for _ in range(5)]


def table_schema_factory():
    param = {
        'table_name': fake.table_name(),
        'dimension': random.randint(0, 999),
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    }
    return param


def records_factory(dimension):
    return [[random.random() for _ in range(dimension)] for _ in range(20)]


def query_ranges_factory():
    param = [('2019-06-25', '2019-06-25')]
    return Prepare.ranges(param)


class TestConnection:
    param = {'host': 'localhost', 'port': '9898'}

    @mock.patch.object(TSocket, 'open')
    @mock.patch.object(Milvus, 'connected')
    def test_true_connect(self, open, connected):
        open.return_value = None
        connected.return_value = True
        cnn = Milvus()

        cnn.connect(**self.param)
        assert cnn.server_address == 'localhost:9898'
        assert cnn.status.OK

        _ = cnn.connect(**self.param)
        status = cnn.connect()
        assert status == Status.CONNECT_FAILED


    def test_false_connect(self):
        cnn = Milvus()
        with pytest.raises(NotConnectError):
            cnn.connect(**self.param, timeout=100)
            LOGGER.error(cnn.status)
            assert not cnn.status.OK()

    @mock.patch.object(TSocket, 'open')
    def test_uri(self, open):
        open.return_value = None
        cnn = Milvus()

        cnn.connect(uri='tcp://127.0.0.1:19530')
        assert cnn.status.OK()

    def test_connect(self):
        cnn = Milvus()
        with pytest.raises(NotConnectError):
            cnn.connect('126.0.0.2', timeout=100)
            assert not cnn.status.OK()

            cnn.connect('127.0.0.1', '9999', timeout=100)
            assert not cnn.status.OK()

            cnn.connect(port='9999', timeout=100)
            assert not cnn.status.OK()

            cnn.connect(uri='cp://127.0.0.1:19530', timeout=100)
            assert not cnn.status.OK()

    def test_connect_timeout(self):
        cnn = Milvus()
        with pytest.raises(NotConnectError):
            cnn.connect(host='123.0.0.2', port='19530', timeout=100)

    def test_connected(self):
        cnn = Milvus()
        with pytest.raises(NotConnectError):
            cnn.connect(host='123.0.0.2', timeout=100)
        assert not cnn.connected()

    @mock.patch.object(TSocket, 'open')
    def test_uri_runtime_error(self, open):
        open.return_value = None
        cnn = Milvus()
        with pytest.raises(RuntimeError):
            cnn.connect(uri='http://127.0.0.1:19530')

        cnn.connect()
        assert cnn.status.OK()

    @mock.patch.object(TSocket, 'close')
    @mock.patch.object(TSocket, 'isOpen')
    @mock.patch.object(TSocket, 'open')
    def test_disconnected(self, close, isOpen, open):
        close.return_value = None
        open.return_value = None
        isOpen.return_value = True

        cnn = Milvus()
        cnn.connect(**self.param)

        assert cnn.disconnect().OK()

    def test_disconnected_error(self):
        cnn = Milvus()
        cnn.status = Status(Status.PERMISSION_DENIED)
        with pytest.raises(DisconnectNotConnectedClientError):
            cnn.disconnect()


class TestTable:

    @pytest.fixture
    @mock.patch.object(TSocket, 'open')
    def client(self, open):
        param = {'host': 'localhost', 'port': '5123'}
        open.return_value = None

        cnn = Milvus()
        cnn.connect(**param)
        return cnn

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'CreateTable')
    def test_create_table(self, CreateTable, connected, client):
        CreateTable.return_value = None
        connected.return_value = True

        param = table_schema_factory()
        param['table_name'] = None
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = table_schema_factory()
        res = client.create_table(param)
        assert res.OK()

        param['index_type'] = 'string'
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = table_schema_factory()
        param['dimension'] = 'string'
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = '09998876565'
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = table_schema_factory()
        param['dimension'] = 0
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = table_schema_factory()
        param['dimension'] = 1000000
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = table_schema_factory()
        param['index_type'] = IndexType.INVALID
        with pytest.raises(ParamError):
            res = client.create_table(param)

        param = table_schema_factory()
        param['table_name'] = 1234456
        res = client.create_table(param)
        assert res.OK()

        param = table_schema_factory()
        param['index_type'] = IndexType.IVF_SQ8
        res = client.create_table(param)
        assert res.OK()

    def test_create_table_connect_failed_status(self, client):
        param = table_schema_factory()
        with pytest.raises(NotConnectError):
            res = client.create_table(param)
            assert res == Status.CONNECT_FAILED

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'DeleteTable')
    def test_delete_table(self, DeleteTable, connected, client):
        DeleteTable.return_value = None
        connected.return_value = True
        table_name = 'fake_table_name'
        res = client.delete_table(table_name)
        assert res.OK

    def test_false_delete_table(self, client):
        table_name = 'fake_table_name'
        with pytest.raises(NotConnectError):
            res = client.delete_table(table_name)
            LOGGER.info(res)
            assert res == Status.CONNECT_FAILED

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'HasTable')
    def test_has_table(self, HasTable, connected, client):
        connected.return_value = True
        HasTable.return_value = True

        table_name = fake.table_name()
        result = client.has_table(table_name)
        assert result

        HasTable.return_value = False
        result = client.has_table(table_name)
        assert not result

class TestVector:

    @pytest.fixture
    @mock.patch.object(TSocket, 'open')
    def client(self, open):
        param = {'host': 'localhost', 'port': '5123'}
        open.return_value = None

        cnn = Milvus()
        cnn.connect(**param)
        return cnn

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'AddVector')
    def test_add_vector(self, AddVector, connected, client):
        connected.return_value = True
        AddVector.return_value = ['a', 'a']

        param = {
            'table_name': fake.table_name(),
            'records': records_factory(256)
        }
        res, ids = client.add_vectors(**param)
        assert res.OK()
        assert isinstance(ids, list)

        param['records'] = [[]]
        with pytest.raises(ParamError):
            res, ids = client.add_vectors(**param)

        param['records'] = [['string']]
        with pytest.raises(ParamError):
            res, ids = client.add_vectors(**param)

    def test_false_add_vector(self, client):
        param = {
            'table_name': fake.table_name(),
            'records': records_factory(256)
        }
        with pytest.raises(NotConnectError):
            res, ids = client.add_vectors(**param)
            assert res == Status.CONNECT_FAILED

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'SearchVector2')
    def test_search_vector(self, SearchVector2, connected, client):
        topk = random.randint(1, 10)
        query_records = records_factory(256)
        SearchVector2.return_value = fake_query_bin_result(1, topk)
        connected.return_value = True
        param = {
            'table_name': fake.table_name(),
            'query_records': query_records,
            'top_k': topk
        }
        res, results = client.search_vectors(**param)
        assert res.OK()
        assert isinstance(results, (list, TopKQueryResult))

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'SearchVector2')
    def test_search_vector_with_range(self, SearchVector2, connected, client):
        topk = random.randint(1, 10)
        query_records = records_factory(256)
        SearchVector2.return_value = fake_query_bin_result(1, topk)
        connected.return_value = True
        param = {
            'table_name': fake.table_name(),
            'query_records': query_records,
            'top_k': topk,
            'query_ranges': query_ranges_factory()

        }
        res, results = client.search_vectors(**param)
        assert res.OK()
        assert isinstance(results, (list, TopKQueryResult))

    @mock.patch.object(Milvus, 'connected')
    def test_false_vector(self, connected, client):
        connected.return_value = True
        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'top_k': random.randint(1, 10)
        }
        with pytest.raises(NotConnectError):
            res, results = client.search_vectors(**param)
            assert res == Status.CONNECT_FAILED

        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'top_k': 'string'
        }
        with pytest.raises(ParamError):
            res, results = client.search_vectors(**param)

        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'top_k': 'string'
        }
        with pytest.raises(ParamError):
            res, results = client.search_vectors(**param)

        param = {'table_name': fake.table_name(),
                 'query_records': records_factory(256),
                 'top_k': random.randint(1, 10),
                 'query_ranges': ['false_date_format']}
        with pytest.raises(ParamError):
            res, results = client.search_vectors(**param)

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'SearchVectorInFiles')
    def test_search_in_files(self, SearchVectorInFiles, connected, client):
        SearchVectorInFiles.return_value = [ttypes.TopKQueryResult([ttypes.QueryResult(111, 111)])]
        connected.return_value = True

        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'file_ids': ['a'],
            'top_k': random.randint(1, 10)
        }
        sta, result = client.search_vectors_in_files(**param)
        assert sta.OK()

    def test_false_search_in_files(self, client):
        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(256),
            'file_ids': ['a'],
            'top_k': random.randint(1, 10)
        }
        with pytest.raises(NotConnectError):
            sta, results = client.search_vectors_in_files(**param)
            assert sta == Status.CONNECT_FAILED

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'DescribeTable')
    def test_describe_table(self, DescribeTable, connected, client):
        DescribeTable.return_value = ttypes.TableSchema(**table_schema_factory())
        connected.return_value = True

        table_name = fake.table_name()
        res, table_schema = client.describe_table(table_name)
        assert res.OK()
        assert isinstance(table_schema, TableSchema)

    def test_false_decribe_table(self, client):
        table_name = fake.table_name()
        with pytest.raises(NotConnectError):
            res, table_schema = client.describe_table(table_name)
            assert not res.OK()
            assert not table_schema

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'ShowTables')
    def test_show_tables(self, ShowTables, connected, client):
        ShowTables.return_value = [fake.table_name() for _ in range(10)]
        connected.return_value = True
        res, tables = client.show_tables()
        assert res.OK()
        assert isinstance(tables, list)

    def test_false_show_tables(self, client):
        with pytest.raises(NotConnectError):
            res, tables = client.show_tables()
            assert not res.OK()
            assert not tables

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'GetTableRowCount')
    def test_get_table_row_count(self, GetTableRowCount, connected, client):
        GetTableRowCount.return_value = 22, None
        connected.return_value = True
        res, count = client.get_table_row_count('fake_table')
        assert res.OK()

    def test_false_get_table_row_count(self, client):
        with pytest.raises(NotConnectError):
            res, count = client.get_table_row_count('fake_table')
            assert not res.OK()
            assert not count

    def test_client_version(self, client):
        res = client.client_version()
        assert isinstance(res, str)

    @mock.patch.object(Milvus, 'connected')
    @mock.patch.object(MilvusService.Client, 'Ping')
    def test_server_status(self, Ping, connected, client):
        Ping.return_value = '00'
        connected.return_value = True
        status, res = client.server_status()
        assert status.OK()
        assert res == 'OK'

        status, res = client.server_status('abc')
        assert status.OK()
        assert res == 'OK'

        Ping.return_value = '0.0.0'
        status, res = client.server_status('version')
        assert status.OK()
        assert res == '0.0.0'


class TestPrepare:

    def test_table_schema(self):
        param = {
            'table_name': fake.table_name(),
            'dimension': random.randint(0, 999),
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }
        res = Prepare.table_schema(param)
        assert isinstance(res, ttypes.TableSchema)

    def test_double_schema(self):
        param = {
            'table_name': fake.table_name(),
            'dimension': random.randint(0, 999),
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }
        res = Prepare.table_schema(param)
        a = Prepare.table_schema(res)
        assert a == res
        assert isinstance(a, ttypes.TableSchema)
        assert isinstance(res, ttypes.TableSchema)

    def test_range(self):
        param = {'start_date': '2019-02-02', 'end_date': '2019-02-02'}
        res = Prepare.range(**param)
        assert isinstance(res, ttypes.Range)

        param['start_date'] = '2019-09-02'
        with pytest.raises(ParamError):
            res = Prepare.range(**param)

    def test_ranges(self):
        param = [('2019-02-02', '2019-02-02')]
        res = Prepare.ranges(param)
        assert isinstance(res, list)
        assert isinstance(res[0], ttypes.Range)

        param = ['false_date_format']
        with pytest.raises(ParamError):
            res = Prepare.ranges(param)

    def test_repeating_range(self):
        param = [('2019-02-02', '2019-02-02')]
        res = Prepare.ranges(param)
        a = Prepare.ranges(res)
        assert isinstance(a[0], ttypes.Range)
        assert a == res

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

    def test_repeating_prepare(self):
        vecs = [[random.random() for _ in range(256)] for _ in range(20)]
        vecs = Prepare.records(vecs)
        vecs = Prepare.records(vecs)
        assert isinstance(vecs, list)
