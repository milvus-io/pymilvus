import logging
import time
from milvus.client.GrpcClient import Prepare, GrpcMilvus, Status
from milvus.client.Abstract import IndexType
from factorys import *

LOGGER = logging.getLogger(__name__)

dim = 16

class TestConnection:
    param = {'host': 'localhost', 'port': '19530'}

    def test_true_connect(self):
        cnn = GrpcMilvus()

        cnn.connect(**self.param)
        assert cnn.status.OK
        assert cnn.connected()

        # Repeating connect
        _ = cnn.connect(**self.param)
        status = cnn.connect()
        assert status == Status.CONNECT_FAILED


    def test_false_connect(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect(uri='tcp://127.0.0.1:7987', timeout=100)
            LOGGER.error(cnn.status)
            assert not cnn.status.OK()

    def test_uri(self):
        cnn = GrpcMilvus()
        cnn.connect(uri='tcp://127.0.0.1:19530')
        assert cnn.status.OK()

    def test_connect(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect('126.0.0.2', timeout=100)
            assert not cnn.status.OK()

            cnn.connect('127.0.0.1', '9999', timeout=100)
            assert not cnn.status.OK()

            cnn.connect(port='9999', timeout=100)
            assert not cnn.status.OK()

            cnn.connect(uri='cp://127.0.0.1:19530', timeout=1000)
            assert not cnn.status.OK()

    def test_connect_timeout(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect(host='123.0.0.2', port='19530', timeout=1000)

    def test_connected(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect(host='123.0.0.2', timeout=1000)
        assert not cnn.connected()

    def test_uri_runtime_error(self):
        cnn = GrpcMilvus()
        with pytest.raises(RuntimeError):
            cnn.connect(uri='http://127.0.0.1:19530')

        cnn.connect()
        assert cnn.status.OK()

    def test_disconnected(self):

        cnn = GrpcMilvus()
        cnn.connect(**self.param)

        assert cnn.disconnect().OK()
        assert not cnn.connected()

        cnn.connect(**self.param)
        assert cnn.connected()

    def test_disconnected_error(self):
        cnn = GrpcMilvus()
        with pytest.raises(DisconnectNotConnectedClientError):
            cnn.disconnect()


class TestPing:

    def test_ping_server_version(self):
        milvus = GrpcMilvus()
        milvus.connect()

        _, version = milvus.server_version()
        assert version == '0.3.1'


class TestCreateTable:
    
    def test_create_table_normal(self, gcon):
        param = table_schema_factory()

        status = gcon.create_table(param)
        assert status.OK()

    def test_create_table_name_wrong(self, gcon):
        param = table_schema_factory()
        param['table_name'] = '.....'
        status = gcon.create_table(param)
        LOGGER.error(status)
        assert not status.OK()
        

class TestDescribTable:

    def test_describe_table_normal(self, gcon):
        param = table_schema_factory()
        gcon.create_table(param)

        status, table = gcon.describe_table(param['table_name'])
        assert status.OK()
        assert table.table_name == param['table_name']

        status, table = gcon.describe_table('table_not_exists')
        assert not status.OK()


class TestShowTables:
    def test_show_tables_normal(self, gcon):
        status, tables = gcon.show_tables()
        assert status.OK()


class TestDeleteTable:
    def test_delete_table_normal(self, gcon):
        param = table_schema_factory()
        s = gcon.create_table(param)
        _, tables = gcon.show_tables()
        assert param['table_name'] in tables
        

        status = gcon.delete_table(param['table_name'])
        _, tables = gcon.show_tables()
        assert param['table_name'] not in tables


class TestHasTable:
    def test_has_table(self, gcon):
        param = table_schema_factory()
        s = gcon.create_table(param)

        flag = gcon.has_table(param['table_name'])
        assert flag


class TestTable:
    def test_create_table(self, gcon):

        param = table_schema_factory()
        param['table_name'] = None
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        res = gcon.create_table(param)
        assert res.OK()

        param['index_type'] = 'string'
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        param['dimension'] = 'string'
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = '09998876565'
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        param['dimension'] = 0
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        param['dimension'] = 1000000
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        param['index_type'] = IndexType.INVALID
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        param['table_name'] = 1234456
        res = gcon.create_table(param)
        assert not res.OK()

        param = table_schema_factory()
        param['index_type'] = IndexType.IVF_SQ8
        res = gcon.create_table(param)
        assert res.OK()

    # TODO
    def test_create_table_connect_failed_status(self):
        g = GrpcMilvus()
        g.connect(uri='tcp://127.0.0.1:20202')
        param = table_schema_factory()
        with pytest.raises(Exception):
            res = g.create_table(param)
            LOGGER.error(g._server_address)
            assert res == Status.CONNECT_FAILED

    def test_delete_table(self, gcon, gtable):
        res = gcon.delete_table(gtable)
        assert res.OK
        assert not gcon.has_table(gtable)

    def test_false_delete_table(self, gcon):
        table_name = 'fake_table_name'
        res = gcon.delete_table(table_name)
        assert not res.OK()

    def test_has_table(self, gcon, gtable):

        table_name = fake.table_name()
        result = gcon.has_table(table_name)
        assert not result

        result = gcon.has_table(gtable)
        assert result

class TestAddVectors:
    
    def test_add_vectors_normal(self, gcon, gtable):
        vectors = records_factory(dim)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()
        assert len(ids) == 20

        time.sleep(2)
        
        status, count = gcon.get_table_row_count(gtable)
        assert status.OK()
        assert count == 20


class TestSearchVectors:
    def test_search_vectors_normal_1_with_ranges(self, gcon, gtable):
        vectors = records_factory(dim)
        status, ids = gcon.add_vectors(gtable, vectors)
        
        ranges = ranges_factory()
        time.sleep(2)

        s_vectors = [vectors[0]]

        status, result = gcon.search_vectors(gtable, 1, s_vectors, ranges)
        assert status.OK()
        assert len(result) == 1
        assert len(result[0]) == 1

