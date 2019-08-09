import logging
import time
from multiprocessing import Process, Queue
from milvus.client.GrpcClient import Prepare, GrpcMilvus, Status
from milvus.client.Abstract import IndexType, TableSchema
from factorys import *

LOGGER = logging.getLogger(__name__)

dim = 16
nq = 20

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
            cnn.connect(uri='tcp://127.0.0.1:7987', timeout=1)
            LOGGER.error(cnn.status)
            assert not cnn.status.OK()

    def test_uri(self):
        cnn = GrpcMilvus()
        cnn.connect(uri='tcp://127.0.0.1:19530')
        assert cnn.status.OK()

    def test_connect(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect('126.0.0.2', timeout=2)
            assert not cnn.status.OK()

            cnn.connect('127.0.0.1', '9999', timeout=2)
            assert not cnn.status.OK()

            cnn.connect(port='9999', timeout=2)
            assert not cnn.status.OK()

            cnn.connect(uri='cp://127.0.0.1:19530', timeout=2)
            assert not cnn.status.OK()

    def test_connect_timeout(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect(host='123.0.0.2', port='19530', timeout=2)

    def test_connected(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect(host='123.0.0.2', timeout=1)
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


class TestTable:

    def test_create_table(self, gcon):

        param = table_schema_factory()
        param['table_name'] = None
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = table_schema_factory()
        res = gcon.create_table(param)
        assert res.OK()
        assert gcon.has_table(param['table_name'])

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
        LOGGER.error(res)
        assert not res.OK()

        param = table_schema_factory()
        param['index_type'] = IndexType.IVF_SQ8
        res = gcon.create_table(param)
        assert res.OK()

    def test_delete_table(self, gcon):
        table_name = 'fake_table_name'
        res = gcon.delete_table(table_name)
        assert res.OK

    def test_false_delete_table(self, gcon):
        table_name = 'fake_table_name'
        res = gcon.delete_table(table_name)
        assert not res.OK()

    def test_repeat_add_table(self, gcon):
        param = table_schema_factory()

        res = gcon.create_table(param)

        res = gcon.create_table(param)
        LOGGER.error(res)
        assert not res.OK()

    def test_has_table(self, gcon, gtable):

        table_name = fake.table_name()
        result = gcon.has_table(table_name)
        assert not result

        result = gcon.has_table(gtable)
        assert result

        with pytest.raises(Exception):
            result = gcon.has_table(1111)


class TestVector:

    def test_add_vector(self, gcon, gtable):

        param = {
            'table_name': gtable,
            'records': records_factory(dim, nq)
        }
        res, ids = gcon.add_vectors(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq

        param['records'] = [['string']]
        with pytest.raises(ParamError):
            res, ids = gcon.add_vectors(**param)

    def test_add_vector_with_no_right_dimention(self, gcon, gtable):
        param = {
            'table_name': gtable,
            'records': records_factory(dim + 1, nq)
        }

        res, ids = gcon.add_vectors(**param)
        assert not res.OK()

    def test_add_vector_records_empty_list(self, gcon, gtable):
        param = {
            'table_name': gtable,
            'records': records_factory(dim, nq)
        }
        
        param['records'] = [[]]
        with pytest.raises(Exception):
            res, ids = gcon.add_vectors(**param)


    def test_false_add_vector(self, gcon):
        param = {
            'table_name': fake.table_name(),
            'records': records_factory(dim, nq)
        }
        res, ids = gcon.add_vectors(**param)
        assert not res.OK()

    #@pytest.mark.skip('Not Complete')
    @pytest.mark.timeout(20)
    def test_add_vector_with_multiprocessing(self, gtable):
        '''
        target: test add vectors, with multi processes
        method: 10 processed add vectors concurrently
        expected: status ok and result length is equal to the length off added vectors
        '''
        vectors = gen_single_vector(dim)
        q = Queue()
        process_num = 5
        processes = []
        # with dependent connection
        def _add(q):
            loop_num = 10
            vector = q.get()
            print(vector)
            milvus = GrpcMilvus()
            milvus.connect()

            status, ids = milvus.add_vectors(gtable, vector)
            assert status.OK()
            milvus.disconnect()

        for i in range(process_num):
            q.put(vectors)
            p = Process(target=_add, args=(q,))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        def check(table):
            milvus = GrpcMilvus()
            milvus.connect()

            status, count = milvus.get_table_row_count(gtable)
            assert count == process_num * 10
            milvus.disconnect()

        time.sleep(3)
        check_count = Process(target=check, args=(gtable,))
        check_count.start()
        check_count.join()


class TestSearch:
    def test_search_vector_normal(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector,
            'query_records': query_records,
            'top_k': topk
        }
        res, results = gcon.search_vectors(**param)
        assert res.OK()
        assert isinstance(results, (list, TopKQueryResult))
        assert len(results) == nq
        assert len(results[0]) == topk

    def test_search_vector_wrong_dim(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim + 1, nq)
        param = {
            'table_name': gvector ,
            'query_records': query_records,
            'top_k': topk
        }
        res, results = gcon.search_vectors(**param)
        assert not res.OK()

    def test_search_vector_wrong_table_name(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector + 'wrong',
            'query_records': query_records,
            'top_k': topk
        }
        res, results = gcon.search_vectors(**param)
        assert res == Status.ILLEGAL_ARGUMENT
        assert not res.OK()

    def test_search_vector_with_range(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector,
            'query_records': query_records,
            'top_k': topk,
            'query_ranges': query_ranges_factory()

        }
        res, results = gcon.search_vectors(**param)
        assert res.OK()
        assert isinstance(results, (list, TopKQueryResult))
        assert len(results) == nq
        assert len(results[0]) == topk

    def test_false_vector(self, gcon):

        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(dim, nq),
            'top_k': 'string'
        }
        with pytest.raises(ParamError):
            res, results = gcon.search_vectors(**param)

        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(dim, nq),
            'top_k': 'string'
        }
        with pytest.raises(ParamError):
            res, results = gcon.search_vectors(**param)

        param = {'table_name': fake.table_name(),
                 'query_records': records_factory(dim, nq),
                 'top_k': random.randint(1, 10),
                 'query_ranges': ['false_date_format']}
        with pytest.raises(ParamError):
            res, results = gcon.search_vectors(**param)

    def test_search_in_files(self, gcon, gvector):

        param = {
            'table_name': gvector,
            'query_records': records_factory(dim, nq),
            'file_ids': ['1'],
            'top_k': random.randint(1, 10)
        }
        sta, result = gcon.search_vectors_in_files(**param)
        assert sta.OK()

    @pytest.mark.skip('Not fixed')
    def test_search_in_files_wrong_file_ids(self, gcon, gvector):
        param = {
            'table_name': gvector,
            'query_records': records_factory(dim, nq),
            'file_ids': ['3333'],
            'top_k': random.randint(1, 10)
        }
        sta, results = gcon.search_vectors_in_files(**param)
        assert not sta.OK()


    # TODO search in files wrong dim, wrong dimention name
    def test_describe_table(self, gcon, gtable):

        res, table_schema = gcon.describe_table(gtable)
        assert res.OK()
        assert isinstance(table_schema, TableSchema)

    def test_false_decribe_table(self, gcon):
        table_name = fake.table_name()
        res, table_schema = gcon.describe_table(table_name)
        assert not res.OK()
        assert not table_schema

    def test_show_tables(self, gcon, gtable):
        res, tables = gcon.show_tables()
        assert res.OK()
        assert len(tables) == 1

    def test_get_table_row_count(self, gcon, gvector):
        res, count = gcon.get_table_row_count(gvector)
        assert res.OK()
        assert count == 1000

    def test_false_get_table_row_count(self, gcon):
        res, count = gcon.get_table_row_count('fake_table')
        assert not res.OK()

    def test_client_version(self, gcon):
        res = gcon.client_version()
        assert isinstance(res, str)

    def test_server_status(self, gcon):
        status, res = gcon.server_status()
        assert status.OK()

        status, res = gcon.server_status('abc')
        assert status.OK()

        status, res = gcon.server_status('version')
        assert status.OK()


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
class TestPing:

    def test_ping_server_version(self):
        milvus = GrpcMilvus()
        milvus.connect()

        _, version = milvus.server_version()
        assert version == '0.4.0'


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
        LOGGER.error(tables)
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


class TestAddVectors:
    
    def test_add_vectors_normal(self, gcon, gtable):
        vectors = records_factory(dim, nq)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()
        assert len(ids) == 20

        time.sleep(2)
        
        status, count = gcon.get_table_row_count(gtable)
        assert status.OK()
        assert count == 20


class TestSearchVectors:
    def test_search_vectors_normal_1_with_ranges(self, gcon, gtable):
        vectors = records_factory(dim, nq)
        status, ids = gcon.add_vectors(gtable, vectors)
        
        ranges = ranges_factory()
        time.sleep(2)

        s_vectors = [vectors[0]]

        status, result = gcon.search_vectors(gtable, 1, s_vectors, ranges)
        assert status.OK()
        assert len(result) == 1
        assert len(result[0]) == 1

