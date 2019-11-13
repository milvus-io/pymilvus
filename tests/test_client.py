import logging
import time
import random
import pytest
import sys

sys.path.append('.')

from milvus import IndexType, MetricType, Prepare, Milvus, Status, ParamError, NotConnectError, ConnectError
from milvus.client.grpc_client import Prepare, GrpcMilvus, Status
from milvus.client.abstract import TableSchema, TopKQueryResult
from milvus.client.utils import check_pass_param

from factorys import (
    table_schema_factory,
    records_factory,
    query_ranges_factory,
    ranges_factory,
    fake
)
from milvus.grpc_gen import milvus_pb2

LOGGER = logging.getLogger(__name__)

dim = 128
nb = 2000
nq = 10


class TestChannel:
    client = GrpcMilvus()

    def test_channel_host_port(self):
        try:
            self.client._set_channel(host="localhost", port="19530")
            self.client._set_channel(host="www.milvus.io", port="19530")
        except Exception:
            assert False

    def test_channel_uri(self):
        try:
            self.client._set_channel(uri="tcp://192.168.1.1:9999")
        except Exception:
            assert False

    def test_channel_host_non_port(self):
        try:
            self.client._set_channel(host="localhost")
        except Exception:
            assert False

    def test_channel_only_port(self):
        with pytest.raises(ParamError):
            self.client._set_channel(port=9999)


class TestConnection:

    def test_true_connect(self, gip):
        cnn = GrpcMilvus()

        cnn.connect(*gip)
        assert cnn.status.OK
        assert cnn.connected()

        # Repeating connect
        _ = cnn.connect(*gip)
        status = cnn.connect()
        assert status == Status.CONNECT_FAILED

    @pytest.mark.parametrize("url", ['tcp://127.0.0.1:7987', 'tcp://123.0.0.1:19530'])
    def test_false_connect(self, url):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.connect(uri=url, timeout=2)
            LOGGER.error(cnn.status)
            assert not cnn.status.OK()

    def test_connected(self, gcon):
        assert gcon.connected()

    def test_non_connected(self):
        cnn = GrpcMilvus()
        assert not cnn.connected()

    def test_uri(self, gip):
        cnn = GrpcMilvus()
        uri = 'tcp://{}:{}'.format(gip[0], gip[1])
        cnn.connect(uri=uri)
        assert cnn.status.OK()

    @pytest.mark.parametrize("url",
                             ['http://127.0.0.1:45678',
                              'tcp://127.0.a.1:9999',
                              'tcp://127.0.0.1:aaa'])
    def test_uri_error(self, url):
        with pytest.raises(Exception):
            cnn = GrpcMilvus()
            cnn.connect(uri=url)

    @pytest.mark.parametrize("h", ['12234', 'aaa', '192.168.1.101'])
    @pytest.mark.parametrize("p", ['...', 'a', 49999])
    def test_host_port_error(self, h, p):
        with pytest.raises(Exception):
            cnn = GrpcMilvus()
            cnn.connect(host=h, port=p)

    def test_disconnected(self, gip):
        cnn = GrpcMilvus()
        cnn.connect(*gip)

        assert cnn.disconnect().OK()
        assert not cnn.connected()

        cnn.connect(*gip)
        assert cnn.connected()

    def test_disconnected_error(self):
        cnn = GrpcMilvus()
        with pytest.raises(NotConnectError):
            cnn.disconnect()

    def test_not_connect(self):
        client = GrpcMilvus()

        with pytest.raises(NotConnectError):
            client.create_table({})

        with pytest.raises(NotConnectError):
            client.has_table("a")

        with pytest.raises(NotConnectError):
            client.describe_table("a")

        with pytest.raises(NotConnectError):
            client.delete_table("a")

        with pytest.raises(NotConnectError):
            client.create_index("a")

        with pytest.raises(NotConnectError):
            client.add_vectors("a", [], None)

        with pytest.raises(NotConnectError):
            client.get_table_row_count("a")

        with pytest.raises(NotConnectError):
            client.show_tables()

        with pytest.raises(NotConnectError):
            client.search_vectors("a", 1, 2, [])

        with pytest.raises(NotConnectError):
            client.search_vectors_in_files("a", [], [], 2, 1)

        with pytest.raises(NotConnectError):
            client._cmd("")

        with pytest.raises(NotConnectError):
            client.preload_table("a")

        with pytest.raises(NotConnectError):
            client.describe_index("a")

        with pytest.raises(NotConnectError):
            client.drop_index("")

    def test_set_channel(self):
        cnn = GrpcMilvus()
        cnn._set_channel(host="www.baidu.com", port="19530")


class TestTable:

    def test_create_table(self, gcon):
        param = table_schema_factory()
        param['table_name'] = None
        with pytest.raises(ParamError):
            gcon.create_table(param)

        param = table_schema_factory()
        res = gcon.create_table(param)
        assert res.OK()
        assert gcon.has_table(param['table_name'])

        param = table_schema_factory()
        param['dimension'] = 'string'
        with pytest.raises(ParamError):
            res = gcon.create_table(param)

        param = '09998876565'
        with pytest.raises(ParamError):
            gcon.create_table(param)

        param = table_schema_factory()
        param['table_name'] = 1234456
        with pytest.raises(ParamError):
            gcon.create_table(param)

    def test_create_table_default(self, gcon):
        _param = {
            'table_name': 'name',
            'dimension': 16,
        }

        _status = gcon.create_table(_param)
        assert _status.OK()
        time.sleep(1)
        gcon.delete_table(_param['table_name'])

    def test_create_table_exception(self, gcon):
        param = {
            'table_name': 'test_151314',
            'dimension': 128,
            'index_file_size': 999999
        }

        status = gcon.create_table(param)
        assert not status.OK()

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
        status, result = gcon.has_table(table_name)
        assert status.OK()
        assert not result

        result = gcon.has_table(gtable)
        assert result

        with pytest.raises(Exception):
            gcon.has_table(1111)

    def test_has_table_invalid_name(self, gcon, gtable):
        table_name = "1234455"
        status, result = gcon.has_table(table_name)
        assert not status.OK()


class TestrecordCount:
    def test_count_table(self, gcon, gvector):
        status, num = gcon.count_table(gvector)
        assert status.OK()
        assert num > 0


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
            gcon.add_vectors(**param)

    def test_insert(self, gcon, gtable):
        param = {
            'table_name': gtable,
            'records': records_factory(dim, nq)
        }

        res, ids = gcon.insert(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq

    def test_add_vector_with_ids(self, gcon, gtable):
        param = {
            'table_name': gtable,
            'records': records_factory(dim, nq),
            'ids': [i + 1 for i in range(nq)]
        }

        res, ids = gcon.add_vectors(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq

    def test_add_vector_with_wrong_ids(self, gcon, gtable):
        param = {
            'table_name': gtable,
            'records': records_factory(dim, nq),
            'ids': [i + 1 for i in range(nq - 3)]
        }

        with pytest.raises(ParamError):
            gcon.add_vectors(**param)

    def test_add_vector_with_no_right_dimension(self, gcon, gtable):
        param = {
            'table_name': gtable,
            'records': records_factory(dim + 1, nq)
        }

        res, ids = gcon.add_vectors(**param)
        assert not res.OK()

    def test_add_vector_records_empty_list(self, gcon, gtable):
        param = {'table_name': gtable, 'records': [[]]}

        with pytest.raises(Exception):
            gcon.add_vectors(**param)

    def test_false_add_vector(self, gcon):
        param = {
            'table_name': fake.table_name(),
            'records': records_factory(dim, nq)
        }
        res, ids = gcon.add_vectors(**param)
        assert not res.OK()

    def test_add_vectors_wrong_table_name(self, gcon):
        table_name = "&*^%&&dvfdgd(()"

        vectors = records_factory(dim, nq)

        status, _ = gcon.add_vectors(table_name, vectors)
        assert not status.OK()

    def test_add_vectors_wrong_insert_param(self, gcon, gvector):
        vectors = records_factory(dim, nq)

        with pytest.raises(ParamError):
            gcon.add_vectors(gvector, vectors, insert_param="w353453")


class TestSearch:
    def test_search_vector_normal(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector,
            'query_records': query_records,
            'top_k': topk,
            'nprobe': 10
        }
        res, results = gcon.search_vectors(**param)
        assert res.OK()
        assert len(results) == nq
        assert len(results[0]) == topk

        assert results.shape[0] == nq
        assert results.shape[1] == topk

    def test_search_normal(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector,
            'query_records': query_records,
            'top_k': topk,
            'nprobe': 10
        }
        res, results = gcon.search(**param)
        assert res.OK()
        assert len(results) == nq
        assert len(results[0]) == topk

        assert results.shape[0] == nq
        assert results.shape[1] == topk

    def test_search_vector_wrong_dim(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim + 1, nq)
        param = {
            'table_name': gvector,
            'query_records': query_records,
            'top_k': topk,
            'nprobe': 10
        }
        res, results = gcon.search_vectors(**param)
        assert not res.OK()

    def test_search_vector_wrong_table_name(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector + 'wrong',
            'query_records': query_records,
            'top_k': topk,
            'nprobe': 10
        }
        res, results = gcon.search_vectors(**param)
        assert not res.OK()

    def test_search_vector_with_range(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)
        param = {
            'table_name': gvector,
            'top_k': topk,
            'nprobe': 10,
            'query_records': query_records,
            'query_ranges': query_ranges_factory()

        }
        res, results = gcon.search_vectors(**param)
        assert res.OK()
        assert len(results) == nq
        assert len(results[0]) == topk

    def test_false_vector(self, gcon):
        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(dim, nq),
            'top_k': 'string',
            'nprobe': 10
        }
        with pytest.raises(ParamError):
            gcon.search_vectors(**param)

        param = {
            'table_name': fake.table_name(),
            'query_records': records_factory(dim, nq),
            'top_k': 'string',
            'nprobe': 10
        }
        with pytest.raises(ParamError):
            gcon.search_vectors(**param)

        param = {'table_name': fake.table_name(),
                 'query_records': records_factory(dim, nq),
                 'top_k': random.randint(1, 10),
                 'nprobe': 10,
                 'query_ranges': ['false_date_format']}
        with pytest.raises(ParamError):
            gcon.search_vectors(**param)

    def test_search_in_files(self, gcon, gvector):
        param = {
            'table_name': gvector,
            'query_records': records_factory(dim, nq),
            'file_ids': [],
            'top_k': random.randint(1, 10),
            'nprobe': 16
        }

        for id_ in range(600):
            param['file_ids'].clear()
            param['file_ids'].append(str(id_))
            sta, result = gcon.search_vectors_in_files(**param)
            if sta.OK():
                param['lazy'] = True
                gcon.search_vectors_in_files(**param)
                return

        print("search in file failed")
        assert False

    def test_search_in_files_wrong_file_ids(self, gcon, gvector):
        param = {
            'table_name': gvector,
            'query_records': records_factory(dim, nq),
            'file_ids': ['3388833'],
            'top_k': random.randint(1, 10)
        }
        sta, results = gcon.search_vectors_in_files(**param)
        assert not sta.OK()

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

    def test_get_table_row_count(self, gcon, gvector, gtable):
        res, count = gcon.get_table_row_count(gvector)
        assert res.OK()
        assert count == 10000

    def test_count_table(self, gcon, gvector, gtable):
        res, count = gcon.count_table(gvector)
        assert res.OK()
        assert count == 10000

    def test_false_get_table_row_count(self, gcon):
        res, count = gcon.get_table_row_count('fake_table')
        assert not res.OK()

    def test_client_version(self, gcon):
        res = gcon.client_version()
        assert isinstance(res, str)

    def test_server_status(self, gcon):
        status, res = gcon.server_status()
        assert status.OK()

        status, res = gcon.server_status()
        assert status.OK()

        status, res = gcon.server_status()
        assert status.OK()


class TestPrepare:

    def test_table_schema(self):
        param = {
            'table_name': fake.table_name(),
            'dimension': random.randint(0, 999),
            'index_file_size': 1024,
            'metric_type': MetricType.L2
        }
        res = Prepare.table_schema(param)
        assert isinstance(res, milvus_pb2.TableSchema)


class TestCreateTable:

    def test_create_table_normal(self, gcon):
        param = table_schema_factory()

        status = gcon.create_table(param)
        assert status.OK()

    def test_create_table_default(self, gcon):
        param = {
            'table_name': 'zilliz_test',
            'dimension': 128
        }

        status = gcon.create_table(param)
        assert status.OK()

        gcon.delete_table('zilliz_test')

    def test_create_table_name_wrong(self, gcon):
        param = table_schema_factory()
        param['table_name'] = '.....'
        status = gcon.create_table(param)
        LOGGER.error(status)
        assert not status.OK()


class TestDescribeTable:

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
        assert s.OK()
        time.sleep(5)
        _, tables = gcon.show_tables()
        assert param['table_name'] in tables

        status = gcon.delete_table(param['table_name'])
        _, tables = gcon.show_tables()
        assert param['table_name'] not in tables

    def test_drop_table(self, gcon, gtable):
        status = gcon.drop_table(gtable)
        assert status.OK()


class TestHasTable:
    def test_has_table(self, gcon):
        param = table_schema_factory()
        s = gcon.create_table(param)
        assert s.OK()

        status, flag = gcon.has_table(param['table_name'])
        assert status.OK()
        assert flag


class TestAddVectors:

    def test_add_vectors_normal(self, gcon, gtable):
        vectors = records_factory(dim, nq)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()
        assert len(ids) == nq

        time.sleep(2)

        status, count = gcon.get_table_row_count(gtable)
        assert status.OK()

        assert count == nq

        gcon.preload_table(gtable)

    def test_add_vectors_ids(self, gcon, gtable):
        vectors = records_factory(dim, nb)
        ids = [i for i in range(nb)]

        status, vectors_ids = gcon.add_vectors(gtable, vectors, ids)
        assert status.OK()
        assert len(ids) == len(vectors_ids)

        time.sleep(5)

        status, count = gcon.get_table_row_count(gtable)
        assert status.OK()

        assert count == nb


class TestIndex:

    def test_describe_index(self, gcon, gtable):
        vectors = records_factory(dim, nb)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()
        assert len(ids) == nb

        time.sleep(3)

        _index = {
            'index_type': IndexType.IVFLAT,
            'nlist': 4096
        }

        gcon.create_index(gtable, _index)
        time.sleep(5)

        status, index_schema = gcon.describe_index(gtable)

        assert status.OK()
        print("\n{}\n".format(index_schema))

    def test_describe_index_wrong_table_name(self, gcon):
        table_name = "%&%&"
        status, _ = gcon.describe_index(table_name)

        assert not status.OK()

    def test_drop_index(self, gcon, gtable):
        vectors = records_factory(dim, nb)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()
        assert len(ids) == nb

        time.sleep(6)

        status, count = gcon.get_table_row_count(gtable)
        assert status.OK()
        assert count == nb

        _index = {
            'index_type': IndexType.IVFLAT,
            'nlist': 16384
        }

        status = gcon.create_index(gtable, _index)

        time.sleep(1)

        status = gcon.drop_index(gtable)
        assert status.OK()


class TestSearchVectors:
    def test_search_vectors_normal_1_with_ranges(self, gcon, gtable):
        vectors = records_factory(dim, nq)
        status, ids = gcon.add_vectors(gtable, vectors)

        assert status.OK()

        ranges = ranges_factory()
        time.sleep(2)

        s_vectors = [vectors[0]]

        status, result = gcon.search_vectors(gtable, 1, 10, s_vectors, ranges)
        assert status.OK()
        assert len(result) == 1
        assert len(result[0]) == 1


class TestBuildIndex:
    def test_build_index(self, gcon, gvector):
        _D = 128

        time.sleep(30)

        _index = {
            'index_type': IndexType.IVFLAT,
            'nlist': 4096
        }

        print("Create index ... ")
        status = gcon.create_index(gvector, _index)
        assert status.OK()

    def test_create_index_wrong_index(self, gcon, gvector):
        _index = "23523423"

        with pytest.raises(ParamError):
            gcon.create_index(gvector, _index)

    def test_create_index_wrong_timeout(self, gcon, gvector):
        _index = {
            'index_type': IndexType.IVFLAT,
            'nlist': 4096
        }

        with pytest.raises(ParamError):
            gcon.create_index(gvector, timeout=-90)

    def test_create_index_wrong_table_name(self, gcon, gvector):
        _index = {
            'index_type': IndexType.IVFLAT,
            'nlist': 4096
        }

        status = gcon.create_index("*^&*^dtedge", timeout=-1)
        assert not status.OK()


class TestCmd:
    versions = ("0.5.3",)

    def test_client_version(self, gcon):
        try:
            import milvus
            assert gcon.client_version() == milvus.__version__
        except ImportError:
            assert False, "Import error"

    def test_server_version(self, gcon):
        _, version = gcon.server_version()
        assert version in self.versions

    def test_server_status(self, gcon):
        _, status = gcon.server_status()
        assert status in ("OK", "ok")

    def test_cmd(self, gcon):
        _, info = gcon._cmd("version")
        assert info in self.versions

        _, info = gcon._cmd("OK")
        assert info in ("OK", "ok")


class TestUtils:

    @pytest.mark.parametrize(
        "key_, value_",
        [("ids", [1, 2]), ("nprobe", 12), ("nlist", 4096), ("cmd", 'OK')]
    )
    def test_param_check_normal(self, key_, value_):
        try:
            check_pass_param(**{key_: value_})
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key_, value_",
        [("ids", []), ("nprobe", "aaa"), ("nlist", "aaa"), ("cmd", 123)]
    )
    def test_param_check_error(self, key_, value_):
        with pytest.raises(ParamError):
            check_pass_param(**{key_: value_})


class TestQueryResult:
    query_vectors = [[random.random() for _ in range(128)] for _ in range(5)]

    def _get_response(self, gcon, gvector):
        return gcon.search_vectors(gvector, 1, 1, self.query_vectors)

    def test_search_result(self, gcon, gvector):
        try:
            status, results = self._get_response(gcon, gvector)
            assert status.OK()

            # test get_item
            shape = results.shape

            # test TopKQueryResult slice
            rows = results[:1]

            # test RowQueryResult
            row = results[shape[0] - 1]

            # test RowQueryResult slice
            items = row[:1]

            # test iter
            for topk_result in results:
                for item in topk_result:
                    print(item)

            # test len
            len(results)
            # test print
            print(results)
        except Exception:
            assert False

    def test_search_in_files_result(self, gcon, gvector):
        try:
            for index in range(500):
                status, results = \
                    gcon.search_vectors_in_files(table_name=gvector,
                                                 top_k=1,
                                                 nprobe=1,
                                                 file_ids=[str(index)],
                                                 query_records=self.query_vectors)
                if status.OK():
                    break

            # test get_item
            shape = results.shape
            item = results[shape[0] - 1][shape[1] - 1]

            # test iter
            for topk_result in results:
                for item in topk_result:
                    print(item)

            # test len
            len(results)
            # test print
            print(results)
        except Exception:
            assert False

    def test_empty_result(self, gcon, gtable):
        status, results = self._get_response(gcon, gtable)
        shape = results.shape

        for topk_result in results:
            for item in topk_result:
                print(item)
