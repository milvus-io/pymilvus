import logging
import numpy as np
import time
import random
import pytest
import sys
import ujson

sys.path.append('.')

from faker import Faker

from milvus import IndexType, MetricType, Prepare, Milvus, Status, ParamError, NotConnectError
from milvus.client.abstract import CollectionSchema, TopKQueryResult
from milvus.client.check import check_pass_param
from milvus.client.hooks import BaseSearchHook

from factorys import (
    collection_schema_factory,
    records_factory,
    fake
)
from milvus.grpc_gen import milvus_pb2

logging.getLogger('faker').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)
faker = Faker(locale='en_US')

dim = 128
nb = 2000
nq = 10


class TestConnection:

    def test_true_connect(self, gip):
        cnn = Milvus()

        cnn.connect(*gip)
        assert cnn.status.OK
        assert cnn.connected()

        # Repeating connect
        # _ = cnn.connect(*gip)
        # status = cnn.connect()
        # assert status == Status.CONNECT_FAILED

    # @pytest.mark.skip
    @pytest.mark.parametrize("url", ['tcp://145.98.234.181:9998', 'tcp://199.67.0.1:2'])
    def test_false_connect(self, url):
        cnn = Milvus()
        with pytest.raises(NotConnectError):
            cnn.connect(uri=url, timeout=1)

    # def test_connected(self, gcon):
    #     assert gcon.connected()

    # def test_non_connected(self):
    #     cnn = Milvus()
    #     # import pdb;pdb.set_trace()
    #     assert not cnn.connected()

    def test_uri(self, ghandler, gip):
        cnn = Milvus(handler=ghandler)
        uri = 'tcp://{}:{}'.format(gip[0], gip[1])
        cnn.connect(uri=uri)
        assert cnn.status.OK()

    @pytest.mark.parametrize("url",
                             ['http://127.0.0.1:45678',
                              'tcp://127.0.a.1:9999',
                              'tcp://127.0.0.1:aaa'])
    def test_uri_error(self, url):
        with pytest.raises(Exception):
            cnn = Milvus()
            cnn.connect(uri=url)

    @pytest.mark.parametrize("h", ['12234', 'aaa', '194.16834.200.200', '134.77.89.34'])
    @pytest.mark.parametrize("p", ['...', 'a', '1', '800000'])
    def test_host_port_error(self, h, p):
        with pytest.raises(Exception):
            Milvus().connect(host=h, port=p)

    # def test_disconnected(self, gip):
    #     cnn = Milvus()
    #     cnn.connect(*gip)
    #
    #     assert cnn.disconnect().OK()
    #     assert not cnn.connected()
    #
    #     cnn.connect(*gip)
    #     assert cnn.connected()

    # def test_disconnected_error(self):
    #     cnn = Milvus()
    #     with pytest.raises(NotConnectError):
    #         cnn.disconnect()

    @pytest.mark.skip
    def test_not_connect(self):
        client = Milvus()

        with pytest.raises(NotConnectError):
            client.create_collection({})

        with pytest.raises(NotConnectError):
            client.has_collection("a")

        with pytest.raises(NotConnectError):
            client.describe_collection("a")

        with pytest.raises(NotConnectError):
            client.drop_collection("a")

        with pytest.raises(NotConnectError):
            client.create_index("a")

        with pytest.raises(NotConnectError):
            client.insert("a", [], None)

        with pytest.raises(NotConnectError):
            client.count_collection("a")

        with pytest.raises(NotConnectError):
            client.show_collections()

        with pytest.raises(NotConnectError):
            client.search("a", 1, 2, [], None)

        with pytest.raises(NotConnectError):
            client.search_in_files("a", [], [], 2, 1, None)

        with pytest.raises(NotConnectError):
            client._cmd("")

        with pytest.raises(NotConnectError):
            client.preload_collection("a")

        with pytest.raises(NotConnectError):
            client.describe_index("a")

        with pytest.raises(NotConnectError):
            client.drop_index("")


class TestCollection:

    def test_create_collection(self, gcon):
        param = collection_schema_factory()
        param['collection_name'] = None
        with pytest.raises(ParamError):
            gcon.create_collection(param)

        param = collection_schema_factory()
        res = gcon.create_collection(param)
        assert res.OK()
        assert gcon.has_collection(param['collection_name'])

        param = collection_schema_factory()
        param['dimension'] = 'string'
        with pytest.raises(ParamError):
            res = gcon.create_collection(param)

        param = '09998876565'
        with pytest.raises(ParamError):
            gcon.create_collection(param)

        param = collection_schema_factory()
        param['collection_name'] = 1234456
        with pytest.raises(ParamError):
            gcon.create_collection(param)

    def test_create_collection_exception(self, gcon):
        param = {
            'collection_name': 'test_151314',
            'dimension': 128,
            'index_file_size': 999999
        }

        status = gcon.create_collection(param)
        assert not status.OK()

    def test_drop_collection(self, gcon, gcollection):
        res = gcon.drop_collection(gcollection)
        assert res.OK()

    def test_false_drop_collection(self, gcon):
        collection_name = 'fake_collection_name'
        res = gcon.drop_collection(collection_name)
        assert not res.OK()

    def test_repeat_create_collection(self, gcon):
        param = collection_schema_factory()

        gcon.create_collection(param)

        res = gcon.create_collection(param)
        LOGGER.error(res)
        assert not res.OK()

    @pytest.mark.skip
    def test_has_collection(self, gcon, gcollection):
        collection_name = fake.collection_name()
        status, result = gcon.has_collection(collection_name)
        assert status.OK(), status.message
        assert not result

        result = gcon.has_collection(gcollection)
        assert result

        with pytest.raises(Exception):
            gcon.has_collection(1111)

    def test_has_collection_invalid_name(self, gcon, gcollection):
        collection_name = "1234455"
        status, result = gcon.has_collection(collection_name)
        assert not status.OK()


class TestRecordCount:
    def test_count_collection(self, gcon, gvector):
        status, num = gcon.count_collection(gvector)
        assert status.OK()
        assert num > 0


class TestVector:

    def test_insert(self, gcon, gcollection):
        param = {
            'collection_name': gcollection,
            'records': records_factory(dim, nq)
        }

        res, ids = gcon.insert(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq

    @pytest.mark.skip
    def test_insert_with_numpy(self, gcon, gcollection):
        vectors = np.random.rand(nq, dim).astype(np.float32)
        param = {
            'collection_name': gcollection,
            'records': vectors
        }

        res, ids = gcon.insert(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq

    def test_insert_with_ids(self, gcon, gcollection):
        param = {
            'collection_name': gcollection,
            'records': records_factory(dim, nq),
            'ids': [i + 1 for i in range(nq)]
        }

        res, ids = gcon.insert(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq

    def test_insert_with_wrong_ids(self, gcon, gcollection):
        param = {
            'collection_name': gcollection,
            'records': records_factory(dim, nq),
            'ids': [i + 1 for i in range(nq - 3)]
        }

        with pytest.raises(ParamError):
            gcon.insert(**param)

    def test_insert_with_no_right_dimension(self, gcon, gcollection):
        param = {
            'collection_name': gcollection,
            'records': records_factory(dim + 1, nq)
        }

        res, ids = gcon.insert(**param)
        assert not res.OK()

    def test_insert_records_empty_list(self, gcon, gcollection):
        param = {'collection_name': gcollection, 'records': [[]]}

        with pytest.raises(Exception):
            gcon.insert(**param)

    def test_false_insert(self, gcon):
        param = {
            'collection_name': fake.collection_name(),
            'records': records_factory(dim, nq)
        }
        res, ids = gcon.insert(**param)
        assert not res.OK()

    def test_insert_wrong_collection_name(self, gcon):
        collection_name = "&*^%&&dvfdgd(()"

        vectors = records_factory(dim, nq)

        status, _ = gcon.insert(collection_name, vectors)
        assert not status.OK()

    # def test_add_vectors_wrong_insert_param(self, gcon, gvector):
    #     vectors = records_factory(dim, nq)
    #
    #     with pytest.raises(ParamError):
    #         gcon.insert(gvector, vectors, insert_param="w353453")


class TestSearch:
    def test_search_normal(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)

        search_param = {
            "nprobe": 10
        }

        param = {
            'collection_name': gvector,
            'query_records': query_records,
            'top_k': topk,
            'params': search_param
        }
        res, results = gcon.search(**param)
        assert res.OK()
        assert len(results) == nq
        assert len(results[0]) == topk

        assert results.shape[0] == nq
        assert results.shape[1] == topk

    def test_search_wrong_dim(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim + 1, nq)

        search_param = {
            "nprobe": 10
        }

        param = {
            'collection_name': gvector,
            'query_records': query_records,
            'top_k': topk,
            'params': search_param
        }
        res, results = gcon.search(**param)
        assert not res.OK()

    def test_search_wrong_collection_name(self, gcon, gvector):
        topk = random.randint(1, 10)
        query_records = records_factory(dim, nq)

        search_param = {
            "nprobe": 10
        }

        param = {
            'collection_name': gvector + 'wrong',
            'query_records': query_records,
            'top_k': topk,
            'params': search_param
        }

        res, _ = gcon.search(**param)
        assert not res.OK()

    def test_false_vector(self, gcon):
        search_param = {
            "nprobe": 10
        }

        param = {
            'collection_name': fake.collection_name(),
            'query_records': records_factory(dim, nq),
            'top_k': 'string',
            'params': search_param
        }
        with pytest.raises(ParamError):
            gcon.search(**param)

        param = {
            'collection_name': fake.collection_name(),
            'query_records': records_factory(dim, nq),
            'top_k': 'string',
            'params': search_param
        }
        with pytest.raises(ParamError):
            gcon.search(**param)

    def test_search_in_files(self, gcon, gvector):
        search_param = {
            "nprobe": 10
        }

        param = {
            'collection_name': gvector,
            'query_records': records_factory(dim, nq),
            'file_ids': [],
            'top_k': random.randint(1, 10),
            'params': search_param
        }

        for id_ in range(5000):
            param['file_ids'].clear()
            param['file_ids'].append(str(id_))
            sta, result = gcon.search_in_files(**param)
            if sta.OK():
                return

        print("search in file failed")
        assert False

    def test_search_in_files_wrong_file_ids(self, gcon, gvector):
        search_param = {
            "nprobe": 10
        }

        param = {
            'collection_name': gvector,
            'query_records': records_factory(dim, nq),
            'file_ids': ['3388833'],
            'top_k': random.randint(1, 10),
            'params': search_param
        }

        sta, results = gcon.search_in_files(**param)
        assert not sta.OK()


class TestCollectionMeta:
    def test_describe_collection(self, gcon, gcollection):
        status, collection_schema = gcon.describe_collection(gcollection)
        assert status.OK()
        assert isinstance(collection_schema, CollectionSchema)

    def test_false_decribe_collection(self, gcon):
        collection_name = fake.collection_name()
        res, collection_schema = gcon.describe_collection(collection_name)
        assert not res.OK()
        assert not collection_schema

    def test_show_collections(self, gcon, gcollection):
        res, collections = gcon.show_collections()
        assert res.OK()
        assert len(collections) == 1

    def test_count_collection(self, gcon, gvector, gcollection):
        res, count = gcon.count_collection(gvector)
        assert res.OK()
        assert count == 10000

    def test_false_count_collection(self, gcon):
        res, count = gcon.count_collection('fake_collection')
        assert not res.OK()

    def test_client_version(self, gcon):
        res = gcon.client_version()
        assert isinstance(res, str)

    def test_server_status(self, gcon):
        status, res = gcon.server_status()
        assert status.OK()


class TestPrepare:

    def test_collection_schema(self):
        res = Prepare.table_schema(fake.collection_name(), random.randint(0, 999), 1024, MetricType.L2, {})
        assert isinstance(res, milvus_pb2.TableSchema)


class TestCreateCollection:

    def test_create_collection_normal(self, gcon):
        param = collection_schema_factory()

        status = gcon.create_collection(param)
        assert status.OK()

    def test_create_collection_default(self, gcon):
        param = {
            'collection_name': 'zilliz_test',
            'dimension': 128
        }

        status = gcon.create_collection(param)
        assert status.OK()

        gcon.drop_collection('zilliz_test')

    def test_create_collection_name_wrong(self, gcon):
        param = collection_schema_factory()
        param['collection_name'] = '.....'
        status = gcon.create_collection(param)
        LOGGER.error(status)
        assert not status.OK()


class TestDescribeCollection:

    def test_describe_collection_normal(self, gcon):
        param = collection_schema_factory()
        gcon.create_collection(param)

        status, collection = gcon.describe_collection(param['collection_name'])
        assert status.OK()
        assert collection.collection_name == param['collection_name']

        status, collection = gcon.describe_collection('collection_not_exists')
        assert not status.OK()


class TestShowCollections:
    def test_show_collections_normal(self, gcon):
        status, collections = gcon.show_collections()
        LOGGER.error(collections)
        assert status.OK()


class TestDropCollection:
    def test_drop_collection_normal(self, gcon):
        param = collection_schema_factory()
        s = gcon.create_collection(param)
        assert s.OK()

        _, collections = gcon.show_collections()
        assert param['collection_name'] in collections

        status = gcon.drop_collection(param['collection_name'])
        _, collections = gcon.show_collections()
        assert param['collection_name'] not in collections

    def test_drop_collection(self, gcon, gcollection):
        status = gcon.drop_collection(gcollection)
        assert status.OK()


class TestHasCollection:
    def test_has_collection(self, gcon):
        param = collection_schema_factory()
        s = gcon.create_collection(param)
        assert s.OK()

        status, flag = gcon.has_collection(param['collection_name'])
        assert status.OK() and flag


class TestAddVectors:

    def test_insert_normal(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)

        assert status.OK()
        assert len(ids) == nq

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, count = gcon.count_collection(gcollection)
        assert status.OK()

        assert count == nq

        gcon.preload_collection(gcollection)

    def test_insert_numpy_array(self, gcon ,gcollection):
        vectors = np.random.rand(10000, 128)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK(), status.message

    def test_insert_ids(self, gcon, gcollection):
        vectors = records_factory(dim, nb)
        ids = [i for i in range(nb)]

        status, vectors_ids = gcon.insert(gcollection, vectors, ids)
        assert status.OK()
        assert len(ids) == len(vectors_ids)

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, count = gcon.count_collection(gcollection)
        assert status.OK()

        assert count == nb


class TestIndex:
    @pytest.mark.skip
    def test_available_index(self, gcon, gcollection):
        for name, member in IndexType.__members__.items():
            if member.value == 0:
                continue

            _index = {
                'nlist': 4096
            }
            status = gcon.create_index(gcollection, member, _index)
            assert status.OK(), "Index {} create failed: {}".format(member, status.message)

            gcon.drop_index(gcollection)

    def test_describe_index(self, gcon, gcollection):
        vectors = records_factory(dim, nb)
        status, ids = gcon.insert(gcollection, vectors)

        assert status.OK()
        assert len(ids) == nb

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        _index = {
            'nlist': 4096
        }

        status = gcon.create_index(gcollection, IndexType.IVF_FLAT, _index)
        assert status.OK(), status.message

        status, index_schema = gcon.describe_index(gcollection)

        assert status.OK()
        print("\n{}\n".format(index_schema))

    def test_describe_index_wrong_collection_name(self, gcon):
        collection_name = "%&%&"
        status, _ = gcon.describe_index(collection_name)

        assert not status.OK()

    def test_drop_index(self, gcon, gcollection):
        vectors = records_factory(dim, nb)
        status, ids = gcon.insert(gcollection, vectors)

        assert status.OK()
        assert len(ids) == nb

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, count = gcon.count_collection(gcollection)
        assert status.OK()
        assert count == nb

        _index = {
            'nlist': 16384
        }

        status = gcon.create_index(gcollection, IndexType.IVFLAT, _index)
        assert status.OK()

        status = gcon.drop_index(gcollection)
        assert status.OK()


@pytest.mark.skip(reason="crud branch")
class TestSearchByID:
    def test_search_by_id_normal(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)

        assert status.OK()

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, result = gcon.search_by_id(gcollection, 2, 10, ids[0])
        assert status.OK()

        print(result)

        assert 1 == len(result)
        assert 2 == len(result[0])
        assert ids[0] == result[0][0].id

    def test_search_by_id_with_partitions(self, gcon, gcollection):
        tag = "search_by_id_partitions_tag"

        status = gcon.create_partition(gcollection, tag)
        assert status.OK()

        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors, partition_tag=tag)
        assert status.OK()

        time.sleep(2)

        status, result = gcon.search_by_id(gcollection, 2, 10, ids[0], partition_tag_array=[tag])
        assert status.OK()

        assert 1 == len(result)
        assert 2 == len(result[0])
        assert ids[0] == result[0][0].id

    def test_search_by_id_with_wrong_param(self, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.search_by_id(gcollection, 'x', 1, 1)

        with pytest.raises(ParamError):
            gcon.search_by_id(gcollection, 1, '1', 1)

        with pytest.raises(ParamError):
            gcon.search_by_id(gcollection, 1, 1, 'aaa')

        status, _ = gcon.search_by_id(gcollection, -1, 1, 1)
        assert not status.OK()

        status, _ = gcon.search_by_id(gcollection, 1, -1, 1)
        assert not status.OK()

    @pytest.mark.skip(reason="except empty result, return result with -1 id instead")
    def test_search_by_id_with_exceed_id(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()

        status, result = gcon.search_by_id(gcollection, 2, 10, ids[0] + 100)
        assert status.OK()
        print(result)
        assert 0 == len(result)


class TestBuildIndex:
    def test_build_index(self, gcon, gvector):
        _D = 128

        time.sleep(5)

        _index = {
            'nlist': 4096
        }

        print("Create index ... ")
        status = gcon.create_index(gvector, IndexType.IVF_FLAT, _index)
        assert status.OK()

    def test_create_index_wrong_index(self, gcon, gvector):
        _index = "23523423"

        with pytest.raises(ParamError):
            gcon.create_index(gvector, _index)

    def test_create_index_wrong_collection_name(self, gcon, gvector):
        _index = {
            'nlist': 4096
        }

        status = gcon.create_index("*^&*^dtedge", IndexType.IVF_FLAT, _index, timeout=None)
        assert not status.OK()


class TestCmd:
    versions = ("0.7.1", "0.8.0")

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

        _, info = gcon._cmd("status")
        assert info in ("OK", "ok")


class TestChecking:

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
    query_vectors = [[random.random() for _ in range(128)] for _ in range(200)]

    def _get_response(self, gcon, gvector, topk, nprobe, nq):
        search_param = {
            "nprobe": nprobe
        }

        return gcon.search(gvector, topk, self.query_vectors[:nq], params=search_param)

    def test_search_result(self, gcon, gvector):
        try:
            status, results = self._get_response(gcon, gvector, 2, 1, 1)
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

            # test result for nq = 10, topk = 10
            status, results = self._get_response(gcon, gvector, 10, 10, 10)
            print(results)
        except Exception:
            assert False

    def test_search_in_files_result(self, gcon, gvector):
        try:
            search_param = {
                "nprobe": 1
            }

            for index in range(1000):
                status, results = \
                    gcon.search_in_files(collection_name=gvector, top_k=1,
                                         file_ids=[str(index)], query_records=self.query_vectors, params=search_param)
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

    def test_empty_result(self, gcon, gcollection):
        status, results = self._get_response(gcon, gcollection, 3, 3, 3)
        shape = results.shape

        for topk_result in results:
            for item in topk_result:
                print(item)


class TestPartition:

    def test_create_partition_in_empty_collection(self, gcon, gcollection):
        status = gcon.create_partition(collection_name=gcollection, partition_tag="1")
        assert status.OK()

        vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        status, _ = gcon.insert(gcollection, vectors, partition_tag="1")
        assert status.OK()

    def test_create_partition_after_insert(self, gcon, gvector):
        status = gcon.create_partition(collection_name=gvector, partition_tag="1")
        assert status.OK()

    def test_insert_with_wrong_partition(self, gcon, gcollection):
        status = gcon.create_partition(collection_name=gcollection, partition_tag="1")
        assert status.OK()

        vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        status, _ = gcon.insert(gcollection, vectors, partition_tag="2")
        assert not status.OK()

    def test_search_with_partition_first(self, gcon, gcollection):
        status = gcon.create_partition(collection_name=gcollection, partition_tag="2")
        assert status.OK()

        status, partitions = gcon.show_partitions(gcollection)
        assert status.OK()

        vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        status, ids = gcon.insert(gcollection, vectors, partition_tag="2")
        assert status.OK()
        assert len(ids) == 100

        gcon.flush([gcollection])

        query_vectors = vectors[:1]

        # search in global scope
        search_param = {
            "nprobe": 1
        }
        status, results = gcon.search(gcollection, 1, query_vectors, params=search_param)
        assert status.OK()
        assert results.shape == (1, 1)

        # search in specific tags
        status, results = gcon.search(gcollection, 1, query_vectors, partition_tags=["2"], params=search_param)
        assert status.OK()
        assert results.shape == (1, 1)

        # search in non-existing tags
        status, results = gcon.search(
            gcollection, 1,
            query_vectors,
            partition_tags=["ee4tergdgdgedgdgergete5465erwtwtwtwtfdf"],
            params=search_param)

        assert status.OK()
        print(results)
        assert results.shape == (0, 0)

    # @pytest.mark.skip
    def test_search_with_partition_insert_first(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()
        assert len(ids) == 100

        # waiting for data prepared
        time.sleep(5)

        partition_tag = "partition_tag_" + faker.word()

        status = gcon.create_partition(collection_name=gcollection, partition_tag=partition_tag)
        assert status.OK()

        status, partitions = gcon.show_partitions(gcollection)
        assert status.OK()

        query_vectors = [[random.random() for _ in range(128)] for _ in range(1)]

        # search in global scope
        search_param = {
            "nprobe": 1
        }
        status, results = gcon.search(gcollection, 1, query_vectors, params=search_param)
        assert status.OK()
        assert results.shape == (1, 1)

        # search in specific tags
        status, results = gcon.search(gcollection, 1, query_vectors, partition_tags=[partition_tag], params=search_param)
        assert status.OK()
        print(results)
        assert results.shape == (0, 0)

        # search in wrong tags
        status, results = gcon.search(gcollection, 1, query_vectors, partition_tags=[faker.word() + "wrong"], params=search_param)
        assert status.OK(), status.message
        print(results)
        assert results.shape == (0, 0)

    def test_drop_partition(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "1")
        assert status.OK()

        vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        status, _ = gcon.insert(gcollection, vectors, partition_tag="1")
        assert status.OK()

        status = gcon.drop_partition(gcollection, "1")
        assert status.OK(), status.message


class TestSegment:
    def test_collection_info(self, gcon, gvector):
        status, info = gcon.collection_info(gvector)
        assert status.OK(), status.message
        assert info.count == 10000
        assert isinstance(info.partitions_stat, list)

        par0 = info.partitions_stat[0]
        assert par0.tag == "_default"
        assert isinstance(par0.segments_stat, list)

        print(info)

    def test_collection_info_wrong_name(self, gcon):
        status, _ = gcon.collection_info("124124122****")
        assert not status.OK()

    def test_get_segment_ids(self, gcon, gvector):
        status, info = gcon.collection_info(gvector)
        assert status.OK()

        seg0 = info.partitions_stat[0].segments_stat[0]

        status, ids = gcon.get_vector_ids(gvector, seg0.segment_name)
        assert status.OK(), status.message
        print(ids[:5])

    def test_get_segment_invalid_ids(self, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_ids(123, "")

        with pytest.raises(ParamError):
            gcon.get_vector_ids("111", [])

    def test_get_segment_non_existent_collection_segment(self, gcon, gcollection):
        status, _ = gcon.get_vector_ids("ijojojononsfsfswgsw", "aaa")
        assert not status.OK()

        status, _ = gcon.get_vector_ids(gcollection, "aaaaaa")
        assert not status.OK()


class TestGetVectorByID:
    def test_get_vector_by_id(self, gcon, gcollection):

        vectors = records_factory(128, 1000)
        ids = [i for i in range(1000)]
        status, ids_out = gcon.insert(collection_name=gcollection, records=vectors, ids=ids)
        assert status.OK(), status.message

        gcon.flush([gcollection])

        status, vec = gcon.get_vector_by_id(gcollection, ids_out[0])
        assert status.OK()


class TestDeleteByID:
    def test_delete_by_id_normal(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()

        time.sleep(2)

        status = gcon.delete_by_id(gcollection, ids[0:10])
        assert status.OK()

    def test_delete_by_id_wrong_param(self, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.delete_by_id(gcollection, "aaa")

    @pytest.mark.skip
    def test_delete_by_id_succeed_id(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()

        time.sleep(2)

        ids_exceed = [ids[-1] + 10]
        status = gcon.delete_by_id(gcollection, ids_exceed)
        assert not status.OK()


class TestFlush:
    def test_flush(self, gcon):
        collection_param = {
            "collection_name": '',
            "dimension": dim
        }

        collection_list = ["test_flush_1", "test_flush_2", "test_flush_3"]
        vectors = records_factory(dim, nq)
        for collection in collection_list:
            collection_param["collection_name"] = collection

            gcon.create_collection(collection_param)

            gcon.insert(collection, vectors)

        status = gcon.flush(collection_list)
        assert status.OK()

        for collection in collection_list:
            gcon.drop_collection(collection)

    def test_flush_with_none(self, gcon, gcollection):
        collection_param = {
            "collection_name": '',
            "dimension": dim
        }

        collection_list = ["test_flush_1", "test_flush_2", "test_flush_3"]
        vectors = records_factory(dim, nq)
        for collection in collection_list:
            collection_param["collection_name"] = collection

            gcon.create_collection(collection_param)

            gcon.insert(collection, vectors)

        status = gcon.flush()
        assert status.OK(), status.message

        for collection in collection_list:
            gcon.drop_collection(collection)


class TestCompact:
    def test_compact_normal(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
        status, ids = gcon.add_vectors(collection_name=gcollection, records=vectors)
        assert status.OK()

        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_after_delete(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
        status, ids = gcon.insert(collection_name=gcollection, records=vectors)
        assert status.OK(), status.message

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status = gcon.delete_by_id(gcollection, ids[100:1000])
        assert status, status.message

        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_with_empty_collection(self, gcon, gcollection):
        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_with_non_exist_name(self, gcon):
        status = gcon.compact(collection_name="die333")
        assert not status.OK()

    def test_compact_with_invalid_name(self, gcon):
        with pytest.raises(ParamError):
            gcon.compact(collection_name=124)


class TestCollectionInfo:
    def test_collection_info_normal(self, gcon, gcollection):
        for _ in range(10):
            records = records_factory(128, 10000)
            status, _ = gcon.insert(gcollection, records)
            assert status.OK()

        gcon.flush([gcollection])

        status, _ = gcon.collection_info(gcollection, timeout=None)
        assert status.OK()

    def test_collection_info_with_partitions(self, gcon, gcollection):
        for i in range(5):
            partition_tag = "tag_{}".format(i)

            status = gcon.create_partition(gcollection, partition_tag)
            assert status.OK(), status.message

            for j in range(3):
                records = records_factory(128, 10000)
                status, _ = gcon.insert(gcollection, records, partition_tag=partition_tag)
                assert status.OK(), status.message

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, _ = gcon.collection_info(gcollection, timeout=None)
        assert status.OK(), status.message

    def test_collection_info_with_empty_collection(self, gcon, gcollection):
        status, _ = gcon.collection_info(gcollection)

        assert status.OK(), status.message

    def test_collection_info_with_non_exist_collection(self, gcon):
        status, _ = gcon.collection_info("Xiaxiede")
        assert not status.OK(), status.message
