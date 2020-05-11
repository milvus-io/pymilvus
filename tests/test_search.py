import pytest
import random

from milvus import ParamError
from factorys import records_factory

dim = 128
nq = 100


class TestSearch:
    topk = random.randint(1, 10)
    query_records = records_factory(dim, nq)
    search_param = {
        "nprobe": 10
    }

    def test_search_normal(self, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': self.search_param
        }
        res, results = gcon.search(**param)
        assert res.OK()
        assert len(results) == nq
        for r in results:
            assert len(r) == self.topk

        assert results.shape[0] == nq
        assert results.shape[1] == self.topk

    def test_search_default_partition(self, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'partition_tags': ["_default"],
            'params': self.search_param
        }
        res, results = gcon.search(**param)
        assert res.OK()
        assert len(results) == nq
        assert len(results[0]) == self.topk

        assert results.shape[0] == nq
        assert results.shape[1] == self.topk

    def test_search_result_contain__1(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(5)]
        status, _ = gcon.insert(gcollection, vectors)
        assert status.OK()

        status = gcon.flush([gcollection])
        assert status.OK()

        param = {
            'collection_name': gcollection,
            'query_records': self.query_records[:2],
            'top_k': 10,
            'params': self.search_param
        }
        res, results = gcon.search(**param)
        assert res.OK()
        assert len(results) == 2
        assert len(results[0]) == 5

        assert results.shape[0] == 2
        # assert results.shape[1] == self.topk

    def test_search_async_normal(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': self.search_param,
            '_async': True
        }
        future = gcon.search(**param)
        status, results = future.result()

        assert status.OK()
        assert len(results) == nq
        assert len(results[0]) == self.topk

        assert results.shape[0] == nq
        assert results.shape[1] == self.topk

    def test_search_async_callback(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': self.search_param,
            '_async': True
        }

        def cb(status, results):
            assert status.OK()
            assert len(results) == nq
            assert len(results[0]) == self.topk

            assert results.shape[0] == nq
            assert results.shape[1] == self.topk

        future = gcon.search(_callback=cb, **param)
        future.done()

    @pytest.mark.parametrize("query_records", [[], None, "", 123])
    def test_search_invalid_query_records(self, query_records, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': query_records,
            'top_k': self.topk,
            'params': self.search_param,
        }
        with pytest.raises(ParamError):
            gcon.search(**param)

    @pytest.mark.parametrize("params", [[], "", 123])
    def test_search_invalid_params(self, params, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': params,
        }
        with pytest.raises(ParamError):
            gcon.search(**param)

    @pytest.mark.parametrize("topk", [[], "", None, {}])
    def test_search_invalid_topk(self, topk, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': topk,
            'params': self.search_param,
        }
        with pytest.raises(ParamError):
            gcon.search(**param)


@pytest.mark.skip(reason="departed")
class TestSearchByIds:
    vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]

    def test_search_by_ids_normal(self, gcon, gcollection):
        status, ids = gcon.insert(gcollection, self.vectors)
        assert status.OK()

        status = gcon.flush([gcollection])
        assert status.OK()

        search_param = {
            "nprobe": 10
        }
        status, results = gcon.search_by_ids(gcollection, ids[0: 10], 10, params=search_param)
        assert status.OK()
        # assert len(results) == 10
        # assert len(results[0]) == 10
        #
        # assert results.shape[0] == 10
        # assert results.shape[1] == 10

    def test_search_by_ids_with_empty_param(self, gcon, gcollection):
        status, _ = gcon.search_by_ids(gcollection, [1], 1, params=None)
        assert not status.OK()

        status, _ = gcon.search_by_ids(gcollection, [1], 1, params={})
        assert not status.OK()

    @pytest.mark.parametrize("ids", [None, "123", False])
    def test_seach_by_ids_with_invalid_ids(self, ids, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.search_by_ids(gcollection, ids, 1, params={"nprobe": 10})

    @pytest.mark.parametrize("collection", [None, [], {}, False])
    def test_search_by_ids_with_invalid_collection(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.search_by_ids(collection, [1], 1, params={"nprobe": 10})

    @pytest.mark.parametrize("topk", [[], "", None, {}])
    def test_search_by_ids_invalid_topk(self, topk, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.search(gcollection, [1], topk, params={"nprobe": 10})

    @pytest.mark.parametrize("params", [[1, 2, 3], "test", True, 128])
    def test_search_by_ids_invalid_params(self, params, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.search(gcollection, [1], 1, params=params)


class TestSearchInFiles:
    def test_search_in_segment_normal(self, gcon, gvector):
        search_param = {
            "nprobe": 10
        }

        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        for i in range(5000):
            status, _ = gcon.search_in_segment(gvector, file_ids=[i], top_k=1,
                                             query_records=query_vectors, params=search_param)
            if status.OK():
                return
        assert False

    def test_search_in_segment_async(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        search_param = {
            "nprobe": 10
        }

        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        for i in range(5000):
            future = gcon.search_in_segment(gvector, file_ids=[i], top_k=1, query_records=query_vectors,
                                          params=search_param, _async=True)
            status, _ = future.result()
            if status.OK():
                return
        assert False

    def test_search_in_segment_async_callback(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        def cb(status, results):
            print("Search status: ", status)

        search_param = {
            "nprobe": 10
        }

        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        for i in range(5000):
            future = gcon.search_in_segment(gvector, file_ids=[i], top_k=1, query_records=query_vectors,
                                          params=search_param, _async=True, _callback=cb)
            status, _ = future.result()
            if status.OK():
                return
        assert False

    @pytest.mark.parametrize("collection", [[], None, "", 123])
    def test_search_in_segment_invalid_collection(self, collection, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_segment(collection, file_ids=[1], top_k=1, query_records=query_vectors, params={"nprobe": 1})

    @pytest.mark.parametrize("ids", [[], None, "", 123])
    def test_search_in_segment_invalid_file_ids(self, ids, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_segment("test", file_ids=ids, top_k=1, query_records=query_vectors, params={"nprobe": 1})

    @pytest.mark.parametrize("topk", [[], None, "", {}, True, False])
    def test_search_in_segment_invalid_topk(self, topk, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_segment("test", file_ids=[1], top_k=topk, query_records=query_vectors, params={"nprobe": 1})

    @pytest.mark.parametrize("records", [[], None, "", 123, True, False])
    def test_search_in_segment_invalid_records(self, records, gcon):
        with pytest.raises(ParamError):
            gcon.search_in_segment("test", file_ids=[1], top_k=1, query_records=records, params={"nprobe": 1})

    @pytest.mark.parametrize("param", [[], "", 123, (), set(), True, False])
    def test_search_in_segment_invalid_param(self, param, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_segment("test", file_ids=[1], top_k=1, query_records=query_vectors, params=param)
