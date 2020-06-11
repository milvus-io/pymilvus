import pytest

from milvus import IndexType
from milvus import ParamError


class TestCreateIndex:
    def test_create_index_normal(self, gcon, gvector):
        status = gcon.create_index(gvector, IndexType.IVF_FLAT, {"nlist": 1024})
        assert status.OK()

    @pytest.mark.parametrize("index,param", [(IndexType.FLAT, {"nlist": 1024}),
                                             (IndexType.IVF_FLAT, {"nlist": 1024}),
                                             (IndexType.IVF_SQ8, {"nlist": 1024}),
                                             (IndexType.IVF_SQ8_H, {"nlist": 1024}),
                                             (IndexType.IVF_PQ, {"m": 16, "nlist": 1024}),
                                             (IndexType.HNSW, {"M": 16, "efConstruction": 500}),
                                             (IndexType.RNSG, {"search_length": 45, "out_degree": 50,
                                                               "candidate_pool_size": 300, "knng": 100}),
                                             (IndexType.ANNOY, {"n_trees": 20})])
    # @pytest.mark.parametrize("index,param", [(IndexType.IVF_SQ8_H, {"nlist": 1024})])
    def test_create_index_whole(self, index, param, gcon, gvector):
        status, mode = gcon._cmd("mode")
        assert status.OK()

        if mode == "GPU" and index in (IndexType.IVF_PQ,):
            pytest.skip("Index {} not support in GPU version".format(index))
        if mode == "CPU" and index in (IndexType.IVF_SQ8_H,):
            pytest.skip("Index {} not support in CPU version".format(index))

        status = gcon.create_index(gvector, index, param)
        assert status.OK()

    def test_create_index_async(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")
        future = gcon.create_index(gvector, IndexType.IVFLAT, params={"nlist": 1024}, _async=True)
        status = future.result()
        assert status.OK()

    def test_create_index_async_callback(self, gcon, gvector, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP handler not support async")

        def cb(status):
            assert status.OK()

        future = gcon.create_index(gvector, IndexType.IVFLAT, params={"nlist": 1024}, _async=True, _callback=cb)
        future.done()

    @pytest.mark.parametrize("index", [IndexType.INVALID, -1, 100, ""])
    def test_create_index_invalid_index(self, index, gcon, gvector):
        with pytest.raises(ParamError):
            gcon.create_index(gvector, index, {})

    def test_create_index_missing_index(self, gcon, gvector):
        status = gcon.create_index(gvector, params={"nlist": 1024})
        assert status.OK()


class TestDescribeIndex:
    def test_get_index_info_normal(self, gcon, gvector):
        status = gcon.create_index(gvector, IndexType.IVFLAT, params={"nlist": 1024})
        assert status.OK()

        status, index = gcon.get_index_info(gvector)
        assert status.OK()
        assert index.collection_name == gvector
        assert index.index_type == IndexType.IVFLAT
        assert index.params == {"nlist": 1024}

    @pytest.mark.parametrize("collection", [123, None, []])
    def test_get_index_info_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.get_index_info(collection)

    def test_get_index_info_non_existent(self, gcon):
        status, _ = gcon.get_index_info("non_existent")
        assert not status.OK()


class TestDropIndex:
    def test_drop_index_normal(self, gcon, gvector):
        status = gcon.create_index(gvector, IndexType.IVFLAT, {"nlist": 1024})
        assert status.OK()

    @pytest.mark.parametrize("collection", [123, None, []])
    def test_drop_index_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.drop_index(collection)

    def test_drop_index_non_existent(self, gcon):
        status = gcon.drop_index("non_existent")
        assert not status.OK()
