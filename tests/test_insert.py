import pytest
import random

import numpy as np

from milvus import ParamError

from factorys import bin_records_factory

dim = 128


class TestInsert:
    vectors = [[random.random() for _ in range(dim)] for _ in range(10000)]

    def test_insert_normal(self, gcon, gcollection):
        status, ids = gcon.insert(gcollection, self.vectors)

        assert status.OK()
        assert len(ids) == len(self.vectors)

    def test_insert_with_ids(self, gcon, gcollection):
        ids = [i for i in range(10000)]
        status, ids_ = gcon.insert(gcollection, self.vectors, ids)

        assert status.OK()
        assert len(ids_) == len(self.vectors)
        assert ids == ids_

    def test_insert_with_numpy(self, gcon, gcollection):
        vectors = np.random.rand(10000, dim).astype(np.float32)
        status, _ = gcon.insert(gcollection, vectors)
        assert status.OK(), status.message

    def test_insert_with_partition(self, gcon, gcollection):
        status = gcon.create_partition(gcollection, "tag01")
        assert status.OK()

        status, ids = gcon.insert(gcollection, self.vectors, partition_tag="tag01")

        assert status.OK()
        assert len(ids) == len(self.vectors)

    def test_insert_async(self, gcon, gcollection, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP not support async invoke")

        future = gcon.insert(gcollection, self.vectors, _async=True)

        status, ids = future.result()
        assert status.OK()
        assert len(ids) == len(self.vectors)

    def test_insert_async_callback(self, gcon, gcollection, ghandler):
        if ghandler == "HTTP":
            pytest.skip("HTTP not support async invoke")

        def cb(status, ids):
            assert status.OK()
            assert len(ids) == len(self.vectors)

        future = gcon.insert(gcollection, self.vectors, _async=True, _callback=cb)
        status, _ = future.result()
        assert status.OK()
        future.done()

    # @pytest.mark.parametrize("vectors", [[], None, "", 12344, [1111], [[]], [[], [1.0, 2.0]]])
    @pytest.mark.parametrize("vectors", [[1111]])
    def test_insert_invalid_vectors(self, vectors, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.insert(gcollection, vectors)

    @pytest.mark.parametrize("ids", [(), "abc", [], [1, 2], [[]]])
    def test_insert_invalid_ids(self, ids, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.insert(gcollection, self.vectors, ids)

    @pytest.mark.parametrize("tag", [[], 123])
    def test_insert_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.insert(gcollection, self.vectors, partition_tag=tag)


class TestInsertBinary:
    def test_insert_binary_normal(self, gcon, gbcollection):
        bin_vectors = bin_records_factory(dim, 10000)
        status, _ = gcon.insert(gbcollection, bin_vectors)
        assert status.OK(), status.message
