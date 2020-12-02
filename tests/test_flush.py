import pytest

from milvus import BaseError

from factorys import records_factory


class TestFlush:
    def test_flush_normal(self, connect, vrecords):
        try:
            connect.flush([vrecords])
        except Exception as e:
            pytest.fail(f"{e}")

    def test_flush_none_param(self, connect, vrecords):
        connect.flush()

    def test_flush_after_insert(self, connect, vcollection, dim):
        count = 10000
        entities = [{"Vec": vector} for vector in records_factory(dim, count)]
        try:
            connect.create_partition(vcollection, 'p_0')
            connect.insert(vcollection, entities, partition_tag='p_0')
        except Exception as e:
            pytest.fail(f"Unexpected MyError: {e}")

        rcount = connect.count_entities(vcollection)
        assert rcount == 0

        connect.flush([vcollection])
        rcount = connect.count_entities(vcollection)
        assert rcount == count

    def test_compact_nonexist_collection(self, connect):
        with pytest.raises(BaseError):
            connect.flush(["test_compact_xxxxxxxxxxx"])

    def test_flush_async(self, connect, vrecords):
        future = connect.flush([vrecords], _async=True)
        future.result()
        future.done()

        def flush_cb(status):
            assert status.OK()

        future2 = connect.flush([vrecords], _async=True, _callback=flush_cb)
        future2.done()


