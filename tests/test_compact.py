import pytest

from milvus import BaseError


class TestCompact:
    def test_compact_normal(self, connect, vrecords):
        try:
            connect.compact(vrecords)
        except Exception as e:
            pytest.fail(f"{e}")

    @pytest.mark.parametrize("threshold", [-1, 1.1])
    def test_compact_invalid_threshold_value(self, threshold, connect, vrecords):
        with pytest.raises(BaseError):
            connect.compact(vrecords, threshold)

    def test_compact_nonexist_collection(self, connect):
        with pytest.raises(BaseError):
            connect.compact("test_compact_xxxxxxxxxxx")

    def test_compact_async(self, connect, vrecords):
        future = connect.compact(vrecords, _async=True)
        future.result()
        future.done()

        def compact_cb(status):
            assert status.OK()

        future2 = connect.compact(vrecords, _async=True, _callback=compact_cb)
        future2.done()


