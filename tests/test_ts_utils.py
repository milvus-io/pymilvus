import threading

from pymilvus.client import ts_utils


class TestTsUtils:
    def test_singleton(self):
        ins1 = ts_utils._get_gts_dict()
        ins2 = ts_utils._get_gts_dict()
        assert id(ins1) == id(ins2)

    def test_singleton_mutiple_thread(self):
        ins = ts_utils._get_gts_dict()

        def _f():
            g = ts_utils._get_gts_dict()
            assert id(g) == id(ins)

        t1 = threading.Thread(target=_f)
        t2 = threading.Thread(target=_f)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

    def test_update_and_get(self):
        # test lru later if necessary.

        ins = ts_utils._get_gts_dict()
        assert ins.get(1) == 0

        ins.update("coll1", -1)
        assert ins.get("coll1") == 0

        ins.update("coll1", 2)
        assert ins.get("coll1") == 2

        ins.update("coll2", 100)
        assert ins.get("coll2") == 100

    def test_get_current_bounded_ts(self):
        ts = ts_utils.get_bounded_ts()
        assert ts == ts_utils.BOUNDED_TS

    def test_get_eventually_ts(self):
        ts = ts_utils.get_eventually_ts()
        assert ts == ts_utils.EVENTUALLY_TS
