import datetime
import threading
import pytest

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

        ins.update(1, -1)
        assert ins.get(1) == 0

        ins.update(1, 2)
        assert ins.get(1) == 2

        ins.update(2, 100)
        assert ins.get(2) == 100

    def test_get_current_bounded_ts(self):
        now = datetime.datetime.now().timestamp()
        ts1 = ts_utils.get_current_bounded_ts()
        assert ts1 < ts_utils.mkts_from_unixtime(now)
        ts2 = ts_utils.get_current_bounded_ts(200)
        assert ts2 > ts1

    def test_get_eventually_ts(self):
        ts = ts_utils.get_eventually_ts()
        assert ts == ts_utils.EVENTUALLY_TS


