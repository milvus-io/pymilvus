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

	def test_update_and_get(self):
		ins = ts_utils._get_gts_dict()
		assert ins.Get(1) == 0

		ins.Update(1, -1)
		assert ins.Get(1) == 0

		ins.Update(1, 2)
		assert ins.Get(1) == 2

		ins.Update(2, 100)
		assert ins.Get(2) == 100

		# test lru later if necessary.
