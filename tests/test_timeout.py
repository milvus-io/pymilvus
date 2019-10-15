import pytest
import random
import sys

sys.path.append('.')

from milvus import Milvus
from milvus.client.exceptions import *

dim = 128


class TestTimeout(object):
    TIMEOUT = 0.0000000001

    def test_connect_timeout(self):
        client = Milvus()
        with pytest.raises(NotConnectError):
            client.connect(timeout=self.TIMEOUT)

    def test_create_table_timeout(self, gcon):
        param = {
            "table_name": "pymilvus",
            "dimension": 128
        }
        status = gcon.create_table(param, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_add_vectors_timeout(self, gcon, gtable):
        vectors = [[random.random() for _ in range(128)] for _ in range(1000)]

        status, _ = gcon.add_vectors(table_name=gtable, records=vectors, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_has_table_timeout(self, gcon, gtable):
        status, _ = gcon.has_table(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_delete_table_timeout(self, gcon, gtable):
        status = gcon.delete_table(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_create_index_timeout(self, gcon, gtable):
        status = gcon.create_index(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_add_vector_timeout(self, gcon, gtable):
        vectors = [[random.random() for _ in range(dim)] for _ in range(2)]
        status, _ = gcon.add_vectors(table_name=gtable, records=vectors, ids=None, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_describe_table_timeout(self, gcon, gtable):
        status, _ = gcon.describe_index(table_name=gtable, timeout=self.TIMEOUT)

        assert not status.OK()

    def test_get_table_row_count_timeout(self, gcon, gtable):
        status, _ = gcon.get_table_row_count(gtable, timeout=self.TIMEOUT)

        assert not status.OK()

    def test_drop_index_timeout(self, gcon, gtable):
        status = gcon.drop_index(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_delete_by_range(self, gcon, gvector):
        status = gcon.delete_vectors_by_range(gvector, "2019-05-01", "2019-12-31", timeout=self.TIMEOUT)
        assert not status.OK()

    def test_preload_table(self, gcon, gvector):
        status = gcon.preload_table(gvector, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_cmd_timeout(self, gcon):
        status, _ = gcon._cmd("666", timeout=self.TIMEOUT)
        assert not status.OK()
