import random
import sys

sys.path.append('.')

from milvus import Milvus
from milvus.client.exceptions import NotConnectError

import pytest

dim = 128


class TestTimeout:
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

    def test_has_table_timeout(self, gcon, gtable):
        status, _ = gcon.has_table(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_drop_table_timeout(self, gcon, gtable):
        status = gcon.drop_table(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_create_index_timeout(self, gcon, gtable):
        status = gcon.create_index(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_insert_timeout(self, gcon, gtable):
        vectors = [[random.random() for _ in range(dim)] for _ in range(2)]
        status, _ = gcon.insert(table_name=gtable, records=vectors,
                                     ids=None, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_describe_table_timeout(self, gcon, gtable):
        status, _ = gcon.describe_index(table_name=gtable, timeout=self.TIMEOUT)

        assert not status.OK()

    def test_count_table_timeout(self, gcon, gtable):
        status, _ = gcon.count_table(gtable, timeout=self.TIMEOUT)

        assert not status.OK()

    def test_drop_index_timeout(self, gcon, gtable):
        status = gcon.drop_index(gtable, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_preload_table(self, gcon, gvector):
        status = gcon.preload_table(gvector, timeout=self.TIMEOUT)
        assert not status.OK()

    def test_cmd_timeout(self, gcon):
        status, _ = gcon._cmd("666", timeout=self.TIMEOUT)
        assert not status.OK()
