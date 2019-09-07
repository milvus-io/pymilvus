import pytest
import random
import sys

sys.path.append('.')

from milvus.client.GrpcClient import Prepare, GrpcMilvus, Status
from milvus.client.Abstract import IndexType, TableSchema, TopKQueryResult, MetricType
from milvus.client.Exceptions import *


dim = 128


class TestTimeout(object):

    def test_connect_timeout(self):
        client = GrpcMilvus()
        with pytest.raises(NotConnectError):
            client.connect(timeout=0.0000000000001)

    def test_create_table_timeout(self, gcon, gtable):

        vectors = [[random.random() for _ in range(128)] for _ in range(1000)]

        status, _ = gcon.add_vectors(table_name=gtable, records=vectors, timeout=0.0001)
        assert not status.OK()
        assert status.code == status.UNEXPECTED_ERROR

    def test_has_table_timeout(self, gcon, gtable):
        flag = gcon.has_table(gtable, timeout=0.00000000001)
        assert not flag

    def test_delete_table_timeout(self, gcon, gtable):
        status = gcon.delete_table(gtable, timeout=0.00000000001)
        assert not status.OK()
        assert status.code == Status.UNEXPECTED_ERROR

    def test_create_index_timeout(self, gcon, gtable):
        status = gcon.create_index(gtable, timeout=0.0000000001)
        assert not status.OK()
        assert status.code == Status.UNEXPECTED_ERROR

    def test_add_vector_timeout(self, gcon, gtable):
        vectors = [[random.random() for _ in range(dim)] for _ in range(2)]
        status, _ = gcon.add_vectors(table_name=gtable, records=vectors, ids=None, timeout=0.0001)
        assert not status.OK()
        assert status.code == Status.UNEXPECTED_ERROR

    def test_describe_table_timeout(self, gcon, gtable):
        status, _ = gcon.describe_index(table_name=gtable, timeout=0.00000001)

        assert not status.OK()
        assert status.code == Status.UNEXPECTED_ERROR

    def test_drop_index_timeout(self, gcon, gtable):
        status = gcon.drop_index(gtable, timeout=0.00000001)
        assert not status.OK()
        assert status.code == Status.UNEXPECTED_ERROR
