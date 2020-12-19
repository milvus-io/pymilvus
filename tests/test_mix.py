import pytest
from milvus import Milvus


class TestMilvusPool:
    @pytest.mark.parametrize("pool", ["QueuePool", "SingletonThread", "Singleton"])
    def test_pool_type(self, pool, host, handler):
        try:
            client = Milvus(*host, handler=handler, pool=pool)
        except Exception as e:
            pytest.fail(f"{e}")
