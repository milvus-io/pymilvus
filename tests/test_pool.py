import threading

import pytest

from milvus import Milvus, NotConnectError, VersionError
from milvus.client.pool import ConnectionPool


class TestPool:
    def test_pool_max_conn(self, gip):
        ip_, port_ = gip
        pool = ConnectionPool(uri=f"tcp://{ip_}:{port_}", pool_size=10)

        def run(_pool):
            conn = _pool.fetch()
            assert conn.conn_id() < 10
            conn.has_collection("test_pool")

        thread_list = []
        for _ in range(10 * 3):
            thread = threading.Thread(target=run, args=(pool,))
            thread.start()
            thread_list.append(thread)

    def test_pool_from_stub(self, gip):
        ip_, port_ = gip
        client = Milvus(uri=f"tcp://{ip_}:{port_}", pool_size=10)

        def run(_client):
            _client.has_collection("test_pool")

        thread_list = []
        for _ in range(10 * 3):
            thread = threading.Thread(target=run, args=(client,))
            thread.start()
            thread_list.append(thread)

    @pytest.mark.skip
    def test_pool_args(self):
        with pytest.raises(NotConnectError):
            ConnectionPool(uri="tcp://123.456.780.0:9999", pool_size=10, try_connect=True)

        with pytest.raises(NotConnectError):
            ConnectionPool(uri="tcp://123.456.780.0:9999", pool_size=10, try_connect=False)
