import queue
import threading
import time

from ..stub import Milvus
from ..exceptions import ConnectionPoolError


class ConnectionRecord:
    def __init__(self, uri, recycle, **kwargs):
        '''
        @param uri server uri
        @param recycle int, time period to recycle connection.
        @param kwargs connection key-wprds
        '''
        self._connection = Milvus(uri=uri, **kwargs)
        self.recycle = recycle
        self._last_use_time = time.time()
        self._kw = kwargs

    # def __getattr__(self, item):
    #     getattr(self.connection(), item)

    def connection(self):
        ''' Return a available connection. If connection is out-of-date,
        return new one.
        '''
        self._connection.connect()
        return self._connection


class ConnectionPool:
    def __init__(self, uri, pool_size=10, recycle=-1, wait_timeout=10, **kwargs):
        self._pool = queue.Queue()
        self._uri = uri
        self._pool_size = pool_size
        self._recycle = recycle
        self._wait_timeout = wait_timeout
        self._used_conn = 0
        self._condition = threading.Condition()
        self._kw = kwargs

    def _inc_used(self):
        with self._condition:
            if self._used_conn < self._pool_size:
                self._used_conn += 1
                return True

            return False

    def _dec_used(self):
        with self._condition:
            if self._used_conn == 0:
                return False
            self._used_conn -= 1
            return True

    def _full(self):
        # When pool is full, all of connection are occupied.
        with self._condition:
            return self._used_conn >= self._pool_size

    def _empty(self):
        with self._condition:
            return self._pool.qsize() <= 0 and self._used_conn <= 0

    def _create_connection(self):
        conn = ConnectionRecord(self._uri, self._recycle, **self._kw)
        return ScopedConnection(self, conn)

    def _inc_connection(self):
        if self._inc_used():
            return self._create_connection()

        return self.fetch(block=True)

    def count(self):
        with self._condition:
            return self._pool.qsize() + self._used_conn

    def fetch(self, block=False):
        if self._empty():
            return self._inc_connection()

        try:
            conn = self._pool.get(block=block, timeout=self._wait_timeout)
            return ScopedConnection(self, conn)
        except queue.Empty:
            if block:
                raise ConnectionPoolError("Connection pool is full.")

        if self._full():
            return self.fetch(block=True)

        return self._inc_connection()

    def release(self, conn):
        # Blocking put
        try:
            self._pool.put(conn, False)
            # self._dec_used()
        except queue.Full:
            pass


class ScopedConnection:
    def __init__(self, pool, connection):
        self._pool = pool
        self._connection = connection
        self._closed = False

    def __del__(self):
        self.close()

    def connection(self):
        if self._closed:
            raise ValueError("Connection has been closed.")

        return self._connection

    def client(self):
        conn = self.connection()
        return conn.connection()

    def close(self):
        self._connection and self._pool.release(self._connection)
        self._connection = None
        self._closed = True
