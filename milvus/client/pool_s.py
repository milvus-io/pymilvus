import threading
import time
import queue

from .exceptions import TimeoutError
from .stub import Milvus


class Duration:
    def __init__(self):
        self.start_ts = time.time()
        self.end_ts = None

    def stop(self):
        if self.end_ts:
            return False

        self.end_ts = time.time()
        return True

    @property
    def value(self):
        if not self.end_ts:
            return None

        return self.end_ts - self.start_ts


class ProxyMixin:
    def __getattr__(self, name):
        target = self.__dict__.get(name, None)
        if target or not self.connection:
            return target
        return getattr(self.connection, name)


class ScopedConnection(ProxyMixin):
    def __init__(self, pool, connection):
        self.pool = pool
        self.connection = connection
        self.duration = Duration()

    def __del__(self):
        self.release()

    def __str__(self):
        return self.connection.__str__()

    def release(self):
        if not self.pool or not self.connection:
            return
        self.pool.release(self.connection)
        self.duration.stop()
        self.pool.record_duration(self.connection, self.duration)
        self.pool = None
        self.connection = None


class ConnectionPool:
    def __init__(self, url, pool_size=5, recycle=-1, wait_timeout=-1):

        # server address
        self._url = url

        # max number of connection
        self._max_connection = pool_size

        # max time interval for waiting
        # if wait_timeout is -1, pool will wait until get a avaliable conn
        self._wait_timeout = None if wait_timeout == -1 else wait_timeout

        # store connected and unemployed stubs
        # this queue is thread-safe
        self._pending_queue = queue.Queue(maxsize=pool_size)

        # store activate stubs
        # thread-safe
        self._activate_queue = queue.Queue(maxsize=pool_size)

        # threading condition
        self._condition = threading.Condition()

    def __del__(self):
        ''' release all 
        '''

        # wait for unfinished connection
        if self._activate_queue.qsize() > 0:
            time.sleep(10 / 1000)

        with self._condition:
            try:
                while True:
                    conn = self._pending_queue.get(block=False)
                    conn.close()
                    del conn
            except queue.Empty as err:
                # pending queue is empty, just pass
                pass

    def _create_connection(self):
        return Milvus(uri=self._url)

    def fetch(self):
        try:
            with self._condition:
                # if self._activate_queue.qsize + self._pending_queue.qsize >= self._max_connection
                # or self._activate_queue.full():
                if self._pending_queue.qsize() <= 0 and not self._activate_queue.full():
                    conn = self._create_connection()
                    self._activate_queue.put(conn)
                    ScopedConnection(self, conn)
        
                conn = self._pending_queue.get(timeout=self._wait_timeout)
                self._activate_queue.put(conn)
                return ScopedConnection(self, conn)
        except queue.Empty as err:
            raise TimeoutError("Waiting for connection instance timeout.")

    def release(self, conn):
        with self._condition:
            if self._pending_queue.full():
                # The pending queue is full, just return 
                return
            # exact from activat queue and put into pending queue
            self._pending_queue.put(conn)

