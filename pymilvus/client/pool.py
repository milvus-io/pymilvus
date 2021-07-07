# -*- coding: UTF-8 -*-

import logging
import os
import queue
import threading
import time
from collections import defaultdict

from . import __version__
from .grpc_handler import GrpcHandler, RegistryHandler
from .exceptions import ConnectionPoolError, NotConnectError, VersionError

support_versions = ('0.11.x',)


def _is_version_match(version):
    version_prefix = version.split(".")
    for support_version in support_versions:
        support_version_prefix = support_version.split(".")
        if version_prefix[0] == support_version_prefix[0] and version_prefix[1] == support_version_prefix[1]:
            return True
    return False


LOGGER = logging.getLogger(__name__)


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


class ConnectionRecord:
    def __init__(self, uri, channel=None, handler="GRPC", conn_id=-1, pre_ping=True, **kwargs):
        '''
        @param uri server uri
        @param recycle int, time period to recycle connection.
        @param kwargs connection key-wprds
        '''
        self._conn_id = conn_id
        self._uri = uri
        # self.recycle = recycle
        self._pre_ping = pre_ping
        self._last_use_time = time.time()
        self._kw = kwargs

        if handler == "GRPC":
            self._connection = GrpcHandler(uri=uri, channel=channel, pre_ping=self._pre_ping, conn_id=conn_id, **self._kw)
        else:
            raise ValueError("Unknown handler type. Use GRPC or HTTP")

    def _register_link(self):
        with RegistryHandler(uri=self._uri, pre_ping=self._pre_ping, conn_id=self._conn_id, **self._kw) as register:
            ip, port = register.register_link()
            self._uri = "tcp://{}:{}".format(ip, port)

    def connection(self):
        ''' Return a available connection. If connection is out-of-date,
        return new one.
        '''
        self._last_use_time = time.time()
        # if self._pre_ping:
        #     self._connection.ping()
        return self._connection


class ConnectionPool:
    def __init__(self, uri, pool_size=10, wait_timeout=30, try_connect=True, **kwargs):
        # Asynchronous queue to store connection
        self._pool = queue.Queue(maxsize=pool_size)
        self._uri = uri
        self._pool_size = pool_size
        self._wait_timeout = wait_timeout

        # Record used connection number.
        self._used_conn = 0
        self._condition = threading.Condition()
        self._kw = kwargs

        self.durations = defaultdict(list)

        self._try_connect = try_connect
        # if self._try_connect:
        #     self._max_retry = kwargs
        self._prepare()

    def _prepare(self):
        conn = self.fetch()
        with self._condition:
            if self._try_connect:
                # LOGGER.debug("Try connect server {}".format(self._uri))
                conn.client().ping()

            status, version = conn.client().server_version(timeout=30)
            if not status.OK():
                raise NotConnectError("Cannot check server version: {}".format(status.message))
            if not _is_version_match(version):
                raise VersionError(
                    "Version of python SDK(v{}) not match that of server v{}, excepted is v{}".format(__version__,
                                                                                                  version,
                                                                                                  support_versions))
        conn.close()

    def _inc_used(self):
        with self._condition:
            if self._used_conn < self._pool_size:
                self._used_conn = self._used_conn + 1
                return True

            return False

    def _dec_used(self):
        with self._condition:
            if self._used_conn == 0:
                return False
            self._used_conn -= 1
            return True

    def _full(self):
        with self._condition:
            return self._used_conn >= self._pool_size

    def _empty(self):
        with self._condition:
            return self._pool.qsize() <= 0 and self._used_conn <= 0

    def _create_connection(self):
        with self._condition:
            conn = ConnectionRecord(self._uri, conn_id=self._used_conn - 1, **self._kw)
            return ScopedConnection(self, conn)

    def _inc_connection(self):
        if self._inc_used():
            return self._create_connection()

        return self.fetch(block=True)

    def record_duration(self, conn, duration):
        if len(self.durations[conn]) >= 10000:
            self.durations[conn].pop(0)

        self.durations[conn].append(duration)

    def stats(self):
        out = {'connections': {}}
        connections = out['connections']
        take_time = []
        for conn, durations in self.durations.items():
            total_time = sum(d.value for d in durations)
            connections[id(conn)] = {
                'total_time': total_time,
                'called_times': len(durations)
            }
            take_time.append(total_time)

        out['max-time'] = max(take_time)
        out['num'] = len(self.durations)
        return out

    def count(self):
        with self._condition:
            return self._used_conn

    def activate_count(self):
        with self._condition:
            return self._used_conn - self._pool.qsize()

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
        try:
            self._pool.put(conn, False)
        except queue.Full:
            pass


class SingletonThreadPool:
    def __init__(self, uri, channel=None, pool_size=10, wait_timeout=30, try_connect=True, **kwargs):
        # Asynchronous queue to store connection
        self._uri = uri
        self._channel = channel
        self._conn = None
        self._pool_size = pool_size
        self._wait_timeout = wait_timeout

        #
        self._local = threading.local()

        # Record used connection number.
        self._kw = kwargs

        self.durations = defaultdict(list)

        self._try_connect = try_connect
        # if self._try_connect:
        #     self._max_retry = kwargs
        self._prepare()

    def _prepare(self):
        conn = self.fetch()
        if self._try_connect:
            conn.client().ping()

    def record_duration(self, conn, duration):
        pass

    def fetch(self):
        try:
            conn = self._local.conn
        except AttributeError:
            t_ident = threading.get_ident()
            conn = ConnectionRecord(self._uri, self._channel, conn_id=t_ident, **self._kw)
            self._local.conn = conn

        return SingleScopedConnection(conn)

    def release(self, conn):
        pass


class SingleConnectionPool:
    def __init__(self, uri, pool_size=10, wait_timeout=30, try_connect=True, **kwargs):
        # Asynchronous queue to store connection
        self._uri = uri
        self._conn = None
        self._pool_size = pool_size
        self._wait_timeout = wait_timeout

        # Record used connection number.
        self._condition = threading.Condition()
        self._kw = kwargs

        self.durations = defaultdict(list)

        self._try_connect = try_connect
        # if self._try_connect:
        #     self._max_retry = kwargs
        self._prepare()

    def _prepare(self):
        conn = ConnectionRecord(self._uri, conn_id=0, **self._kw)
        self._conn = SingleScopedConnection(conn)
        with self._condition:
            if self._try_connect:
                self._conn.client().ping()

            status, version = self._conn.client().server_version(timeout=30)
            if not status.OK():
                raise NotConnectError("Cannot check server version: {}".format(status.message))
            if not _is_version_match(version):
                raise VersionError(
                    "Version of python SDK({}) not match that of server{}, excepted is {}".format(__version__,
                                                                                                  version,
                                                                                                  support_versions))

    def record_duration(self, conn, duration):
        pass

    def fetch(self):
        return self._conn

    def release(self, conn):
        pass


class SingleScopedConnection:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, item):
        return getattr(self.client(), item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        self.close()

    def connection(self):
        return self._conn

    def client(self):
        conn = self.connection()
        return conn.connection()

    def close(self):
        self._conn = None


class ScopedConnection:
    def __init__(self, pool, connection):
        self._pool = pool
        self._connection = connection
        self._duration = Duration()
        self._closed = False

    def __getattr__(self, item):
        return getattr(self.client(), item)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def connection(self):
        if self._closed:
            raise ValueError("Connection has been closed.")

        return self._connection

    def client(self):
        conn = self.connection()
        return conn.connection()

    def conn_id(self):
        return self._connection._conn_id

    def close(self):
        self._closed = True
        if not self._pool:
            return

        self._connection and self._pool.release(self._connection)
        self._connection = None
        if self._duration:
            self._duration.stop()
            self._pool.record_duration(self._connection, self._duration)
        self._duration = None
