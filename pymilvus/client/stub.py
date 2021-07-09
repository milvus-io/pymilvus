# -*- coding: UTF-8 -*-

import collections
import functools
import logging
import time

import grpc
from urllib.parse import urlparse

from . import __version__
from .types import Status, DataType, DeployMode
from .check import check_pass_param, is_legal_host, is_legal_port, is_legal_index_metric_type, is_legal_binary_index_metric_type
from .pool import ConnectionPool, SingleConnectionPool, SingletonThreadPool
from .exceptions import BaseException, ParamError, DeprecatedError

from ..settings import DefaultConfig as config
from .utils import valid_binary_metric_types
from .utils import valid_index_types
from .utils import valid_binary_index_types
from .utils import valid_index_params_keys
from .utils import check_invalid_binary_vector

LOGGER = logging.getLogger(__name__)


def deprecated(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        error_str = "Function {} has been deprecated".format(func.__name__)
        LOGGER.error(error_str)
        raise DeprecatedError(error_str)

    return inner


def check_connect(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return inner


def retry_on_rpc_failure(retry_times=10, wait=1):
    def wrapper(func):
        @functools.wraps(func)
        def handler(self, *args, **kwargs):
            counter = 1
            try:
                return func(self, *args, **kwargs)
            except grpc.RpcError as e:
                # DEADLINE_EXCEEDED means that the task wat not completed
                # UNAVAILABLE means that the service is not reachable currently
                # Reference: https://grpc.github.io/grpc/python/grpc.html#grpc-status-code
                if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED and e.code() != grpc.StatusCode.UNAVAILABLE:
                    raise e
                if counter >= retry_times:
                    raise e
                time.sleep(wait)
                self._update_connection_pool()
            except Exception as e:
                raise e
            finally:
                counter += 1

        return handler

    return wrapper


def _pool_args(**kwargs):
    pool_kwargs = dict()
    for k, v in kwargs.items():
        if k in ("pool_size", "wait_timeout", "handler", "try_connect", "pre_ping", "max_retry"):
            pool_kwargs[k] = v

    return pool_kwargs


def _set_uri(host, port, uri, handler="GRPC"):
    default_port = config.GRPC_PORT if handler == "GRPC" else config.HTTP_PORT
    default_uri = config.GRPC_URI if handler == "GRPC" else config.HTTP_URI
    uri_prefix = "tcp://" if handler == "GRPC" else "http://"

    if host is not None:
        _port = port if port is not None else default_port
        _host = host
    elif port is None:
        try:
            _uri = urlparse(uri) if uri else urlparse(default_uri)
            _host = _uri.hostname
            _port = _uri.port
        except (AttributeError, ValueError, TypeError) as e:
            raise ParamError("uri is illegal: {}".format(e))
    else:
        raise ParamError("Param is not complete. Please invoke as follow:\n"
                         "\t(host = ${HOST}, port = ${PORT})\n"
                         "\t(uri = ${URI})\n")

    if not is_legal_host(_host) or not is_legal_port(_port):
        raise ParamError("host {} or port {} is illegal".format(_host, _port))

    return "{}{}:{}".format(uri_prefix, str(_host), str(_port))


class Milvus:
    def __init__(self, host=None, port=None, handler="GRPC", pool="SingletonThread", channel=None, **kwargs):
        self._name = kwargs.get('name', None)
        self._uri = None
        self._status = None
        self._connected = False
        self._handler = handler

        if handler != "GRPC":
            raise NotImplementedError("only grpc handler is supported now!")

        _uri = kwargs.get('uri', None)
        self._pool_type = pool
        self._pool_uri = _set_uri(host, port, _uri, self._handler)
        self._pool_kwargs = _pool_args(handler=handler, **kwargs)
        self._update_connection_pool(channel=channel)

        # store extra key-words arguments
        self._kw = kwargs
        self._hooks = collections.defaultdict()

        self._deploy_mode = DeployMode.Distributed

    @check_connect
    def _wait_for_healthy(self, timeout=30, retry=10):
        with self._connection() as handler:
            start_time = time.time()
            while retry > 0:
                if (time.time() - start_time > timeout):
                    break
                try:
                    status = handler.fake_register_link(timeout)
                    if status.error_code == 0:
                        self._deploy_mode = status.reason
                        return
                except:
                    pass
                finally:
                    time.sleep(1)
                    retry -= 1
            raise Exception("server is not healthy, please try again later")

    def __enter__(self):
        self._conn = self._pool.fetch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        self._conn = None

    def __del__(self):
        self._pool = None

    def _connection(self):
        if self._pool:
            return self._pool.fetch()

    def _update_connection_pool(self, channel=None):
        self._pool = None
        if self._pool_type == "QueuePool":
            self._pool = ConnectionPool(self._pool_uri, **self._pool_kwargs)
        elif self._pool_type == "SingletonThread":
            self._pool = SingletonThreadPool(self._pool_uri, channel=channel, **self._pool_kwargs)
        elif self._pool_type == "Singleton":
            self._pool = SingleConnectionPool(self._pool_uri, **self._pool_kwargs)
        else:
            raise ParamError("Unknown pool value: {}".format(self._pool_type))

        if not channel:
            self._wait_for_healthy()

    @property
    def name(self):
        return self._name

    @property
    def handler(self):
        return self._handler

    def close(self):
        """
        Close client instance
        """
        if self._pool:
            self._pool = None
            return
        raise Exception("connection was already closed!")

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def create_collection(self, collection_name, fields, timeout=None, **kwargs):
        """
        Creates a collection.

        :param collection_name: The name of the collection. A collection name can only include
        numbers, letters, and underscores, and must not begin with a number.
        :type  collection_name: str

        :param fields: Field parameters.
        :type  fields: dict

            ` {"fields": [
                    {"field": "A", "type": DataType.INT32}
                    {"field": "B", "type": DataType.INT64},
                    {"field": "C", "type": DataType.FLOAT},
                    {"field": "Vec", "type": DataType.FLOAT_VECTOR,
                     "params": {"dim": 128}}
                ],
            "auto_id": True}`

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.create_collection(collection_name, fields, timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def drop_collection(self, collection_name, timeout=None):
        """
        Deletes a specified collection.

        :param collection_name: The name of the collection to delete.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.drop_collection(collection_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def has_collection(self, collection_name, timeout=None):
        """
        Checks whether a specified collection exists.

        :param collection_name: The name of the collection to check.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: If specified collection exists
        :rtype: bool

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.has_collection(collection_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def describe_collection(self, collection_name, timeout=None):
        """
        Returns the schema of specified collection.
        Example: {'collection_name': 'create_collection_eXgbpOtn', 'auto_id': True, 'description': '',
                 'fields': [{'field_id': 100, 'name': 'INT32', 'description': '', 'type': 4, 'params': {},
                 {'field_id': 101, 'name': 'FLOAT_VECTOR', 'description': '', 'type': 101,
                 'params': {'dim': '128'}}]}

        :param collection_name: The name of the collection to describe.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: The schema of collection to describe.
        :rtype: dict

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.describe_collection(collection_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def load_collection(self, collection_name, timeout=None):
        """
        Loads a specified collection from disk to memory.

        :param collection_name: The name of the collection to load.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.load_collection("", collection_name=collection_name, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def release_collection(self, collection_name, timeout=None):
        """
        Clear collection data from memory.

        :param collection_name: The name of collection to release.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.release_collection(db_name="", collection_name=collection_name, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        """
        Returns collection statistics information.
        Example: {"row_count": 10}

        :param collection_name: The name of collection.
        :type  collection_name: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: statistics information
        :rtype: dict

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        with self._connection() as handler:
            stats = handler.get_collection_stats(collection_name, timeout, **kwargs)
            result = {stat.key: stat.value for stat in stats}
            result["row_count"] = int(result["row_count"])
            return result

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def list_collections(self, timeout=None):
        """
        Returns a list of all collection names.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: List of collection names, return when operation is successful
        :rtype: list[str]

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.list_collections(timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def create_partition(self, collection_name, partition_name, timeout=None):
        """
        Creates a partition in a specified collection. You only need to import the
        parameters of partition_name to create a partition. A collection cannot hold
        partitions of the same tag, whilst you can insert the same tag in different collections.

        :param collection_name: The name of the collection to create partitions in.
        :type  collection_name: str

        :param partition_name: The tag name of the partition to create.
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        with self._connection() as handler:
            return handler.create_partition(collection_name, partition_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def drop_partition(self, collection_name, partition_name, timeout=None):
        """
        Deletes the specified partition in a collection. Note that the default partition
        '_default' is not permitted to delete. When a partition deleted, all data stored in it
        will be deleted.

        :param collection_name: The name of the collection to delete partitions from.
        :type  collection_name: str

        :param partition_name: The tag name of the partition to delete.
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        with self._connection() as handler:
            return handler.drop_partition(collection_name, partition_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def has_partition(self, collection_name, partition_name, timeout=None):
        """
        Checks if a specified partition exists in a collection.

        :param collection_name: The name of the collection to find the partition in.
        :type  collection_name: str

        :param partition_name: The tag name of the partition to check
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: Whether a specified partition exists in a collection.
        :rtype: bool

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        with self._connection() as handler:
            return handler.has_partition(collection_name, partition_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def load_partitions(self, collection_name, partition_names, timeout=None):
        """
        Load specified partitions from disk to memory.

        :param collection_name: The collection name which partitions belong to.
        :type  collection_name: str

        :param partition_names: The specified partitions to load.
        :type  partition_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.load_partitions(db_name="", collection_name=collection_name,
                                           partition_names=partition_names, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def release_partitions(self, collection_name, partition_names, timeout=None):
        """
        Clear partitions data from memory.

        :param collection_name: The collection name which partitions belong to.
        :type  collection_name: str

        :param partition_names: The specified partition to release.
        :type  partition_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.release_partitions(db_name="", collection_name=collection_name,
                                              partition_names=partition_names, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def list_partitions(self, collection_name, timeout=None):
        """
        Returns a list of all partition tags in a specified collection.

        :param collection_name: The name of the collection to retrieve partition tags from.
        :type  collection_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: A list of all partition tags in specified collection.
        :rtype: list[str]

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)

        with self._connection() as handler:
            return handler.list_partitions(collection_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def get_partition_stats(self, collection_name, partition_name, timeout=None, **kwargs):
        """
        Returns partition statistics information.
        Example: {"row_count": 10}

        :param collection_name: The name of collection.
        :type  collection_name: str.

        :param partition_name: The name of partition.
        :type  partition_name: str.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: statistics information
        :rtype: dict

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            stats = handler.get_partition_stats(collection_name, partition_name, timeout, **kwargs)
            result = {stat.key: stat.value for stat in stats}
            result["row_count"] = int(result["row_count"])
            return result

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        """
        Creates an index for a field in a specified collection. Milvus does not support creating multiple
        indexes for a field. In a scenario where the field already has an index, if you create another one,
        the server will replace the existing index files with the new ones.

        Note that you need to call load_collection() or load_partitions() to make the new index take effect
        on searching tasks.

        :param collection_name: The name of the collection to create field indexes.
        :type  collection_name: str

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param params: Indexing parameters.
        :type  params: dict
            There are examples of supported indexes:

            IVF_FLAT:
                ` {
                    "metric_type":"L2",
                    "index_type": "IVF_FLAT",
                    "params":{"nlist": 1024}
                }`

            IVF_PQ:
                `{
                    "metric_type": "L2",
                    "index_type": "IVF_PQ",
                    "params": {"nlist": 1024, "m": 8, "nbits": 8}
                }`

            IVF_SQ8:
                `{
                    "metric_type": "L2",
                    "index_type": "IVF_SQ8",
                    "params": {"nlist": 1024}
                }`

            BIN_IVF_FLAT:
                `{
                    "metric_type": "JACCARD",
                    "index_type": "BIN_IVF_FLAT",
                    "params": {"nlist": 1024}
                }`

            HNSW:
                `{
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {"M": 48, "efConstruction": 50}
                }`

            RHNSW_FLAT:
                `{
                    "metric_type": "L2",
                    "index_type": "RHNSW_FLAT",
                    "params": {"M": 48, "efConstruction": 50}
                }`

            RHNSW_PQ:
                `{
                    "metric_type": "L2",
                    "index_type": "RHNSW_PQ",
                    "params": {"M": 48, "efConstruction": 50, "PQM": 8}
                }`

            RHNSW_SQ:
                `{
                    "metric_type": "L2",
                    "index_type": "RHNSW_SQ",
                    "params": {"M": 48, "efConstruction": 50}
                }`

            ANNOY:
                `{
                    "metric_type": "L2",
                    "index_type": "ANNOY",
                    "params": {"n_trees": 8}
                }`

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a IndexFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        # params preliminary validate
        if 'index_type' not in params:
            raise ParamError("Params must contains key: 'index_type'")
        if 'params' not in params:
            raise ParamError("Params must contains key: 'params'")
        if 'metric_type' not in params:
            raise ParamError("Params must contains key: 'metric_type'")
        if not isinstance(params['params'], dict):
            raise ParamError("Params['params'] must be a dictionary type")
        if params['index_type'] not in valid_index_types:
            raise ParamError("Invalid index_type: " + params['index_type'] +
                             ", which must be one of: " + str(valid_index_types))
        for k in params['params'].keys():
            if k not in valid_index_params_keys:
                raise ParamError("Invalid params['params'].key: " + k)
        for v in params['params'].values():
            if not isinstance(v, int):
                raise ParamError("Invalid params['params'].value: " + v + ", which must be an integer")

        # filter invalid metric type
        if params['index_type'] in valid_binary_index_types:
            if not is_legal_binary_index_metric_type(params['index_type'], params['metric_type']):
                raise ParamError("Invalid metric_type: " + params['metric_type'] +
                                 ", which does not match the index type: " + params['index_type'])
        else:
            if not is_legal_index_metric_type(params['index_type'], params['metric_type']):
                raise ParamError("Invalid metric_type: " + params['metric_type'] +
                                 ", which does not match the index type: " + params['index_type'])
        with self._connection() as handler:
            return handler.create_index(collection_name, field_name, params, timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def drop_index(self, collection_name, field_name, timeout=None):
        """
        Removes the index of a field in a specified collection.

        :param collection_name: The name of the collection to remove the field index from.
        :type  collection_name: str

        :param field_name: The name of the field to remove the index of.
        :type  field_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        check_pass_param(field_name=field_name)
        with self._connection() as handler:
            return handler.drop_index(collection_name=collection_name,
                                      field_name=field_name, index_name="_default_idx", timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def describe_index(self, collection_name, index_name="", timeout=None):
        """
        Returns the schema of index built on specified field.
        Example: {'index_type': 'FLAT', 'metric_type': 'L2', 'params': {'nlist': 128}}

        :param collection_name: The name of the collection which field belong to.
        :type  collection_name: str

        :param field_name: The name of field to describe.
        :type  field_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: the schema of index built on specified field.
        :rtype: dict

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.describe_index(collection_name, index_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        """
        Inserts entities in a specified collection.

        :param collection_name: The name of the collection to insert entities in.
        :type  collection_name: str.

        :param entities: The entities to insert.
        :type  entities: list

        :param partition_name: The name of the partition to insert entities in. The default value is
         None. The server stores entities in the “_default” partition by default.
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a MutationFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: list of ids of the inserted vectors.
        :rtype: list[int]

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        # filter invalid binary data #1352: https://github.com/zilliztech/milvus-distributed/issues/1352
        if not check_invalid_binary_vector(entities):
            raise ParamError("Invalid binary vector data exists")

        with self._connection() as handler:
            return handler.bulk_insert(collection_name, entities, partition_name, timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def flush(self, collection_names=None, timeout=None, **kwargs):
        """
        Internally, Milvus organizes data into segments, and indexes are built in a per-segment manner.
        By default, a segment will be sealed if it grows large enough (according to segment size configuration).
        If any index is specified on certain field, the index-creating task will be triggered automatically
        when a segment is sealed.

        The flush() call will seal all the growing segments immediately of the given collection,
        and force trigger the index-creating tasks.

        :param collection_names: The name of collection to flush.
        :type  collection_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a FlushFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: None
        :rtype: NoneType

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        if collection_names in (None, []):
            raise ParamError("Collection name list can not be None or empty")

        if not isinstance(collection_names, list):
            raise ParamError("Collection name array must be type of list")

        if len(collection_names) <= 0:
            raise ParamError("Collection name array is not allowed to be empty")

        for name in collection_names:
            check_pass_param(collection_name=name)
        with self._connection() as handler:
            return handler.flush(collection_names, timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def search(self, collection_name, dsl, partition_names=None, fields=None, timeout=None, **kwargs):
        """
        Searches a collection based on the given DSL clauses and returns query results.

        :param collection_name: The name of the collection to search.
        :type  collection_name: str

        :param dsl: The DSL that defines the query.
        :type  dsl: dict

            ` {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "A": {
                                    "GT": 1,
                                    "LT": "100"
                                }
                            }
                        },
                        {
                            "vector": {
                                "Vec": {
                                    "metric_type": "L2",
                                    "params": {
                                        "nprobe": 10
                                    },
                                    "query": vectors,
                                    "topk": 10
                                }
                            }
                        }
                    ]
                }
            }`

        :param partition_names: The tags of partitions to search.
        :type  partition_names: list[str]

        :param fields: The fields to return in the search result
        :type  fields: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a SearchFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: Query result. QueryResult is iterable and is a 2d-array-like class, the first dimension is
                 the number of vectors to query (nq), the second dimension is the number of topk.
        :rtype: QueryResult

        Suppose the nq in dsl is 4, topk in dsl is 10:
        :example:
        >>> client = Milvus(host='localhost', port='19530')
        >>> result = client.search(collection_name, dsl)
        >>> print(len(result))
        4
        >>> print(len(result[0]))
        10
        >>> print(len(result[0].ids))
        10
        >>> result[0].ids
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> len(result[0].distances)
        10
        >>> result[0].distances
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> top1 = result[0][0]
        >>> top1.id
        0
        >>> top1.distance
        0.1
        >>> top1.score # now, the score is equal to distance
        0.1

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        with self._connection() as handler:
            kwargs["_deploy_mode"] = self._deploy_mode
            return handler.search(collection_name, dsl, partition_names, fields, timeout=timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def search_with_expression(self, collection_name, data, anns_field, param, limit, expression=None, partition_names=None,
                               output_fields=None, timeout=None, **kwargs):
        """
        Searches a collection based on the given expression and returns query results.

        :param collection_name: The name of the collection to search.
        :type  collection_name: str
        :param data: The vectors of search data, the length of data is number of query (nq), the dim of every vector in
                     data must be equal to vector field's of collection.
        :type  data: list[list[float]]
        :param anns_field: The vector field used to search of collection.
        :type  anns_field: str
        :param param: The parameters of search, such as nprobe, etc.
        :type  param: dict
        :param limit: The max number of returned record, we also called this parameter as topk.
        :type  limit: int
        :param expression: The boolean expression used to filter attribute.
        :type  expression: str
        :param partition_names: The names of partitions to search.
        :type  partition_names: list[str]
        :param output_fields: The fields to return in the search result, not supported now.
        :type  output_fields: list[str]
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a SearchFuture object;
              otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only take
              effect when _async is set to True.

        :return: Query result. QueryResult is iterable and is a 2d-array-like class, the first dimension is
                 the number of vectors to query (nq), the second dimension is the number of limit(topk).
        :rtype: QueryResult

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        check_pass_param(limit=limit)
        with self._connection() as handler:
            kwargs["_deploy_mode"] = self._deploy_mode
            return handler.search_with_expression(collection_name, data, anns_field, param, limit, expression, partition_names, output_fields, timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def calc_distance(self, vectors_left, vectors_right, params=None, timeout=None, **kwargs):
        """
        Calculate distance between two vector arrays.

        :param vectors_left: The vectors on the left of operator.
        :type  vectors_left: dict
        `{"ids": [1, 2, 3, .... n], "collection": "c_1", "partition": "p_1", "field": "v_1"}`
        or
        `{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]}`
        or
        `{"bin_vectors": [b'\x94', b'N', ... b'\xca']}`

        :param vectors_right: The vectors on the right of operator.
        :type  vectors_right: dict
        `{"ids": [1, 2, 3, .... n], "collection": "col_1", "partition": "p_1", "field": "v_1"}`
        or
        `{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]}`
        or
        `{"bin_vectors": [b'\x94', b'N', ... b'\xca']}`

        :param params: parameters, currently only support "metric_type", default value is "L2"
                       extra parameter for "L2" distance: "sqrt", true or false, default is false
                       extra parameter for "HAMMING" and "TANIMOTO": "dim", set this value if dimension is not a multiple of 8, otherwise the dimension will be calculted by list length
        :type  params: dict
            There are examples of supported metric_type:
                `{"metric": "L2"}`
                `{"metric": "IP"}`
                `{"metric": "HAMMING"}`
                `{"metric": "TANIMOTO"}`
            Note: "L2", "IP", "HAMMING", "TANIMOTO" are case insensitive

        :return: 2-d array distances
        :rtype: list[list[int]] for "HAMMING" or list[list[float]] for others
            Assume the vectors_left: L_1, L_2, L_3
            Assume the vectors_right: R_a, R_b
            Distance between L_n and R_m we called "D_n_m"
            The returned distances are arranged like this:
              [D_1_a, D_1_b, D_2_a, D_2_b, D_3_a, D_3_b]

        """
        with self._connection() as handler:
            return handler.calc_distance(vectors_left, vectors_right, params, timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def load_collection_progress(self, collection_name, timeout=None):
        with self._connection() as handler:
            return handler.load_collection_progress(collection_name, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def load_partitions_progress(self, collection_name, partition_names, timeout=None):
        with self._connection() as handler:
            return handler.load_partitions_progress(collection_name, partition_names, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def wait_for_loading_collection_complete(self, collection_name, timeout=None):
        with self._connection() as handler:
            return handler.wait_for_loading_collection(collection_name, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def wait_for_loading_partitions_complete(self, collection_name, partition_names, timeout=None):
        with self._connection() as handler:
            return handler.wait_for_loading_partitions(collection_name, partition_names, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def get_index_build_progress(self, collection_name, index_name, timeout=None):
        with self._connection() as handler:
            return handler.get_index_build_progress(collection_name, index_name, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def wait_for_creating_index(self, collection_name, index_name, timeout=None):
        with self._connection() as handler:
            return handler.wait_for_creating_index(collection_name, index_name, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def dummy(self, request_type, timeout=None):
        with self._connection() as handler:
            return handler.dummy(request_type, timeout=timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @check_connect
    def query(self, collection_name, expr, output_fields=None, partition_names=None, timeout=None):
        """
        Query with a set of criteria, and results in a list of records that match the query exactly.

        :param collection_name: Name of the collection to retrieve entities from
        :type  collection_name: str

        :param expr: The query expression
        :type  expr: str

        :param output_fields: A list of fields to return
        :type  output_fields: list[str]

        :param partition_names: Name of partitions that contain entities
        :type  partition_names: list[str]

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :return: A list that contains all results
        :rtype: list

        :raises:
            RpcError: If gRPC encounter an error
            ParamError: If parameters are invalid
            BaseException: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.query(collection_name, expr, output_fields, partition_names, timeout=timeout)
