# -*- coding: UTF-8 -*-

import collections
import copy
import functools
import logging

from urllib.parse import urlparse

from . import __version__
from .types import IndexType, MetricType, Status
from .check import check_pass_param, is_legal_host, is_legal_port
from .pool import ConnectionPool, SingleConnectionPool, SingletonThreadPool
from .exceptions import ParamError, DeprecatedError

from ..settings import DefaultConfig as config

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


def _pool_args(**kwargs):
    pool_kwargs = dict()
    for k, v in kwargs.items():
        if k in ("pool_size", "wait_timeout", "handler", "try_connect", "pre_ping", "max_retry"):
            pool_kwargs[k] = v

    return pool_kwargs


def _set_uri(host, port, uri, handler="GRPC"):
    default_port = config.GRPC_PORT if handler == "GRPC" else config.HTTP_PORT
    default_uri = config.GRPC_URI if handler == "GRPC" else config.HTTP_URI

    if host is not None:
        _port = port if port is not None else default_port
        _host = host
        if handler == "HTTP":
            _proto = "https" if _port == 443 else "http"
        else:
            _proto = "tcp"
    elif port is None:
        try:
            _uri = urlparse(uri) if uri else urlparse(default_uri)
            _host = _uri.hostname
            _port = _uri.port
            _proto = _uri.scheme
        except (AttributeError, ValueError, TypeError) as e:
            raise ParamError("uri is illegal: {}".format(e))
    else:
        raise ParamError("Param is not complete. Please invoke as follow:\n"
                         "\t(host = ${HOST}, port = ${PORT})\n"
                         "\t(uri = ${URI})\n")

    if not is_legal_host(_host) or not is_legal_port(_port):
        raise ParamError("host {} or port {} is illegal".format(_host, _port))

    return "{}://{}:{}".format(str(_proto), str(_host), str(_port))


class Milvus:
    def __init__(self, host=None, port=None, handler="GRPC", pool="SingletonThread", **kwargs):
        """Constructor method
        """
        self._name = kwargs.get('name', None)
        self._uri = None
        self._status = None
        self._connected = False
        self._handler = handler

        #
        self._conn = None

        _uri = kwargs.get('uri', None)
        pool_uri = _set_uri(host, port, _uri, self._handler)
        pool_kwargs = _pool_args(handler=handler, **kwargs)
        # self._pool = SingleConnectionPool(pool_uri, **pool_kwargs)
        if pool == "QueuePool":
            self._pool = ConnectionPool(pool_uri, **pool_kwargs)
        elif pool == "SingletonThread":
            self._pool = SingletonThreadPool(pool_uri, **pool_kwargs)
        elif pool == "Singleton":
            self._pool = SingleConnectionPool(pool_uri, **pool_kwargs)
        else:
            raise ParamError("Unknown pool value: {}".format(pool))

        # store extra key-words arguments
        self._kw = kwargs
        self._hooks = collections.defaultdict()

    def __enter__(self):
        self._conn = self._pool.fetch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._conn.close()
        self._conn = None

    def __del__(self):
        return self.close()

    def _connection(self):
        return self._pool.fetch()

    @deprecated
    def set_hook(self, **kwargs):
        """
        Deprecated
        """
        # TODO: may remove it.
        if self._stub:
            self._stub.set_hook(**kwargs)
        else:
            self._hooks.update(kwargs)

    @property
    def name(self):
        return self._name

    @property
    def handler(self):
        return self._handler

    @deprecated
    def connect(self, host=None, port=None, uri=None, timeout=2):
        """
        Deprecated
        """
        if self.connected() and self._connected:
            return Status(message="You have already connected {} !".format(self._uri),
                          code=Status.CONNECT_FAILED)

        if self._stub is None:
            self._init(host, port, uri, handler=self._handler)

        if self.ping(timeout):
            self._status = Status(message="Connected")
            self._connected = True
            return self._status

        return Status()

    @deprecated
    def connected(self):
        """
        Deprecated
        """
        return self._status and self._status.OK()

    @deprecated
    def disconnect(self):
        """
        Deprecated
        """
        pass

    def close(self):
        """
        Close client instance
        """
        self._pool = None

    def client_version(self):
        """
        Returns the version of the client.

        :return: Version of the client.

        :rtype: (str)
        """
        return __version__

    def server_status(self, timeout=30):
        """
        Returns the status of the Milvus server.

        :return:
            Status: Whether the operation is successful.

            str : Status of the Milvus server.

        :rtype: (Status, str)
        """
        return self._cmd("status", timeout)

    def server_version(self, timeout=30):
        """
        Returns the version of the Milvus server.

        :return:
           Status: Whether the operation is successful.

           str : Version of the Milvus server.

        :rtype: (Status, str)
        """

        return self._cmd("version", timeout)

    @check_connect
    def _cmd(self, cmd, timeout=30):
        check_pass_param(cmd=cmd)

        with self._connection() as handler:
            return handler._cmd(cmd, timeout)

    @check_connect
    def create_collection(self, param, timeout=30):
        """
        Creates a collection.

        :type  param: dict
        :param param: Information needed to create a collection. It contains items:

            * *collection_name* (``str``) -- Collection name.
            * *dimension* (``int``) -- Dimension of embeddings stored in collection.
            * *index_file_size* (``int``) -- Segment size. See
              `Storage Concepts <https://milvus.io/docs/v1.0.0/storage_concept.md>`_.
            * *metric_type* (``MetricType``) -- Distance Metrics type. Valued form
              :class:`~milvus.MetricType`. See
              `Distance Metrics <https://milvus.io/docs/v1.0.0/metric.md>`_.


            A demo is as follow:

            .. code-block:: python

                param={'collection_name': 'name',
                   'dimension': 16,
                   'index_file_size': 1024 # Optional, default 1024，
                   'metric_type': MetricType.L2 # Optional, default MetricType.L2
                  }


        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        if not isinstance(param, dict):
            raise ParamError('Param type incorrect, expect {} but get {} instead'
                             .format(type(dict), type(param)))

        collection_param = copy.deepcopy(param)

        if 'collection_name' not in collection_param:
            raise ParamError('collection_name is required')
        collection_name = collection_param["collection_name"]
        collection_param.pop('collection_name')

        if 'dimension' not in collection_param:
            raise ParamError('dimension is required')
        dim = collection_param["dimension"]
        collection_param.pop("dimension")

        index_file_size = collection_param.get('index_file_size', 1024)
        collection_param.pop('index_file_size', None)

        metric_type = collection_param.get('metric_type', MetricType.L2)
        collection_param.pop('metric_type', None)

        check_pass_param(collection_name=collection_name, dimension=dim,
                         index_file_size=index_file_size, metric_type=metric_type)

        with self._connection() as handler:
            return handler.create_collection(collection_name, dim, index_file_size,
                                             metric_type, collection_param, timeout)

    @check_connect
    def has_collection(self, collection_name, timeout=30):
        """

        Checks whether a collection exists.

        :param collection_name: Name of the collection to check.
        :type  collection_name: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and the flag indicating if collection exists. Succeed
                 if `Status.OK()` is `True`. If status is not OK, the flag is always `False`.
        :rtype: Status, bool

        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.has_collection(collection_name, timeout)

    @check_connect
    def get_collection_info(self, collection_name, timeout=30):
        """
        Returns information of a collection.

        :type  collection_name: str
        :param collection_name: Name of the collection to describe.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and collection information. Succeed if `Status.OK()`
                 is `True`. If status is not OK, the returned information is always `None`.
        :rtype: Status, CollectionSchema
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.describe_collection(collection_name, timeout)

    @check_connect
    def count_entities(self, collection_name, timeout=30):
        """
        Returns the number of vectors in a collection.

        :type  collection_name: str
        :param collection_name: target table name.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and row count. Succeed if `Status.OK()` is `True`.
                 If status is not OK, the returned value of is always `None`.
        :rtype: Status, int
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.count_collection(collection_name, timeout)

    @check_connect
    def list_collections(self, timeout=30):
        """
        Returns collection list.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and collection name list. Succeed if `Status.OK()` is `True`.
                 If status is not OK, the returned name list is always `[]`.
        :rtype: Status, list[str]
        """
        with self._connection() as handler:
            return handler.show_collections(timeout)

    @check_connect
    def get_collection_stats(self, collection_name, timeout=30):
        """

        Returns collection statistics information.

        :type  collection_name: str
        :param collection_name: target table name.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and collection statistics information. Succeed
                 if `Status.OK()` is `True`. If status is not OK, the returned information
                 is always `[]`.
        :rtype: Status, dict
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.show_collection_info(collection_name, timeout)

    @check_connect
    def load_collection(self, collection_name, partition_tags=None, timeout=None):
        """
        Loads a collection for caching.

        :param collection_name: collection to load
        :type collection_name: str
        :param partition_tags: partition tag list. `None` indicates to load whole collection,
                               otherwise to load specified partitions.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.preload_collection(collection_name, partition_tags, timeout)

    @check_connect
    def release_collection(self, collection_name, partition_tags=None, timeout=None):
        """
        Release a collection from memory and cache.

        :param collection_name: collection to release
        :type collection_name: str
        :param partition_tags: partition tag list. `None` indicates to release whole collection,
                               otherwise to release specified partitions.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.release_collection(collection_name, partition_tags, timeout)

    @check_connect
    def reload_segments(self, collection_name, segment_ids, timeout=None):
        """
        Reloads segment DeletedDocs data to cache. This API is not recommended for users.

        :param collection_name: Name of the collection being deleted
        :type  collection_name: str
        :param segment_ids: Segment IDs.
        :type segment_ids: list[str]
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.reload_segments(collection_name, segment_ids, timeout)

    @check_connect
    def drop_collection(self, collection_name, timeout=30):
        """
        Deletes a collection by name.

        :param collection_name: Name of the collection being deleted
        :type  collection_name: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.drop_collection(collection_name, timeout)

    @check_connect
    def insert(self, collection_name, records, ids=None, partition_tag=None,
               params=None, timeout=None, **kwargs):
        """
        Insert vectors to a collection.

        :type  collection_name: str
        :param collection_name: Name of the collection to insert vectors to.
        :param ids: ID list. `None` indicates ID is generated by server system. Note that if the
                    first time when insert() is invoked ids is not passed into this method, each
                    of the rest time when inset() is invoked ids is not permitted to pass,
                    otherwise server will return an error and the insertion process will fail.
                    And vice versa.
        :type  ids: list[int]
        :param records: List of vectors to insert.
        :type  records: list[list[float]]
        :param partition_tag: Tag of a partition.
        :type partition_tag: str or None. If partition_tag is None, vectors will be inserted to the
                             default partition `_default`.
        :param params: Insert param. Reserved.
        :type params: dict
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a InsertFuture
              object; otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :return: The operation status and IDs of inserted entities. Succeed if `Status.OK()`
                 is `True`. If status is not OK, the returned IDs is always `[]`.
        :rtype: Status, list[int]
        """
        if kwargs.get("insert_param", None) is not None:
            with self._connection() as handler:
                return handler.insert(None, None, timeout=timeout, **kwargs)

        check_pass_param(collection_name=collection_name, records=records)
        _ = partition_tag is not None and check_pass_param(partition_tag=partition_tag)
        if ids is not None:
            check_pass_param(ids=ids)
            if len(records) != len(ids):
                raise ParamError("length of vectors do not match that of ids")

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.insert(collection_name, records, ids, partition_tag,
                                  params, timeout, **kwargs)

    def get_entity_by_id(self, collection_name, ids, timeout=None, partition_tag=None):
        """
        Returns raw vectors according to ids.

        :param collection_name: Name of the collection
        :type collection_name: str

        :param ids: list of vector id
        :type ids: list
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :param partition_tag: The partition tag of entity
        :type partition_tag: str

        :return: The operation status and entities. Succeed if `Status.OK()` is `True`.
                 If status is not OK, the returned entities is always `[]`.
        :rtype: Status, list[list[float]]
        """
        check_pass_param(collection_name=collection_name, ids=ids)
        _ = partition_tag is None or check_pass_param(partition_tag=partition_tag)

        with self._connection() as handler:
            return handler.get_vectors_by_ids(collection_name, ids, timeout=timeout,
                                              partition_tag=partition_tag)

    @check_connect
    def list_id_in_segment(self, collection_name, segment_name, timeout=None):
        """
        Get IDs of entity stored in the specified segment.

        :param collection_name: Collection the segment belongs to.
        :type collection_name: str
        :param segment_name: Segment name.
        :type segment_name: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and entity IDs. Succeed if `Status.OK()` is `True`.
                 If status is not OK, the returned IDs is always `[]`.
        :rtype: Status, list[int]
        """
        check_pass_param(collection_name=collection_name)
        check_pass_param(collection_name=segment_name)
        with self._connection() as handler:
            return handler.get_vector_ids(collection_name, segment_name, timeout)

    @check_connect
    def create_index(self, collection_name, index_type=None, params=None, timeout=None, **kwargs):
        """
        Creates index for a collection.

        :param collection_name: Collection used to create index.
        :type collection_name: str
        :param index_type: index params. See `index params <param.html>`_ for supported indexes.
        :type index_type: IndexType
        :param params: Index param. See `index params <param.html>`_ for detailed index param of
                       supported indexes.
        :type params: dict
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a IndexFuture
              object; otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        _index_type = IndexType.FLAT if index_type is None else index_type
        check_pass_param(collection_name=collection_name, index_type=_index_type)

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.create_index(collection_name, _index_type, params, timeout, **kwargs)

    @check_connect
    def get_index_info(self, collection_name, timeout=30):
        """
        Show index information of a collection.

        :type collection_name: str
        :param collection_name: table name been queried
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status and index info. Succeed if `Status.OK()` is `True`.
                 If status is not OK, the returned index info is always `None`.
        :rtype: Status, IndexParam

        """
        check_pass_param(collection_name=collection_name)

        with self._connection() as handler:
            return handler.describe_index(collection_name, timeout)

    @check_connect
    def drop_index(self, collection_name, timeout=30):
        """
        Removes an index.

        :param collection_name: target collection name.
        :type collection_name: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)

        with self._connection() as handler:
            return handler.drop_index(collection_name, timeout)

    @check_connect
    def create_partition(self, collection_name, partition_tag, timeout=30):
        """
        create a partition for a collection.

        :param collection_name: Name of the collection.
        :type  collection_name: str
        :param partition_tag: Name of the partition tag.
        :type  partition_tag: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.create_partition(collection_name, partition_tag, timeout)

    @check_connect
    def has_partition(self, collection_name, partition_tag, timeout=30):
        """
        Check if specified partition exists.

        :param collection_name: target table name.
        :type  collection_name: str
        :param partition_tag: partition tag.
        :type  partition_tag: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :returns: The operation status and a flag indicating if partition exists. Succeed
                  if `Status.OK()` is `True`. If status is not ok, the flag is always `False`.
        :rtype: Status, bool

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.has_partition(collection_name, partition_tag, timeout)

    @check_connect
    def list_partitions(self, collection_name, timeout=30):
        """
        Show all partitions in a collection.

        :param collection_name: target table name.
        :type  collection_name: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :returns: The operation status and partition list. Succeed if `Status.OK()` is `True`.
                  If status is not OK, returned partition list is `[]`.
        :rtype: Status, list[PartitionParam]

        """
        check_pass_param(collection_name=collection_name)

        with self._connection() as handler:
            return handler.show_partitions(collection_name, timeout)

    @check_connect
    def drop_partition(self, collection_name, partition_tag, timeout=30):
        """
        Deletes a partition in a collection.

        :param collection_name: Collection name.
        :type  collection_name: str
        :param partition_tag: Partition name.
        :type  partition_tag: str
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float

        :return: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.drop_partition(collection_name, partition_tag, timeout)

    @check_connect
    def search(self, collection_name, top_k, query_records, partition_tags=None,
               params=None, timeout=None, **kwargs):
        """
        Search vectors in a collection.

        :param collection_name: Name of the collection.
        :type  collection_name: str
        :param top_k: number of vectors which is most similar with query vectors
        :type  top_k: int
        :param query_records: vectors to query
        :type  query_records: list[list[float32]]
        :param partition_tags: tags to search. `None` indicates to search in whole collection.
        :type  partition_tags: list
        :param params: Search params. The params is related to index type the collection is built.
                       See `index params <param.html>`_ for more detailed information.
        :type  params: dict
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a SearchFuture
              object; otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :returns: The operation status and search result. See <a>here</a> to find how to handle
                  search result. Succeed if `Status.OK()` is `True`. If status is not OK,
                  results is always `None`.
        :rtype: Status, TopKQueryResult
        """
        check_pass_param(collection_name=collection_name, topk=top_k, records=query_records)
        if partition_tags is not None:
            check_pass_param(partition_tag_array=partition_tags)

        params = dict() if params is None else params
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.search(collection_name, top_k, query_records,
                                  partition_tags, params, timeout, **kwargs)

    @check_connect
    def search_in_segment(self, collection_name, file_ids, query_records, top_k,
                          params=None, timeout=None, **kwargs):
        """
        Searches for vectors in specific segments of a collection.
        This API is not recommended for users.

        The Milvus server stores vector data into multiple files. Searching for vectors in specific
        files is a method used in Mishards. Obtain more detail about Mishards, see
        `Mishards <https://github.com/milvus-io/milvus/tree/master/shards>`_.

        :param collection_name: table name been queried
        :type  collection_name: str
        :param file_ids: Specified files id array
        :type  file_ids: list[str] or list[int]
        :param query_records: all vectors going to be queried
        :type  query_records: list[list[float]]
        :param top_k: how many similar vectors will be searched
        :type  top_k: int
        :param params: Search params. The params is related to index type the collection is built.
                       See <a></a> for more detailed information.
        :type  params: dict
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a SearchFuture
              object; otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :returns: The operation status and search result. See <a>here</a> to find how to handle
                  search result. Succeed if `Status.OK()` is `True`. If status is not OK, results
                  is always `None`.
        :rtype: Status, TopKQueryResult
        """
        check_pass_param(collection_name=collection_name, topk=top_k,
                         records=query_records, ids=file_ids)

        params = dict() if params is None else params
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.search_in_files(collection_name, file_ids,
                                           query_records, top_k, params, timeout, **kwargs)

    @check_connect
    def delete_entity_by_id(self, collection_name, id_array, timeout=None, partition_tag=None):
        """
        Deletes vectors in a collection by vector ID.

        :param collection_name: Name of the collection.
        :type  collection_name: str
        :param id_array: list of vector id
        :type  id_array: list[int]
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param partition_tag: The partition tag of entity
        :type partition_tag: str

        :returns: The operation status. If the specified ID doesn't exist, Milvus server skip it
                  and try to delete next entities, which is regard as one successful operation.
                  Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name, ids=id_array)
        _ = partition_tag is None or check_pass_param(partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.delete_by_id(collection_name, id_array, timeout, partition_tag)

    @check_connect
    def flush(self, collection_name_array=None, timeout=None, **kwargs):
        """
        Flushes vector data in one collection or multiple collections to disk.

        :type  collection_name_array: list
        :param collection_name_array: Name of one or multiple collections to flush.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a FlushFuture
              object; otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :returns: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """

        if collection_name_array in (None, []):
            with self._connection() as handler:
                return handler.flush([], timeout)

        if not isinstance(collection_name_array, list):
            raise ParamError("Collection name array must be type of list")

        if len(collection_name_array) <= 0:
            raise ParamError("Collection name array is not allowed to be empty")

        for name in collection_name_array:
            check_pass_param(collection_name=name)
        with self._connection() as handler:
            return handler.flush(collection_name_array, timeout, **kwargs)

    @check_connect
    def compact(self, collection_name, timeout=None, **kwargs):
        """
        Compacts segments in a collection. This function is recommended after deleting vectors.

        :type  collection_name: str
        :param collection_name: Name of the collections to compact.
        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server responses or error occurs.
        :type  timeout: float
        :param kwargs:
            * *_async* (``bool``) --
              Indicate if invoke asynchronously. When value is true, method returns a CompactFuture
              object; otherwise, method returns results from server.
            * *_callback* (``function``) --
              The callback function which is invoked after server response successfully. It only
              takes effect when _async is set to True.

        :returns: The operation status. Succeed if `Status.OK()` is `True`.
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.compact(collection_name, timeout, **kwargs)

    def get_config(self, parent_key, child_key):
        """
        Gets Milvus configurations.

        """
        cmd = "get_config {}.{}".format(parent_key, child_key)

        return self._cmd(cmd)

    def set_config(self, parent_key, child_key, value):
        """
        Sets Milvus configurations.

        """
        cmd = "set_config {}.{} {}".format(parent_key, child_key, value)

        return self._cmd(cmd)
