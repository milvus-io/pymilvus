# -*- coding: UTF-8 -*-

import collections
import copy
import functools
import logging
import threading

from urllib.parse import urlparse

from . import __version__
from .types import IndexType, MetricType, Status
from .check import check_pass_param, is_legal_host, is_legal_port
from .pool import ConnectionPool, SingleConnectionPool, SingletonThreadPool
from .grpc_handler import GrpcHandler
from .http_handler import HttpHandler
from .exceptions import ParamError, NotConnectError, DeprecatedError

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
    def __init__(self, host=None, port=None, handler="GRPC", pool="SingletonThread", **kwargs):
        self._name = kwargs.get('name', None)
        self._uri = None
        self._status = None
        self._connected = False
        self._handler = handler

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
            return self._stub.set_hook(**kwargs)

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

    @deprecated
    def connected(self):
        """
        Deprecated
        """
        return True if self._status and self._status.OK() else False

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
        :param param: Information needed to create a collection.

                `param={'collection_name': 'name',
                                'dimension': 16,
                                'index_file_size': 1024 (default)，
                                'metric_type': Metric_type.L2 (default)
                                }`

        :param timeout: Timeout in seconds.
        :type  timeout: double

        :return: Whether the operation is successful.
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

        check_pass_param(collection_name=collection_name, dimension=dim, index_file_size=index_file_size,
                         metric_type=metric_type)

        with self._connection() as handler:
            return handler.create_collection(collection_name, dim, index_file_size, metric_type, collection_param, timeout)

    @check_connect
    def create_hybrid_collection(self, collection_name, fields, timeout=30):
        with self._connection() as handler:
            return handler.create_hybrid_collection(collection_name, fields, timeout)

    @check_connect
    def has_collection(self, collection_name, timeout=30):
        """

        Checks whether a collection exists.

        :param collection_name: Name of the collection to check.
        :type  collection_name: str
        :param timeout: Timeout in seconds.
        :type  timeout: int

        :return:
            Status: indicate whether the operation is successful.
            bool if given collection_name exists

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

        :returns: (Status, table_schema)
            Status: indicate if query is successful
            table_schema: return when operation is successful

        :rtype: (Status, TableSchema)
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

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.count_collection(collection_name, timeout)

    @check_connect
    def list_collections(self, timeout=30):
        """
        Returns collection list.

        :return:
            Status: indicate if this operation is successful

            collections: list of collection names, return when operation
                    is successful
        :rtype:
            (Status, list[str])
        """
        with self._connection() as handler:
            return handler.show_collections(timeout)

    @check_connect
    def get_collection_stats(self, collection_name, timeout=30):
        """
        Returns collection statistics information

        :return:
            Status: indicate if this operation is successful

            statistics: statistics information
        :rtype:
            (Status, dict)
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.show_collection_info(collection_name, timeout)

    @check_connect
    def load_collection(self, collection_name, partition_tags=None, timeout=None):
        """
        Loads a collection for caching.

        :type collection_name: str
        :param collection_name: collection to load

        :returns:
            Status:  indicate if invoke is successful
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.preload_collection(collection_name, partition_tags, timeout)

    @check_connect
    def reload_segments(self, collection_name, segment_ids):
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.reload_segments(collection_name, segment_ids)

    @check_connect
    def drop_collection(self, collection_name, timeout=30):
        """
        Deletes a collection by name.

        :type  collection_name: str
        :param collection_name: Name of the collection being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.drop_collection(collection_name, timeout)

    @check_connect
    def insert(self, collection_name, records, ids=None, partition_tag=None, params=None, timeout=None, **kwargs):
        """
        Insert vectors to a collection.

        :param ids: list of id
        :type  ids: list[int]

        :type  collection_name: str
        :param collection_name: Name of the collection to insert vectors to.

        :type  records: list[list[float]]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param records: List of vectors to insert.

        :type partition_tag: str or None.
            If partition_tag is None, vectors will be inserted to the collection rather than partitions.

        :param partition_tag: Tag of a partition.

        :returns:
            Status: Whether vectors are inserted successfully.
            ids: IDs of the inserted vectors.
        :rtype: (Status, list(int))
        """
        if kwargs.get("insert_param", None) is not None:
            with self._connection() as handler:
                return handler.insert(None, None, timeout=timeout, **kwargs)

        check_pass_param(collection_name=collection_name, records=records)
        partition_tag is not None and check_pass_param(partition_tag=partition_tag)
        if ids is not None:
            check_pass_param(ids=ids)
            if len(records) != len(ids):
                raise ParamError("length of vectors do not match that of ids")

        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.insert(collection_name, records, ids, partition_tag, params, timeout, **kwargs)

    @check_connect
    def insert_hybrid(self, collection_name, entities, vector_entities, ids=None, partition_tag=None, params=None):
        with self._connection() as handler:
            return handler.insert_hybrid(collection_name, entities, vector_entities, ids, partition_tag, params)

    def get_entity_by_id(self, collection_name, ids, timeout=None):
        """
        Returns raw vectors according to ids.

        :param collection_name: Name of the collection
        :type collection_name: str

        :param ids: list of vector id
        :type ids: list

        :returns:
            Status: indicate if invoke is successful

        """
        check_pass_param(collection_name=collection_name, ids=ids)

        with self._connection() as handler:
            return handler.get_vectors_by_ids(collection_name, ids, timeout=timeout)

    def get_hybrid_entity_by_id(self, collection_name, ids):
        check_pass_param(collection_name=collection_name, ids=ids)
        with self._connection() as handler:
            return handler.get_hybrid_entity_by_id(collection_name, ids)

    @check_connect
    def list_id_in_segment(self, collection_name, segment_name, timeout=None):
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
        :param index: index params
        :type index: dict

            index_param can be None

            `example (default) param={'index_type': IndexType.FLAT,
                            'nlist': 16384}`

        :param timeout: grpc request timeout.

            if `timeout` = -1, method invoke a synchronous call, waiting util grpc response
            else method invoke a asynchronous call, timeout work here

        :type  timeout: int

        :return: Whether the operation is successful.
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

        :returns:
            Status:  Whether the operation is successful.
            IndexSchema:

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

        :return:
            Status: Whether the operation is successful.

        ：:rtype: Status
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

        :param partition_name: Name of the partition.
        :type  partition_name: str

        :param partition_tag: Name of the partition tag.
        :type  partition_tag: str

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            Status: Whether the operation is successful.

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.create_partition(collection_name, partition_tag, timeout)

    @check_connect
    def has_partition(self, collection_name, partition_tag):
        """
        Check if specified partition exists.

        :param collection_name: target table name.
        :type  collection_name: str

        :param partition_tag: partition tag.
        :type  partition_tag: str

        :return:
            Status: Whether the operation is successful.
            exists: If specified partition exists

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.has_partition(collection_name, partition_tag)

    @check_connect
    def list_partitions(self, collection_name, timeout=30):
        """
        Show all partitions in a collection.

        :param collection_name: target table name.
        :type  collection_name: str

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            Status: Whether the operation is successful.
            partition_list:

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

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            Status: Whether the operation is successful.

        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.drop_partition(collection_name, partition_tag, timeout)

    @check_connect
    def search(self, collection_name, top_k, query_records, partition_tags=None, params=None, timeout=None, **kwargs):
        """
        Search vectors in a collection.

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param top_k: number of vertors which is most similar with query vectors
        :type  top_k: int

        :param nprobe: cell number of probe
        :type  nprobe: int

        :param query_records: vectors to query
        :type  query_records: list[list[float32]]

        :param partition_tags: tags to search
        :type  partition_tags: list

        :return
            Status: Whether the operation is successful.
            result: query result

        :rtype: (Status, TopKQueryResult)

        """
        check_pass_param(collection_name=collection_name, topk=top_k, records=query_records)
        if partition_tags is not None:
            check_pass_param(partition_tag_array=partition_tags)

        params = dict() if params is None else params
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.search(collection_name, top_k, query_records, partition_tags, params, timeout, **kwargs)

    @check_connect
    def search_hybrid_pb(self, collection_name, query_entities, partition_tags=None, params=None, **kwargs):
        with self._connection() as handler:
            return handler.search_hybrid_pb(collection_name, query_entities, partition_tags, params, **kwargs)

    @check_connect
    def search_hybrid(self, collection_name, vector_params, dsl, partition_tags=None, params=None, **kwargs):
        with self._connection() as handler:
            return handler.search_hybrid(collection_name, vector_params, dsl, partition_tags, params, **kwargs)

    @check_connect
    def search_in_segment(self, collection_name, file_ids, query_records, top_k, params=None, timeout=None, **kwargs):
        """
        Searches for vectors in specific segments of a collection.

        The Milvus server stores vector data into multiple files. Searching for vectors in specific files is a
        method used in Mishards. Obtain more detail about Mishards, see
        <a href="https://github.com/milvus-io/milvus/tree/master/shards">

        :type  collection_name: str
        :param collection_name: table name been queried

        :type  file_ids: list[str] or list[int]
        :param file_ids: Specified files id array

        :type  query_records: list[list[float]]
        :param query_records: all vectors going to be queried

        :param query_ranges: Optional ranges for conditional search.

            If not specified, search in the whole table

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            results: query result

        :rtype: (Status, TopKQueryResult)
        """
        check_pass_param(collection_name=collection_name, topk=top_k, records=query_records, ids=file_ids)

        params = dict() if params is None else params
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.search_in_files(collection_name, file_ids,
                                           query_records, top_k, params, timeout, **kwargs)

    @check_connect
    def delete_entity_by_id(self, collection_name, id_array, timeout=None):
        """
        Deletes vectors in a collection by vector ID.

        :param collection_name: Name of the collection.
        :type  collection_name: str

        :param id_array: list of vector id
        :type  id_array: list[int]

        :return:
            Status: Whether the operation is successful.
        """
        check_pass_param(collection_name=collection_name, ids=id_array)
        with self._connection() as handler:
            return handler.delete_by_id(collection_name, id_array, timeout)

    @check_connect
    def flush(self, collection_name_array=None, timeout=None, **kwargs):
        """
        Flushes vector data in one collection or multiple collections to disk.

        :type  collection_name_array: list
        :param collection_name: Name of one or multiple collections to flush.

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

