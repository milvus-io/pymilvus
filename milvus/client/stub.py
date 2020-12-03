# -*- coding: UTF-8 -*-

import collections
import functools
import logging
import threading

from collections import defaultdict

from urllib.parse import urlparse

from . import __version__
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
        if k in ("client_tag", "pool_size", "wait_timeout", "handler",
                 "try_connect", "pre_ping", "max_retry"):
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
    def __init__(self, host=None, port=None, **kwargs):
        self._name = kwargs.get('name', None)
        self._uri = None
        self._status = None
        self._connected = False
        self._handler = kwargs.get("handler", "GRPC")

        _uri = kwargs.get('uri', None)
        pool_uri = _set_uri(host, port, _uri, self._handler)
        pool_kwargs = _pool_args(**kwargs)
        # self._pool = SingleConnectionPool(pool_uri, **pool_kwargs)
        pool = kwargs.get("pool", "SingletonThread")
        if pool == "QueuePool":
            self._pool = ConnectionPool(pool_uri, **pool_kwargs)
        elif pool == "SingletonThread":
            self._pool = SingletonThreadPool(pool_uri, **pool_kwargs)
        elif pool == "Singleton":
            self._pool = SingleConnectionPool(pool_uri, **pool_kwargs)
        else:
            raise ParamError("Unknown pool value: {}".format(pool))

        #
        self._conn = None

        # store extra key-words arguments
        self._kw = kwargs
        self._hooks = collections.defaultdict()

        # cache collection_info
        self._c_cache = defaultdict(dict)
        self._cache_cv = threading.Condition()

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
            return handler.cmd(cmd, timeout)

    @check_connect
    def create_collection(self, collection_name, fields, timeout=30):
        '''
        Creates a collection.

        :param collection_name: The name of the collection. A collection name can only include
        numbers, letters, and underscores, and must not begin with a number.
        :type  str
        :param fields: Field parameters.
        :type  fields: dict

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        '''
        with self._connection() as handler:
            return handler.create_collection(collection_name, fields, timeout)

    @check_connect
    def has_collection(self, collection_name, timeout=30):
        """
        Checks whether a specified collection exists.

        :param collection_name: The name of the collection to check.
        :type  collection_name: str

        :return: If specified collection exists
        :rtype: bool

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.has_collection(collection_name, timeout)

    @check_connect
    def get_collection_info(self, collection_name, timeout=30):
        """
        Returns information of a specified collection, including field
        information of the collection and index information of fields.

        :param collection_name: The name of the collection to describe.
        :type  collection_name: str

        :return: The information of collection to describe.
        :rtype: dict

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.get_collection_info(collection_name, timeout)

    @check_connect
    def count_entities(self, collection_name, timeout=30):
        """
        Returns the number of entities in a specified collection.

        :param collection_name: The name of the collection to count entities of.
        :type  collection_name: str

        :return: The number of entities
        :rtype: int

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.count_entities(collection_name, timeout)

    @check_connect
    def list_collections(self, timeout=30):
        """
        Returns a list of all collection names.

        :return: List of collection names, return when operation is successful
        :rtype: list[str]

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.list_collections(timeout)

    @check_connect
    def get_collection_stats(self, collection_name, timeout=30):
        """
        Returns statistical information about a specified collection, including
        the number of entities and the storage size of each segment of the collection.

        :param collection_name: The name of the collection to get statistics about.
        :type  collection_name: str

        :return: The collection stats.
        :rtype: dict

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.get_collection_stats(collection_name, timeout)

    @check_connect
    def load_collection(self, collection_name, timeout=None):
        """
         Loads a specified collection from disk to memory.

        :param collection_name: The name of the collection to load.
        :type  collection_name: str

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.load_collection(collection_name, timeout)

    @check_connect
    def drop_collection(self, collection_name, timeout=30):
        """
        Deletes a specified collection.

        :param collection_name: The name of the collection to delete.
        :type  collection_name: str

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.drop_collection(collection_name, timeout)

    @check_connect
    def bulk_insert(self, collection_name, bulk_entities, ids=None,
                    partition_tag=None, params=None, timeout=None, **kwargs):
        """
        Inserts columnar entities in a specified collection.

        :param collection_name: The name of the collection to insert entities in.
        :type  collection_name: str.
        :param bulk_entities: The columnar entities to insert.
        :type  bulk_entities: list
        :param ids: The list of ids corresponding to the inserted entities.
        :type  ids: list[int]
        :param partition_tag: The name of the partition to insert entities in. The default value is
         None. The server stores entities in the “_default” partition by default.
        :type  partition_tag: str

        :return: list of ids of the inserted vectors.
        :rtype: list[int]

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """

        if kwargs.get("insert_param", None) is not None:
            with self._connection() as handler:
                return handler.bulk_insert(None, None, timeout=timeout, **kwargs)

        specified_num = sum([1 for field in bulk_entities if "type" in field])
        if 0 < specified_num < len(bulk_entities):
            raise ParamError("Some fields did not specify field type")

        fields = dict()
        if specified_num == 0:
            fields = self._c_cache[collection_name]
            if not fields:
                info = self.get_collection_info(collection_name)
                for field in info["fields"]:
                    fields[field["name"]] = field["type"]
        else:
            # map(lambda x: fields[x["name"]] = x["type"], bulk_entities)
            for bulk in bulk_entities:
                fields[bulk["name"]] = bulk["type"]

        if ids is not None:
            check_pass_param(ids=ids)
        with self._connection() as handler:
            results = handler.bulk_insert(collection_name, bulk_entities, fields,
                                          ids, partition_tag, params, timeout, **kwargs)
            with self._cache_cv:
                self._c_cache[collection_name] = fields
            return results

    def insert(self, collection_name, entities,
               partition_tag=None, params=None, timeout=None, **kwargs):
        """
        Inserts linear entities in a specified collection.

        :param collection_name: The name of the collection to insert entities in.
        :type  collection_name: str.
        :param entities: The linear entities to insert.
        :type  entities: list
        :param partition_tag: The name of the partition to insert entities in. The default
                              value is None. The server stores entities in the “_default”
                              partition by default.
        :type  partition_tag: str

        :return: list of ids of the inserted vectors.
        :rtype: list[int]

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """

        fields = self._c_cache[collection_name]
        if not fields:
            info = self.get_collection_info(collection_name)
            for field in info["fields"]:
                fields[field["name"]] = field["type"]

        if kwargs.get("insert_param", None) is not None:
            with self._connection() as handler:
                return handler.insert(None, None, timeout=timeout, **kwargs)

        with self._connection() as handler:
            results = handler.insert(collection_name, entities, fields,
                                     partition_tag, params, timeout, **kwargs)
            with self._cache_cv:
                self._c_cache[collection_name] = fields
            return results

    def get_entity_by_id(self, collection_name, ids, fields=None, timeout=None):
        """
        Returns the entities specified by given IDs.

        :param collection_name: The name of the collection to retrieve entities from.
        :type  collection_name: str

        :param ids: A list of IDs of the entities to retrieve.
        :type ids: list[int]

        :return: The entities specified by given IDs.
        :rtype: Entities

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, ids=ids)

        with self._connection() as handler:
            return handler.get_entity_by_id(collection_name, ids, fields, timeout=timeout)

    @check_connect
    def list_id_in_segment(self, collection_name, segment_id, timeout=None):
        """
        Returns all entity IDs in a specified segment.

        :param collection_name: The name of the collection that contains the specified segment
        :type  collection_name: str
        :param segment_id: The ID of the segment. You can get segment IDs by calling the
                           get_collection_stats() method.
        :type  segment_id: int

        :return: List of IDs in a specified segment.
        :rtype: list[int]

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        check_pass_param(ids=[segment_id])
        with self._connection() as handler:
            return handler.list_id_in_segment(collection_name, segment_id, timeout)

    @check_connect
    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        """
        Creates an index for a field in a specified collection. Milvus does not support
        creating multiple indexes for a field. In a scenario where the field already has
        an index, if you create another one that is equivalent (in terms of type and
        parameters) to the existing one, the server returns this index to the client;
        otherwise, the server replaces the existing index with the new one.

        :param collection_name: The name of the collection to create field indexes.
        :type  collection_name: str
        :param field_name: The name of the field to create an index for.
        :type  field_name: str
        :param params: Indexing parameters.
        :type  params: dict

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        params = params or dict()
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.create_index(collection_name, field_name, params, timeout, **kwargs)

    @check_connect
    def drop_index(self, collection_name, field_name, timeout=30):
        """
        Removes the index of a field in a specified collection.

        :param collection_name: The name of the collection to remove the field index from.
        :type  collection_name: str
        :param field_name: The name of the field to remove the index of.
        :type  field_name: str

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)

        with self._connection() as handler:
            return handler.drop_index(collection_name, field_name, timeout)

    @check_connect
    def create_partition(self, collection_name, partition_tag, timeout=30):
        """
        Creates a partition in a specified collection. You only need to import the
        parameters of partition_tag to create a partition. A collection cannot hold
        partitions of the same tag, whilst you can insert the same tag in different collections.

        :param collection_name: The name of the collection to create partitions in.
        :type  collection_name: str

        :param partition_tag: Name of the partition.
        :type  partition_tag: str

        :param partition_tag: The tag name of the partition.
        :type  partition_tag: str

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.create_partition(collection_name, partition_tag, timeout)

    @check_connect
    def has_partition(self, collection_name, partition_tag, timeout=30):
        """
        Checks if a specified partition exists in a collection.

        :param collection_name: The name of the collection to find the partition in.
        :type  collection_name: str

        :param partition_tag: The tag name of the partition to check
        :type  partition_tag: str

        :return: Whether a specified partition exists in a collection.
        :rtype: bool

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.has_partition(collection_name, partition_tag, timeout)

    @check_connect
    def list_partitions(self, collection_name, timeout=30):
        """
        Returns a list of all partition tags in a specified collection.

        :param collection_name: The name of the collection to retrieve partition tags from.
        :type  collection_name: str

        :return: A list of all partition tags in specified collection.
        :rtype: list[str]

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)

        with self._connection() as handler:
            return handler.list_partitions(collection_name, timeout)

    @check_connect
    def drop_partition(self, collection_name, partition_tag, timeout=30):
        """
        Deletes the specified partitions in a collection.

        :param collection_name: The name of the collection to delete partitions from.
        :type  collection_name: str

        :param partition_tag: The tag name of the partition to delete.
        :type  partition_tag: str

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, partition_tag=partition_tag)
        with self._connection() as handler:
            return handler.drop_partition(collection_name, partition_tag, timeout)

    @check_connect
    def search(self, collection_name, dsl, partition_tags=None,
               fields=None, timeout=None, **kwargs):
        """
        Searches a collection based on the given DSL clauses and returns query results.

        :param collection_name: The name of the collection to search.
        :type  collection_name: str
        :param dsl: The DSL that defines the query.
        :type  dsl: dict
        :param partition_tags: The tags of partitions to search.
        :type  partition_tags: list[str]
        :param fields: The fields to return in the search result
        :type  fields: list[str]

        :return: Query result.
        :rtype: QueryResult

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        with self._connection() as handler:
            return handler.search(collection_name, dsl, partition_tags,
                                  fields, timeout=timeout, **kwargs)

    @check_connect
    def search_in_segment(self, collection_name, segment_ids, dsl,
                          fields=None, timeout=None, **kwargs):
        """
        Searches in the specified segments of a collection.

        The Milvus server stores entity data into multiple files. Searching for entities
        in specific files is a method used in Mishards. Obtain more detail about Mishards,
        see <a href="https://github.com/milvus-io/milvus/tree/master/shards">

        :param collection_name: The name of the collection to search.
        :type  collection_name: str
        :param segment_ids: The list of segment id to search.
        :type  segment_ids: list[int]
        :param dsl: The DSL that defines the query.
        :type  dsl: dict
        :param fields: The fields to return in the search result
        :type  fields: list[str]:type  query_records: list[list[float]]

        :return: Query result.
        :rtype: QueryResult

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """

        # params = dict() if params is None else params
        # if not isinstance(params, dict):
        #     raise ParamError("Params must be a dictionary type")
        with self._connection() as handler:
            return handler.search_in_segment(collection_name, segment_ids, dsl,
                                             fields, timeout, **kwargs)

    @check_connect
    def delete_entity_by_id(self, collection_name, ids, timeout=None):
        """
        Deletes the entities specified by a given list of IDs.

        :param collection_name:  The name of the collection to remove entities from.
        :type  collection_name: str
        :param ids: A list of IDs of the entities to delete.
        :type  ids: list[int]

        :return: Status of delete request. The delete request will still execute successfully
                 if Some of ids may not exist in specified collection, in this case the returned
                 status will differ. Note that in current version his is an EXPERIMENTAL function.
        :rtype:  Status.

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name, ids=ids)
        with self._connection() as handler:
            return handler.delete_entity_by_id(collection_name, ids, timeout)

    @check_connect
    def flush(self, collection_name_array=None, timeout=None, **kwargs):
        """
        Flushes data in the specified collections from memory to disk. When you insert or
        delete data, the server stores the data in the memory temporarily and then flushes
        it to the disk at fixed intervals. Calling flush ensures that the newly inserted
        data is visible and the deleted data is no longer recoverable.

        :type  collection_name_array: An array of names of the collections to flush.
        :param collection_name_array: list[str]

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """

        if collection_name_array in (None, []):
            with self._connection() as handler:
                return handler.flush([], timeout, **kwargs)

        if not isinstance(collection_name_array, list):
            raise ParamError("Collection name array must be type of list")

        if len(collection_name_array) <= 0:
            raise ParamError("Collection name array is not allowed to be empty")

        for name in collection_name_array:
            check_pass_param(collection_name=name)
        with self._connection() as handler:
            return handler.flush(collection_name_array, timeout, **kwargs)

    @check_connect
    def compact(self, collection_name, threshold=0.2, timeout=None, **kwargs):
        """
        Compacts a specified collection. After deleting some data in a segment, you can call
        compact to free up the disk space occupied by the deleted data. Calling compact also
        deletes empty segments, but does not merge segments.
        
        :param collection_name: The name of the collection to compact.
        :type  collection_name: str
        :param threshold: The threshold for compact. When the percentage of deleted entities
                          in a segment is below the threshold, the server skips this segment
                          when compacting the collection. The default value is 0.2, range is
                          [0, 1].

        :return: Status of compact request. The compact request will still execute successfully
                 if server skip some of collections, in this case the returned status will differ.
                 Note that in current version his is an EXPERIMENTAL function.
        :rtype:  Status.

        :raises:
            RpcError: If grpc encounter an error
            ParamError: If parameters are invalid
            BaseError: If the return result from server is not ok
        """
        check_pass_param(collection_name=collection_name)
        with self._connection() as handler:
            return handler.compact(collection_name, threshold, timeout, **kwargs)

    def get_config(self, key):
        """
        Gets Milvus configurations.

        """
        cmd = "GET {}".format(key)

        return self._cmd(cmd)

    def set_config(self, key, value):
        """
        Sets Milvus configurations.

        """
        cmd = "SET {} {}".format(key, value)

        return self._cmd(cmd)
