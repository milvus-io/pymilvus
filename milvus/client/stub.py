# -*- coding: UTF-8 -*-

import copy
import functools
import threading

from urllib.parse import urlparse

from . import __version__
from .types import IndexType, MetricType
from .check import check_pass_param, is_legal_host, is_legal_port
from .pool import ConnectionPool
from .grpc_handler import GrpcHandler
from .http_handler import HttpHandler
from .exceptions import ParamError, NotConnectError

from ..settings import DefaultConfig as config


class Milvus:

    def __init__(self, host=None, port=None, handler="GRPC", **kwargs):
        self._handler = handler
        self._host = host
        self._port = port
        self._uri = kwargs.get('uri', None)

        # create connection pool
        _url = self._set_uri(host, port, self._uri, handler)
        self._pool = ConnectionPool(_url)

    def __enter__(self):
        self.__conn = self._get_connection()
        return self.__conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__conn.close()
        self.__conn = None

    def _set_uri(self, host, port, uri, handler="GRPC"):
        default_port = config.GRPC_PORT if handler == "GRPC" else config.HTTP_PORT
        default_uri = config.GRPC_URI if handler == "GRPC" else config.HTTP_URI
        uri_prefix = "tcp://" if handler == "GRPC" else "http://"

        if host is not None:
            _port = port if port is not None else default_port
            _host = host
        elif port is None:
            try:
                # Ignore uri check here
                # if not is_legal_uri(_uri):
                #     raise ParamError("uri {} is illegal".format(_uri))
                #
                # If uri is empty (None or '') use default uri instead
                # (the behavior may change in the future)
                # _uri = urlparse(_uri) if _uri is not None else urlparse(config.GRPC_URI)
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
            raise ParamError("host or port is illegal")

        return "{}{}:{}".format(uri_prefix, str(_host), str(_port))

    def _get_connection(self):
        return self._pool.fetch()

    def _put_connection(self, conn):
        conn.release()
        # self._pool.release(conn)

    def set_hook(self, **kwargs):
        # TODO: may remove it. 
        return self._handler.set_hook(**kwargs)

    @property
    def status(self):
        return self._handler.status

    @property
    def handler(self):
        return self._handler

    def client_version(self):
        """
        Returns the version of the client.

        :return: Version of the client.

        :rtype: (str)
        """
        return __version__

    def server_status(self, timeout=10):
        """
        Returns the status of the Milvus server.

        :return:
            Status: Whether the operation is successful.

            str : Status of the Milvus server.

        :rtype: (Status, str)
        """
        return self._cmd("status", timeout)

    def server_version(self, timeout=10):
        """
       Returns the version of the Milvus server.

       :return:
           Status: Whether the operation is successful.

           str : Version of the Milvus server.

       :rtype: (Status, str)
       """

        return self._cmd("version", timeout)

    def _cmd(self, cmd, timeout=10):
        check_pass_param(cmd=cmd)

        conn = self._get_connection()
        try:
            return conn._cmd(cmd, timeout)
        except:
            raise
        finally:
            conn.close()

    def create_collection(self, param, timeout=10):
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

        conn = self._get_connection()
        try:
            return conn.create_collection(collection_name, dim, index_file_size, metric_type, collection_param)
        except:
            raise
        finally:
            conn.close()

    def has_collection(self, collection_name, timeout=10):
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
        conn = self._get_connection()
        try:
            print("Connection number: ------>", conn.conn_id())
            return conn.has_collection(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def describe_collection(self, collection_name, timeout=10):
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

        conn = self._get_connection()
        try:
            return conn.describe_collection(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def count_collection(self, collection_name, timeout=10):
        """
        Returns the number of vectors in a collection.

        :type  collection_name: str
        :param collection_name: target table name.

        :returns:
            Status: indicate if operation is successful

            res: int, table row count
        """
        check_pass_param(collection_name=collection_name)

        conn = self._get_connection()
        try:
            return conn.count_collection(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def show_collections(self, timeout=10):
        """
        Returns information of all collections.

        :return:
            Status: indicate if this operation is successful

            collections: list of table names, return when operation
                    is successful
        :rtype:
            (Status, list[str])
        """
        conn = self._get_connection()
        try:
            return conn.show_collections(timeout)
        except:
            raise
        finally:
            conn.close()

    def collection_info(self, collection_name, timeout=10):
        check_pass_param(collection_name=collection_name)
        conn = self._get_connection()
        try:
            return conn.show_collection_info(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def preload_collection(self, collection_name, timeout=None):
        """
        Loads a collection for caching.

        :type collection_name: str
        :param collection_name: table to preload

        :returns:
            Status:  indicate if invoke is successful
        """
        check_pass_param(collection_name=collection_name)

        conn = self._get_connection()
        try:
            return conn.preload_collection(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def drop_collection(self, collection_name, timeout=10):
        """
        Deletes a collection by name.

        :type  collection_name: str
        :param collection_name: Name of the table being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """
        check_pass_param(collection_name=collection_name)

        conn = self._get_connection()
        try:
            return conn.drop_collection(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def insert(self, collection_name, records, ids=None, partition_tag=None, params=None, **kwargs):
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

        :type  timeout: int
        :param timeout: Time to wait for server response before timeout.

        :returns:
            Status: Whether vectors are inserted successfully.
            ids: IDs of the inserted vectors.
        :rtype: (Status, list(int))
        """
        conn = self._get_connection()
        try:
            if kwargs.get("insert_param", None) is not None:
                return conn.insert(None, None, **kwargs)

            check_pass_param(collection_name=collection_name, records=records)
            partition_tag is not None and check_pass_param(partition_tag=partition_tag)
            ids is not None and check_pass_param(ids=ids)

            if ids is not None and len(records) != len(ids):
                raise ParamError("length of vectors do not match that of ids")

            params = params or dict()
            if not isinstance(params, dict):
                raise ParamError("Params must be a dictionary type")

            return conn.insert(collection_name, records, ids, partition_tag, params, None, **kwargs)
        except:
            raise
        finally:
            conn.close()

    def get_vector_by_id(self, collection_name, vector_id, timeout=None):
        check_pass_param(collection_name=collection_name, ids=[vector_id])

        conn = self._get_connection()
        try:
            return conn.get_vector_by_id(collection_name, vector_id, timeout=timeout)
        except:
            raise
        finally:
            conn.close()

    def get_vector_ids(self, collection_name, segment_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        check_pass_param(collection_name=segment_name)

        conn = self._get_connection()
        try:
            return conn.get_vector_ids(collection_name, segment_name, timeout)
        except:
            raise
        finally:
            conn.close()

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

        conn = self._get_connection()
        try:
            return conn.create_index(collection_name, _index_type, params, timeout, **kwargs)
        except:
            raise
        finally:
            conn.close()

    def describe_index(self, collection_name, timeout=10):
        """
        Show index information of a collection.

        :type collection_name: str
        :param collection_name: table name been queried

        :returns:
            Status:  Whether the operation is successful.
            IndexSchema:

        """
        check_pass_param(collection_name=collection_name)

        conn = self._get_connection()
        try:
            return conn.describe_index(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def drop_index(self, collection_name, timeout=10):
        """
        Removes an index.

        :param collection_name: target collection name.
        :type collection_name: str

        :return:
            Status: Whether the operation is successful.

        ：:rtype: Status
        """
        check_pass_param(collection_name=collection_name)

        conn = self._get_connection()
        try:
            return conn.drop_index(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def create_partition(self, collection_name, partition_tag, timeout=10):
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

        conn = self._get_connection()
        try:
            return conn.create_partition(collection_name, partition_tag, timeout)
        except:
            raise
        finally:
            conn.close()

    def show_partitions(self, collection_name, timeout=10):
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

        conn = self._get_connection()
        try:
            return conn.show_partitions(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

    def drop_partition(self, collection_name, partition_tag, timeout=10):
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

        conn = self._get_connection()
        try:
            return conn.drop_partition(collection_name, partition_tag, timeout)
        except:
            raise
        finally:
            conn.close()

    def search(self, collection_name, top_k, query_records, partition_tags=None, params=None, **kwargs):
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
        check_pass_param(collection_name=collection_name, topk=top_k,
                         records=query_records, partition_tag_array=partition_tags)

        params = dict() if params is None else params
        if not isinstance(params, dict):
            raise ParamError("Params must be a dictionary type")

        conn = self._get_connection()
        try:
            return conn.search(collection_name, top_k, query_records, partition_tags, params, **kwargs)
        except:
            raise
        finally:
            conn.close()

    def search_in_files(self, collection_name, file_ids, query_records, top_k, params=None, **kwargs):
        """
        Searches for vectors in specific files of a collection.

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

        conn = self._get_connection()
        try:
            return conn.search_in_files(collection_name, file_ids,
                                        query_records, top_k, params, **kwargs)
        except:
            raise
        finally:
            conn.close()

    def delete_by_id(self, collection_name, id_array, timeout=None):
        """
        Deletes vectors in a collection by vector ID.

        """
        check_pass_param(collection_name=collection_name, ids=id_array)

        conn = self._get_connection()
        try:
            return conn.delete_by_id(collection_name, id_array, timeout)
        except:
            raise
        finally:
            conn.close()

    def flush(self, collection_name_array=None):
        """
        Flushes vector data in one collection or multiple collections to disk.

        :type  collection_name_array: list
        :param collection_name: Name of one or multiple collections to flush.

        """
        conn = self._get_connection()

        if collection_name_array in (None, []):
            return conn.flush([])

        if not isinstance(collection_name_array, list):
            raise ParamError("Collection name array must be type of list")

        if len(collection_name_array) <= 0:
            raise ParamError("Collection name array is not allowed to be empty")

        for name in collection_name_array:
            check_pass_param(collection_name=name)

        try:
            return conn.flush(collection_name_array)
        except:
            raise
        finally:
            conn.close()

    def compact(self, collection_name, timeout=None):
        """
        Compacts segments in a collection. This function is recommended after deleting vectors.

        :type  collection_name: str
        :param collection_name: Name of the collections to compact.

        """
        check_pass_param(collection_name=collection_name)

        conn = self._get_connection()
        try:
            return conn.compact(collection_name, timeout)
        except:
            raise
        finally:
            conn.close()

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

    # In old version of pymilvus, some methods are different from the new.
    # apply alternative method name for compatibility

    # get_collection_row_count = count_collection
    # delete_collection = drop_collection
    add_vectors = insert
    search_vectors = search
    search_vectors_in_files = search_in_files
