import datetime
from urllib.parse import urlparse
import logging
import threading
import ujson

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import milvus_pb2 as grpc_types
from .abstract import ConnectIntf, CollectionSchema, IndexParam, PartitionParam, Entities, QueryResult
from .prepare import Prepare
from .types import MetricType, Status
from .check import (
    int_or_str,
    is_legal_host,
    is_legal_port,
)

from .abs_client import AbsMilvus
from .asynch import SearchFuture, InsertFuture, CreateIndexFuture, CompactFuture, FlushFuture

from .hooks import BaseSearchHook
from .client_hooks import SearchHook, HybridSearchHook
from .exceptions import *
from ..settings import DefaultConfig as config
from . import __version__

LOGGER = logging.getLogger(__name__)


def error_handler(*rargs):
    def wrapper(func):
        def handler(self, *args, **kwargs):
            record_dict = {}
            try:
                record_dict["API start"] = str(datetime.datetime.now())
                if self._pre_ping:
                    self.ping()
                record_dict["RPC start"] = str(datetime.datetime.now())
                return func(self, *args, **kwargs)
            except BaseException as e:
                LOGGER.error("Error: {}".format(e))
                if e.code == Status.ILLEGAL_COLLECTION_NAME:
                    raise IllegalCollectionNameException(e.code, e.message)
                if e.code == Status.COLLECTION_NOT_EXISTS:
                    raise CollectionNotExistException(e.code, e.message)

                raise e

            except grpc.FutureTimeoutError as e:
                record_dict["RPC timeout"] = str(datetime.datetime.now())
                LOGGER.error("\nAddr [{}] {}\nRequest timeout: {}\n\t{}".format(self.server_address, func.__name__, e, record_dict))
                raise e
            except grpc.RpcError as e:
                record_dict["RPC error"] = str(datetime.datetime.now())
                LOGGER.error("RPC error: {}\n\t{}".format(e, record_dict))
                raise e
            except Exception as e:
                record_dict["Exception"] = str(datetime.datetime.now())
                LOGGER.error("\nAddr [{}] {}\nExcepted error: {}\n\t{}".format(self.server_address, func.__name__, e, record_dict))
                raise e

        return handler

    return wrapper


def set_uri(host, port, uri):
    if host is not None:
        _port = port if port is not None else config.GRPC_PORT
        _host = host
    elif port is None:
        try:
            _uri = urlparse(uri) if uri else urlparse(config.GRPC_URI)
            _host = _uri.hostname
            _port = _uri.port
        except (AttributeError, ValueError, TypeError) as e:
            raise ParamError("uri is illegal: {}".format(e))
    else:
        raise ParamError("Param is not complete. Please invoke as follow:\n"
                         "\t(host = ${HOST}, port = ${PORT})\n"
                         "\t(uri = ${URI})\n")

    if not is_legal_host(_host) or not is_legal_port(_port):
        raise ParamError("host or port is illeagl")

    return "{}:{}".format(str(_host), str(_port))


class GrpcHandler(AbsMilvus):
    def __init__(self, host=None, port=None, pre_ping=True, **kwargs):
        self._channel = None
        self._stub = None
        self._uri = None
        self.status = None
        self._connected = False
        self._pre_ping = pre_ping
        # if self._pre_ping:
        self._max_retry = kwargs.get("max_retry", 3)

        # record
        self._id = kwargs.get("conn_id", 0)

        # condition
        self._condition = threading.Condition()
        self._request_id = 0

        # client hook
        self._search_hook = SearchHook()
        self._hybrid_search_hook = HybridSearchHook()
        self._search_file_hook = SearchHook()

        # set server uri if object is initialized with parameter
        _uri = kwargs.get("uri", None)
        self._setup(host, port, _uri, pre_ping)

    def __str__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items() if not key.startswith('_')]
        return '<Milvus: {}>'.format(', '.join(attr_list))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _setup(self, host, port, uri, pre_ping=False):
        """
        Create a grpc channel and a stub

        :raises: NotConnectError

        """
        self._uri = set_uri(host, port, uri)
        self._channel = grpc.insecure_channel(
            self._uri,
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                     ('grpc.enable_retries', 1),
                     ('grpc.keepalive_time_ms', 55000)]
        )
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)
        self.status = Status()

    def _pre_request(self):
        if self._pre_ping:
            self.ping()

    def _get_request_id(self):
        with self._condition:
            _id = self._request_id
            self._request_id += 1
            return _id

    def set_hook(self, **kwargs):
        """
        specify client hooks.
        The client hooks are used in methods which interact with server.
        Use key-value method to set hooks. Supported hook setting currently is as follow.

            search hook,
            search-in-file hook

        """

        # config search hook
        _search_hook = kwargs.get('search', None)
        if _search_hook:
            if not isinstance(_search_hook, BaseSearchHook):
                raise ParamError("search hook must be a subclass of `BaseSearchHook`")

            self._search_hook = _search_hook

        _search_file_hook = kwargs.get('search_in_file', None)
        if _search_file_hook:
            if not isinstance(_search_file_hook, BaseSearchHook):
                raise ParamError("search hook must be a subclass of `BaseSearchHook`")

            self._search_file_hook = _search_file_hook

    def ping(self, timeout=30):
        ft = grpc.channel_ready_future(self._channel)
        retry = self._max_retry
        try:
            while retry > 0:
                try:
                    ft.result(timeout=timeout)
                    return True
                except:
                    retry -= 1
                    LOGGER.debug("Retry connect addr <{}> {} times".format(self._uri, self._max_retry - retry))
                    if retry > 0:
                        continue
                    else:
                        LOGGER.error("Retry to connect server {} failed.".format(self._uri))
                        raise
        except grpc.FutureTimeoutError:
            raise NotConnectError('Fail connecting to server on {}. Timeout'.format(self._uri))
        except grpc.RpcError as e:
            raise NotConnectError("Connect error: <{}>".format(e))
        # Unexpected error
        except Exception as e:
            raise NotConnectError("Error occurred when trying to connect server:\n"
                                  "\t<{}>".format(str(e)))

    @property
    def server_address(self):
        """
        Server network address
        """
        return self._uri

    def server_version(self, timeout=30):
        """
        Provide server version

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self._cmd(cmd='version', timeout=timeout)

    def server_status(self, timeout=30):
        """
        Provide server status

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self._cmd(cmd='status', timeout=timeout)

    @error_handler(None)
    def _cmd(self, cmd, timeout=30):
        cmd = Prepare.cmd(cmd)
        rf = self._stub.Cmd.future(cmd, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        return response.string_reply

    @error_handler()
    def create_collection(self, collection_name, fields, timeout=30):
        collection_schema = Prepare.collection_schema(collection_name, fields)
        rf = self._stub.CreateCollection.future(collection_schema, wait_for_ready=True, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            LOGGER.error(status)
            raise BaseException(status.error_code, status.reason)

    @error_handler(False)
    def has_collection(self, collection_name, timeout=30, **kwargs):
        """

        This method is used to test collection existence.

        :param collection_name: collection name is going to be tested.
        :type  collection_name: str
        :param timeout: time waiting for server response
        :type  timeout: int

        :return:
            Status: indicate if vectors inserted successfully
            bool if given collection_name exists

        """

        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.HasCollection.future(collection_name, wait_for_ready=True, timeout=timeout)
        reply = rf.result()
        if reply.status.error_code == 0:
            return reply.bool_reply

        raise BaseException(reply.status.error_code, reply.status.reason)

    @error_handler(None)
    def describe_collection(self, collection_name, timeout=30, **kwargs):
        collection_name = Prepare.collection_name(collection_name)
        rf = self._stub.DescribeCollection.future(collection_name, wait_for_ready=True, timeout=timeout)
        response = rf.result()

        if response.status.error_code == 0:
            return CollectionSchema(raw=response).dict()

        LOGGER.error(response.status)
        raise BaseException(response.status.error_code, response.status.reason)

    @error_handler(None)
    def count_collection(self, collection_name, timeout=30, **kwargs):
        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.CountCollection.future(collection_name, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.status.error_code == 0:
            return response.collection_row_count

        raise BaseException(response.status.error_code, response.status.reason)

    @error_handler([])
    def show_collections(self, timeout=30):
        cmd = Prepare.cmd('show_collections')
        rf = self._stub.ShowCollections.future(cmd, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.status.error_code == 0:
            return [name for name in response.collection_names if len(name) > 0]
        raise BaseException(response.status.error_code, message=response.status.reason)

    @error_handler(None)
    def show_collection_info(self, collection_name, timeout=30):
        request = grpc_types.CollectionName(collection_name=collection_name)

        rf = self._stub.ShowCollectionInfo.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        rpc_status = response.status

        if rpc_status.error_code == 0:
            json_info = response.json_info
            return {} if not json_info else ujson.loads(json_info)

        raise BaseException(rpc_status.error_code, rpc_status.reason)

    @error_handler()
    def preload_collection(self, collection_name, timeout=None):
        """
        Load collection to cache in advance

        :type collection_name: str
        :param collection_name: collection to preload

        :returns:
            Status:  indicate if invoke is successful
        """

        collection_name = Prepare.collection_name(collection_name)
        status = self._stub.PreloadCollection.future(collection_name, wait_for_ready=True, timeout=timeout).result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

    @error_handler()
    def reload_segments(self, collection_name, segment_ids, timeout=30):
        file_ids = list(map(int_or_str, segment_ids))
        request = Prepare.reload_param(collection_name, file_ids)
        status = self._stub.ReloadSegments.future(request, wait_for_ready=True, timeout=timeout).result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

    @error_handler()
    def drop_collection(self, collection_name, timeout=20):
        """
        Delete collection with collection_name

        :type  collection_name: str
        :param collection_name: Name of the collection being deleted

        :return: Status, indicate if operation is successful
        :rtype: Status
        """

        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.DropCollection.future(collection_name, wait_for_ready=True, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        return Status(status.error_code, status.reason)

    @error_handler([])
    def insert(self, collection_name, entities, ids=None, partition_tag=None, params=None, timeout=None, **kwargs):
        """
        Add vectors to collection

        :param ids: list of id
        :type  ids: list[int]

        :type  collection_name: str
        :param collection_name: collection name been inserted

        :type  records: list[list[float]]

                `example records: [[1.2345],[1.2345]]`

                `OR using Prepare.records`

        :param records: list of vectors been inserted

        :type partition_tag: str or None.
            If partition_tag is None, vectors will be inserted into collection rather than partitions.

        :param partition_tag: the tag string of collection

        :type

        :type  timeout: int
        :param timeout: time waiting for server response

        :returns:
            Status: indicate if vectors inserted successfully
            ids: list of id, after inserted every vector is given a id
        :rtype: (Status, list(int))
        """
        insert_param = kwargs.get('insert_param', None)

        if insert_param and not isinstance(insert_param, grpc_types.InsertParam):
            raise ParamError("The value of key 'insert_param' is invalid")

        body = insert_param if insert_param \
            else Prepare.insert_param(collection_name, entities, partition_tag, ids, params)

        # rf = self._stub.Insert.future(body, wait_for_ready=True, timeout=timeout)
        rf = self._stub.Insert.future(body, wait_for_ready=True, timeout=timeout)
        if kwargs.get("_async", False) is True:
            cb = kwargs.get("_callback", None)
            return InsertFuture(rf, cb)

        response = rf.result()
        if response.status.error_code == 0:
            return list(response.entity_id_array)

        raise BaseException(response.status.error_code, response.status.reason)

    @error_handler([])
    def get_entities_by_ids(self, collection_name, ids, fields, timeout=30):
        request = Prepare.get_entity_by_id_param(collection_name, ids, fields)

        rf = self._stub.GetEntityByID.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            # TODO: handler response
            return Entities(response)

        raise BaseException(code=status.error_code, message=status.reason)

    @error_handler([])
    def get_vector_ids(self, collection_name, segment_id, timeout=30):
        request = grpc_types.GetEntityIDsParam(collection_name=collection_name, segment_id=segment_id)

        rf = self._stub.GetEntityIDs.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()

        if response.status.error_code == 0:
            return list(response.entity_id_array)

        raise BaseException(response.status.error_code, response.status.reason)

    @error_handler()
    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        index_param = Prepare.index_param(collection_name, field_name, params)
        future = self._stub.CreateIndex.future(index_param, wait_for_ready=True, timeout=timeout)
        if kwargs.get('_async', False):
            cb = kwargs.get("_callback", None)
            return CreateIndexFuture(future, cb)
        status = future.result()

        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        return Status(status.error_code, status.reason)

    @error_handler()
    def drop_index(self, collection_name, field_name, index_name, timeout=30):
        # TODO: TODO
        request = Prepare.index_param(collection_name, field_name, index_name, None)
        rf = self._stub.DropIndex.future(request, wait_for_ready=True, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        return Status(status.error_code, status.reason)

    @error_handler()
    def create_partition(self, collection_name, partition_tag, timeout=30):
        request = Prepare.partition_param(collection_name, partition_tag)
        rf = self._stub.CreatePartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @error_handler(False)
    def has_partition(self, collection_name, partition_tag, timeout=30):
        request = Prepare.partition_param(collection_name, partition_tag)
        rf = self._stub.HasPartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.bool_reply

        raise BaseException(status.error_code, status.reason)

    @error_handler([])
    def show_partitions(self, collection_name, timeout=30):
        request = Prepare.collection_name(collection_name)

        rf = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return [p for p in response.partition_tag_array]

        raise BaseException(status.error_code, status.reason)

    @error_handler()
    def drop_partition(self, collection_name, partition_tag, timeout=30):
        """
        Drop specific partition under designated collection.

        :param collection_name: target collection name.
        :type  collection_name: str

        :param partition_tag: tag name of specific partition
        :type  partition_tag: str

        :param timeout: time waiting for response.
        :type  timeout: int

        :return:
            Status: indicate if operation is successful

        """
        request = grpc_types.PartitionParam(collection_name=collection_name, tag=partition_tag)

        rf = self._stub.DropPartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()

        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

        return Status(response.error_code, response.reason)

    @error_handler(None)
    def search(self, collection_name, query_entities, partition_tags=None, fields=None, **kwargs):
        request = Prepare.search_param(collection_name, query_entities, partition_tags, fields)
        self._search_hook.pre_search()
        to = kwargs.get("timeout", None)
        ft = self._stub.Search.future(request, wait_for_ready=True, timeout=to)
        if kwargs.get("_async", False) is True:
            func = kwargs.get("_callback", None)
            return SearchFuture(ft, func)

        response = ft.result()
        self._search_hook.aft_search()

        if self._search_hook.on_response():
            return response

        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        # TODO: handler response
        # resutls = self._search_hook.handle_response(response)

        return QueryResult(response)

    @error_handler(None)
    def search_by_ids(self, collection_name, ids, top_k, partition_tags=None, params=None, timeout=None, **kwargs):
        request = Prepare.search_by_ids_param(collection_name, ids, top_k, partition_tags, params)
        if kwargs.get("_async", False) is True:
            future = self._stub.SearchByID.future(request, wait_for_ready=True, timeout=timeout)

            func = kwargs.get("_callback", None)
            return SearchFuture(future, func)

        ft = self._stub.SearchByID.future(request, wait_for_ready=True, timeout=timeout)
        response = ft.result()
        self._search_hook.aft_search()

        if self._search_hook.on_response():
            return response

        if response.status.error_code != 0:
            return Status(code=response.status.error_code,
                          message=response.status.reason), []

        # return Status(message='Search vectors successfully!'), \
        #        self._search_hook.handle_response(response)
        return QueryResult(response)

    @error_handler(None)
    def search_in_files(self, collection_name, segment_ids, query_entities, fields, params=None, timeout=None, **kwargs):
        """
        Query vectors in a collection, in specified files.

        The server store vector data into multiple files if the size of vectors
        exceeds file size threshold. It is supported to search in several files
        by specifying file ids. However, these file ids are stored in db in server,
        and python sdk doesn't apply any APIs get them at client. It's a specific
        method used in shards. Obtain more detail about milvus shards, see
        <a href="https://github.com/milvus-io/milvus/tree/0.6.0/shards">

        :type  collection_name: str
        :param collection_name: collection name been queried

        :type  file_ids: list[str] or list[int]
        :param file_ids: Specified files id array

        :type  query_records: list[list[float]]
        :param query_records: all vectors going to be queried

        :param query_ranges: Optional ranges for conditional search.
            If not specified, search in the whole collection

        :type  top_k: int
        :param top_k: how many similar vectors will be searched

        :returns:
            Status:  indicate if query is successful
            results: query result

        :rtype: (Status, TopKQueryResult)
        """

        file_ids = list(map(int_or_str, segment_ids))
        infos = Prepare.search_vector_in_files_param(collection_name, file_ids, query_entities, fields, params=params)

        self._search_file_hook.pre_search()

        if kwargs.get("_async", False) is True:
            future = self._stub.SearchInFiles.future(infos, wait_for_ready=True, timeout=timeout)

            func = kwargs.get("_callback", None)
            return SearchFuture(future, func)

        ft = self._stub.SearchInFiles.future(infos, wait_for_ready=True, timeout=timeout)
        response = ft.result()
        self._search_file_hook.aft_search()

        if self._search_file_hook.on_response():
            return response

        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        return QueryResult(response)

    @error_handler()
    def delete_by_id(self, collection_name, id_array, timeout=None):
        request = Prepare.delete_by_id_param(collection_name, id_array)

        rf = self._stub.DeleteByID.future(request, wait_for_ready=True, timeout=timeout)
        status = rf.result()
        # status = self._stub.DeleteByID.future(request).result(timeout=timeout)
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        return Status(status.error_code, status.reason)

    @error_handler()
    def flush(self, collection_name_array, timeout=None, **kwargs):
        request = Prepare.flush_param(collection_name_array)
        future = self._stub.Flush.future(request, wait_for_ready=True, timeout=timeout)
        if kwargs.get("_async", False):
            cb = kwargs.get("_callback", None)
            return FlushFuture(future, cb)
        response = future.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @error_handler()
    def compact(self, collection_name, timeout, **kwargs):
        request = Prepare.compact_param(collection_name)
        future = self._stub.Compact.future(request, wait_for_ready=True, timeout=timeout)
        if kwargs.get("_async", False):
            cb = kwargs.get("_callback", None)
            return CompactFuture(future, cb)
        response = future.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

        return Status(response.error_code, response.reason)

    def set_config(self, parent_key, child_key, value):
        cmd = "SET {}.{} {}".format(parent_key, child_key, value)
        return self._cmd(cmd)

    def get_config(self, parent_key, child_key):
        cmd = "GEt {}.{}".format(parent_key, child_key)
        return self._cmd(cmd)
