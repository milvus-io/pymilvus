import datetime
import time
from urllib.parse import urlparse
import logging
import threading
import json
import functools
import copy
import math

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import common_pb2 as common_types
from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import milvus_pb2 as milvus_types
from .abstract import QueryResult, CollectionSchema, ChunkedQueryResult, MutationResult
from .prepare import Prepare
from .types import Status, IndexState, DataType, DeployMode, ErrorCode
from .check import (
    int_or_str,
    is_legal_host,
    is_legal_port,
)
from .utils import len_of

from .abs_client import AbsMilvus
from .asynch import (
    SearchFuture,
    MutationFuture,
    CreateIndexFuture,
    CreateFlatIndexFuture,
    FlushFuture,
    LoadCollectionFuture,
    LoadPartitionsFuture,
    ChunkedSearchFuture
)

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
                LOGGER.error("\nAddr [{}] {}\nRequest timeout: {}\n\t{}".format(self.server_address, func.__name__, e,
                                                                                record_dict))
                raise e
            except grpc.RpcError as e:
                record_dict["RPC error"] = str(datetime.datetime.now())
                LOGGER.error(
                    "\nAddr [{}] {}\nRPC error: {}\n\t{}".format(self.server_address, func.__name__, e, record_dict))
                raise e
            except Exception as e:
                record_dict["Exception"] = str(datetime.datetime.now())
                LOGGER.error("\nAddr [{}] {}\nExcepted error: {}\n\t{}".format(self.server_address, func.__name__, e,
                                                                               record_dict))
                raise e

        return handler

    return wrapper


def check_has_collection(func):
    @functools.wraps(func)
    def handler(self, *args, **kwargs):
        collection_name = args[0]
        if not self.has_collection(collection_name):
            raise CollectionNotExistException(ErrorCode.CollectionNotExists,
                                              f"collection {collection_name} doesn't exist!")
        return func(self, *args, **kwargs)

    return handler


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


class RegistryHandler:
    def __init__(self, host=None, port=None, pre_ping=True, **kwargs):
        self._channel = None
        self._stub = None
        self._uri = None
        self.status = None
        self._connected = False
        self._pre_ping = pre_ping
        # if self._pre_ping:
        self._max_retry = kwargs.get("max_retry", 5)

        # record
        self._id = kwargs.get("conn_id", 0)

        # condition
        self._condition = threading.Condition()
        self._request_id = 0

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
        # self._channel.close()
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
        # self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)
        self._stub = milvus_pb2_grpc.ProxyServiceStub(self._channel)
        self.status = Status()

    def _pre_request(self):
        if self._pre_ping:
            self.ping()

    def _get_request_id(self):
        with self._condition:
            _id = self._request_id
            self._request_id += 1
            return _id

    def ping(self):
        begin_timeout = 1
        timeout = begin_timeout
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
                        timeout *= 2
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

    @error_handler(None)
    def register_link(self, timeout=20):
        request = common_types.Empty()
        rf = self._stub.RegisterLink.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.status.error_code != 0:
            LOGGER.error(response.status)
            raise BaseException(response.status.error_code, response.status.reason)
        return response.address.ip, response.address.port


class GrpcHandler(AbsMilvus):
    def __init__(self, host=None, port=None, pre_ping=True, **kwargs):
        self._channel = None
        self._stub = None
        self._uri = None
        self.status = None
        self._connected = False
        self._pre_ping = pre_ping
        # if self._pre_ping:
        self._max_retry = kwargs.get("max_retry", 5)

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
        _channel = kwargs.get("channel", None)
        self._setup(host, port, _uri, _channel, pre_ping)

    def __str__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items() if not key.startswith('_')]
        return '<Milvus: {}>'.format(', '.join(attr_list))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _setup(self, host, port, uri, channel=None, pre_ping=False):
        """
        Create a grpc channel and a stub

        :raises: NotConnectError

        """
        self._uri = set_uri(host, port, uri)
        if channel:
            self._channel = channel
        else:
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

    def ping(self):
        begin_timeout = 1
        timeout = begin_timeout
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
                        timeout *= 2
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

    @error_handler()
    def create_collection(self, collection_name, fields, timeout=None, **kwargs):
        request = Prepare.create_collection_request(collection_name, fields, **kwargs)

        # TODO(wxyu): In grpcio==1.37.1, `wait_for_ready` is an EXPERIMENTAL argument, while it's not supported in
        #  grpcio-testing==1.37.1 . So that we remove the argument in order to using grpc-testing in unittests.
        # rf = self._stub.CreateCollection.future(request, wait_for_ready=True, timeout=timeout)

        rf = self._stub.CreateCollection.future(request, timeout=timeout)
        if kwargs.get("_async", False):
            return rf
        status = rf.result()
        if status.error_code != 0:
            LOGGER.error(status)
            raise BaseException(status.error_code, status.reason)

        # self.load_collection("", collection_name)

        # return Status(status.error_code, status.reason)

    @error_handler()
    def drop_collection(self, collection_name, timeout=None):
        request = Prepare.drop_collection_request(collection_name)

        rf = self._stub.DropCollection.future(request, wait_for_ready=True, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        # return Status(status.error_code, status.reason)

    @error_handler(False)
    def has_collection(self, collection_name, timeout=None, **kwargs):
        request = Prepare.has_collection_request(collection_name)

        rf = self._stub.HasCollection.future(request, wait_for_ready=True, timeout=timeout)
        reply = rf.result()
        if reply.status.error_code == 0:
            # return Status(reply.status.error_code, reply.status.reason), reply.value
            return reply.value

        raise BaseException(reply.status.error_code, reply.status.reason)

    @error_handler(None)
    def describe_collection(self, collection_name, timeout=None, **kwargs):
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status

        if status.error_code == 0:
            # return Status(status.error_code, status.reason), CollectionSchema(raw=response).dict()
            return CollectionSchema(raw=response).dict()

        LOGGER.error(status)
        raise DescribeCollectionException(status.error_code, status.reason)
        # return Status(status.error_code, status.reason), CollectionSchema(None)

    @error_handler([])
    def list_collections(self, timeout=None):
        request = Prepare.show_collections_request()
        rf = self._stub.ShowCollections.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if response.status.error_code == 0:
            # return Status(status.error_code, status.reason), [name for name in response.values if len(name) > 0]
            return list(response.collection_names)
        raise BaseException(status.error_code, status.reason)

    @error_handler()
    def create_partition(self, collection_name, partition_name, timeout=None):
        request = Prepare.create_partition_request(collection_name, partition_name)
        rf = self._stub.CreatePartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)
        # self.load_partitions("", collection_name, [partition_name], timeout=timeout)
        # return Status(response.status.error_code, response.status.reason)

    @error_handler()
    def drop_partition(self, collection_name, partition_name, timeout=None):
        request = Prepare.drop_partition_request(collection_name, partition_name)

        rf = self._stub.DropPartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()

        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

        # return Status(response.error_code, response.reason)

    @error_handler(False)
    def has_partition(self, collection_name, partition_name, timeout=None):
        request = Prepare.has_partition_request(collection_name, partition_name)
        rf = self._stub.HasPartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.value

        raise BaseException(status.error_code, status.reason)

    @error_handler(None)
    def get_partition_info(self, collection_name, partition_name, timeout=None):
        request = Prepare.partition_stats_request(collection_name, partition_name)
        rf = self._stub.DescribePartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            statistics = response.statistics
            info_dict = dict()
            for kv in statistics:
                info_dict[kv.key] = kv.value
            return info_dict
        raise BaseException(status.error_code, status.reason)

    @error_handler([])
    def list_partitions(self, collection_name, timeout=None):
        request = Prepare.show_partitions_request(collection_name)

        rf = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return list(response.partition_names)

        raise BaseException(status.error_code, status.reason)

    @error_handler(None)
    def get_partition_stats(self, collection_name, partition_name, timeout=None, **kwargs):
        index_param = Prepare.get_partition_stats_request(collection_name, partition_name)
        future = self._stub.GetPartitionStatistics.future(index_param, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise BaseException(status.error_code, status.reason)

    def _prepare_bulk_insert_request(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        insert_param = kwargs.get('insert_param', None)

        if insert_param and not isinstance(insert_param, milvus_types.RowBatch):
            raise ParamError("The value of key 'insert_param' is invalid")
        if not isinstance(entities, list):
            raise ParamError("None entities, please provide valid entities.")

        collection_schema = self.describe_collection(collection_name, timeout=timeout, **kwargs)

        fields_name = list()
        for i in range(len(entities)):
            if "name" in entities[i]:
                fields_name.append(entities[i]["name"])

        fields_info = collection_schema["fields"]

        request = insert_param if insert_param \
            else Prepare.bulk_insert_param(collection_name, entities, partition_name, fields_info)

        return request

    @error_handler([])
    def bulk_insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        try:
            request = self._prepare_bulk_insert_request(collection_name, entities, partition_name, timeout, **kwargs)
            rf = self._stub.Insert.future(request, wait_for_ready=True, timeout=timeout)
            if kwargs.get("_async", False) is True:
                cb = kwargs.get("_callback", None)
                return MutationFuture(rf, cb)

            response = rf.result()
            if response.status.error_code == 0:
                return MutationResult(response)

            raise BaseException(response.status.error_code, response.status.reason)
        except Exception as err:
            if kwargs.get("_async", False):
                return MutationFuture(None, None, err)
            raise err

    def _prepare_search_request(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None,
                                **kwargs):
        rf = self._stub.HasCollection.future(Prepare.has_collection_request(collection_name), wait_for_ready=True,
                                             timeout=timeout)
        reply = rf.result()
        if reply.status.error_code != 0 or not reply.value:
            raise CollectionNotExistException(reply.status.error_code, "collection not exists")

        collection_schema = self.describe_collection(collection_name, timeout)
        auto_id = collection_schema["auto_id"]
        request = Prepare.search_request(collection_name, query_entities, partition_names, fields,
                                         schema=collection_schema)

        return request, auto_id

    def _divide_search_request(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None,
                               **kwargs):
        rf = self._stub.HasCollection.future(Prepare.has_collection_request(collection_name), wait_for_ready=True,
                                             timeout=timeout)
        reply = rf.result()
        if reply.status.error_code != 0 or not reply.value:
            raise CollectionNotExistException(reply.status.error_code, "collection not exists")

        collection_schema = self.describe_collection(collection_name, timeout)
        auto_id = collection_schema["auto_id"]
        requests = Prepare.divide_search_request(collection_name, query_entities, partition_names, fields,
                                                 schema=collection_schema)

        return requests, auto_id

    @error_handler(None)
    def _execute_search_requests(self, requests, timeout=None, **kwargs):
        auto_id = kwargs.get("auto_id", True)

        try:
            raws = []
            futures = []

            # step 1: get future object
            for request in requests:
                self._search_hook.pre_search()

                ft = self._stub.Search.future(request, wait_for_ready=True, timeout=timeout)
                futures.append(ft)

            if kwargs.get("_async", False):
                func = kwargs.get("_callback", None)
                return ChunkedSearchFuture(futures, func, auto_id)

            # step2: get results
            for ft in futures:
                response = ft.result()

                if response.status.error_code != 0:
                    raise BaseException(response.status.error_code, response.status.reason)

                raws.append(response)

            return ChunkedQueryResult(raws, auto_id)

        except Exception as pre_err:
            if kwargs.get("_async", False):
                return SearchFuture(None, None, True, pre_err)
            raise pre_err

    def _batch_search(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None, **kwargs):
        requests, auto_id = self._divide_search_request(collection_name, query_entities, partition_names,
                                                        fields, **kwargs)
        kwargs["auto_id"] = auto_id
        return self._execute_search_requests(requests, timeout, **kwargs)

    @error_handler(None)
    def _total_search(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None, **kwargs):
        request, auto_id = self._prepare_search_request(collection_name, query_entities, partition_names,
                                                        fields, timeout, **kwargs)
        kwargs["auto_id"] = auto_id
        return self._execute_search_requests([request], timeout, **kwargs)

    @error_handler(None)
    def search(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None, **kwargs):
        if kwargs.get("_deploy_mode", DeployMode.Distributed) == DeployMode.StandAlone:
            return self._total_search(collection_name, query_entities, partition_names, fields, timeout, **kwargs)
        return self._batch_search(collection_name, query_entities, partition_names, fields, timeout, **kwargs)

    @error_handler(None)
    @check_has_collection
    def search_with_expression(self, collection_name, data, anns_field, param, limit, expression=None,
                               partition_names=None,
                               output_fields=None, timeout=None, **kwargs):
        _kwargs = copy.deepcopy(kwargs)
        collection_schema = self.describe_collection(collection_name, timeout)
        auto_id = collection_schema["auto_id"]
        _kwargs["schema"] = collection_schema
        requests = Prepare.search_requests_with_expr(collection_name, data, anns_field, param, limit, expression,
                                                     partition_names, output_fields, **_kwargs)
        _kwargs.pop("schema")
        _kwargs["auto_id"] = auto_id
        return self._execute_search_requests(requests, timeout, **_kwargs)

    @error_handler(None)
    def get_query_segment_infos(self, collection_name, timeout=30, **kwargs):
        req = Prepare.get_query_segment_info_request(collection_name)
        future = self._stub.GetQuerySegmentInfo.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of QuerySegmentInfo
        raise BaseException(status.error_code, status.reason)

    @error_handler()
    def wait_for_load_index_done(self, collection_name, timeout=None, **kwargs):
        def load_index_done():
            query_segment_infos = self.get_query_segment_infos(collection_name)
            persistent_segment_infos = self.get_persistent_segment_infos(collection_name)

            query_segment_ids = [info.segmentID for info in query_segment_infos]
            persistent_segment_ids = [info.segmentID for info in persistent_segment_infos]

            if len(persistent_segment_ids) != len(query_segment_ids):
                return False

            if len(query_segment_ids) == 0:
                return True

            query_segment_ids.sort()
            persistent_segment_ids.sort()
            if query_segment_ids != persistent_segment_ids:
                return False

            filtered_query_segment_info = list(
                filter(lambda info: info.index_name == "_default_idx", query_segment_infos))
            filtered_query_segment_index_ids = list(map(lambda info: info.indexID, filtered_query_segment_info))
            return len(set(filtered_query_segment_index_ids)) == 1

        while True:
            time.sleep(0.5)
            if load_index_done():
                return

    @error_handler()
    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        index_type = params["index_type"].upper()
        if index_type == "FLAT":
            try:
                collection_desc = self.describe_collection(collection_name, timeout=timeout, **kwargs)
                valid_field = False
                for fields in collection_desc["fields"]:
                    if field_name != fields["name"]:
                        continue
                    if fields["type"] != DataType.FLOAT_VECTOR and fields["type"] != DataType.BINARY_VECTOR:
                        # TODO: add new error type
                        raise BaseException(Status.UNEXPECTED_ERROR,
                                            "cannot create index on non-vector field: " + field_name)
                    valid_field = True
                    break
                if not valid_field:
                    # TODO: add new error type
                    raise BaseException(Status.UNEXPECTED_ERROR,
                                        "cannot create index on non-existed field: " + field_name)
                index_desc = self.describe_index(collection_name, "", timeout=timeout, **kwargs)
                if index_desc is not None:
                    self.drop_index(collection_name, field_name, "_default_idx", timeout=timeout, **kwargs)
                res_status = Status(Status.SUCCESS, "Warning: It is not necessary to build index with index_type: FLAT")
                if kwargs.get("_async", False):
                    return CreateFlatIndexFuture(res_status)
                return res_status
            except Exception as err:
                if kwargs.get("_async", False):
                    return CreateFlatIndexFuture(None, None, err)
                raise err

        # sync flush
        _async = kwargs.get("_async", False)
        kwargs["_async"] = False
        self.flush([collection_name], timeout, **kwargs)

        index_param = Prepare.create_index__request(collection_name, field_name, params)
        future = self._stub.CreateIndex.future(index_param, wait_for_ready=True, timeout=timeout)

        if _async:
            def _check():
                if kwargs.get("sync", True):
                    if not self.wait_for_creating_index(collection_name=collection_name, field_name=field_name):
                        raise BaseException(Status.UNEXPECTED_ERROR, "create index failed")

            index_future = CreateIndexFuture(future)
            index_future.add_callback(_check)
            user_cb = kwargs.get("_callback", None)
            if user_cb:
                index_future.add_callback(user_cb)
            return index_future

        status = future.result()

        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        if kwargs.get("sync", True):
            if not self.wait_for_creating_index(collection_name=collection_name, field_name=field_name):
                raise BaseException(Status.UNEXPECTED_ERROR, "create index failed")

        return Status(status.error_code, status.reason)

    @error_handler(None)
    def describe_index(self, collection_name, index_name, timeout=None, **kwargs):
        request = Prepare.describe_index_request(collection_name, index_name)

        rf = self._stub.DescribeIndex.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            info_dict = {kv.key: kv.value for kv in response.index_descriptions[0].params}
            info_dict['field_name'] = response.index_descriptions[0].field_name
            if info_dict.get("params", None):
                info_dict["params"] = json.loads(info_dict["params"])
            return info_dict
        if status.error_code == Status.INDEX_NOT_EXIST:
            return None
        raise BaseException(status.error_code, status.reason)

    @error_handler(IndexState.Failed)
    def get_index_build_progress(self, collection_name, index_name, timeout=None):
        request = Prepare.get_index_build_progress(collection_name, index_name)
        rf = self._stub.GetIndexBuildProgress.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return {'total_rows': response.total_rows, 'indexed_rows': response.indexed_rows}
        raise BaseException(status.error_code, status.reason)

    @error_handler(IndexState.Failed)
    def get_index_state(self, collection_name, field_name, timeout=None):
        request = Prepare.get_index_state_request(collection_name, field_name)
        rf = self._stub.GetIndexState.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.state
        raise BaseException(status.error_code, status.reason)

    @error_handler(False)
    def wait_for_creating_index(self, collection_name, field_name, timeout=None):
        while True:
            time.sleep(0.5)
            state = self.get_index_state(collection_name, field_name, timeout)
            if state == IndexState.Finished:
                return True
            if state == IndexState.Failed:
                return False

    @error_handler()
    def load_collection(self, db_name, collection_name, timeout=None, **kwargs):
        request = Prepare.load_collection(db_name, collection_name)
        rf = self._stub.LoadCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)
        sync = kwargs.get("sync", True)
        if sync:
            self.wait_for_loading_collection(collection_name)

    @error_handler()
    def load_collection_progress(self, collection_name, timeout=None):
        """
        Block until load collection complete.
        """

        loaded_segments_nums = sum(info.num_rows for info in
                                   self.get_query_segment_infos(collection_name, timeout))

        total_segments_nums = sum(info.num_rows for info in
                                  self.get_persistent_segment_infos(collection_name, timeout))

        return {'num_loaded_entities': loaded_segments_nums, 'num_total_entities': total_segments_nums}

    @error_handler()
    def wait_for_loading_collection(self, collection_name, timeout=None):
        """
        Block until load collection complete.
        """
        unloaded_segments = {info.segmentID: info.num_rows for info in
                             self.get_persistent_segment_infos(collection_name, timeout)}

        while len(unloaded_segments) > 0:
            time.sleep(0.5)

            for info in self.get_query_segment_infos(collection_name, timeout):
                if 0 <= unloaded_segments.get(info.segmentID, -1) <= info.num_rows:
                    unloaded_segments.pop(info.segmentID)

    @error_handler()
    def release_collection(self, db_name, collection_name, timeout=None):
        request = Prepare.release_collection(db_name, collection_name)
        rf = self._stub.ReleaseCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @error_handler()
    def load_partitions(self, db_name, collection_name, partition_names, timeout=None, **kwargs):
        request = Prepare.load_partitions(db_name=db_name, collection_name=collection_name,
                                          partition_names=partition_names)
        future = self._stub.LoadPartitions.future(request, wait_for_ready=True, timeout=timeout)

        if kwargs.get("_async", False):
            def _check():
                if kwargs.get("sync", True):
                    self.wait_for_loading_partitions(collection_name, partition_names)

            load_partitions_future = LoadPartitionsFuture(future)
            load_partitions_future.add_callback(_check)

            user_cb = kwargs.get("_callback", None)
            if user_cb:
                load_partitions_future.add_callback(user_cb)

            return load_partitions_future

        response = future.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)
        sync = kwargs.get("sync", True)
        if sync:
            self.wait_for_loading_partitions(collection_name, partition_names)

    @error_handler()
    def wait_for_loading_partitions(self, collection_name, partition_names, timeout=None):
        """
        Block until load partition complete.
        """
        request = Prepare.show_partitions_request(collection_name)
        rf = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        pIDs = [response.partitionIDs[index] for index, p_name in enumerate(response.partition_names)
                if p_name in partition_names]

        unloaded_segments = {info.segmentID: info.num_rows for info in
                             self.get_persistent_segment_infos(collection_name, timeout)
                             if info.partitionID in pIDs}

        while len(unloaded_segments) > 0:
            time.sleep(0.5)

            for info in self.get_query_segment_infos(collection_name, timeout):
                if 0 <= unloaded_segments.get(info.segmentID, -1) <= info.num_rows:
                    unloaded_segments.pop(info.segmentID)

    @error_handler()
    def load_partitions_progress(self, collection_name, partition_names, timeout=None):
        """
        Block until load collection complete.
        """
        request = Prepare.show_partitions_request(collection_name)
        rf = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        pIDs = [response.partitionIDs[index] for index, p_name in enumerate(response.partition_names)
                if p_name in partition_names]

        total_segments_nums = sum(info.num_rows for info in
                                  self.get_persistent_segment_infos(collection_name, timeout)
                                  if info.partitionID in pIDs)

        loaded_segments_nums = sum(info.num_rows for info in
                                   self.get_query_segment_infos(collection_name, timeout)
                                   if info.partitionID in pIDs)

        return {'num_loaded_entities': loaded_segments_nums, 'num_total_entities': total_segments_nums}

    @error_handler()
    def release_partitions(self, db_name, collection_name, partition_names, timeout=None):
        request = Prepare.release_partitions(db_name, collection_name,
                                             partition_names)
        rf = self._stub.ReleasePartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @error_handler(None)
    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        index_param = Prepare.get_collection_stats_request(collection_name)
        future = self._stub.GetCollectionStatistics.future(index_param, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise BaseException(status.error_code, status.reason)

    @error_handler(None)
    def get_persistent_segment_infos(self, collection_name, timeout=None, **kwargs):
        req = Prepare.get_persistent_segment_info_request(collection_name)
        future = self._stub.GetPersistentSegmentInfo.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of PersistentSegmentInfo
        raise BaseException(status.error_code, status.reason)

    @error_handler()
    def _wait_for_flushed(self, collection_name, get_first_segment_ids_func, timeout=None, **kwargs):
        def flushed(segment_ids_to_wait):
            infos = self.get_persistent_segment_infos(collection_name, timeout, **kwargs)
            need_cnt = len(segment_ids_to_wait)
            have_cnt = 0
            for info in infos:
                if info.segmentID not in segment_ids_to_wait:
                    continue

                if info.state == common_types.SegmentState.Flushed:
                    have_cnt += 1
                    # return False
            return need_cnt == have_cnt

        first_segment_ids = get_first_segment_ids_func()
        while True:
            time.sleep(0.5)
            if flushed(first_segment_ids):
                return

    @error_handler()
    def flush(self, collection_name_array: list, timeout=None, **kwargs):
        request = Prepare.flush_param(collection_name_array)
        future = self._stub.Flush.future(request, wait_for_ready=True, timeout=timeout)

        if kwargs.get("_async", False):
            def _check():
                if kwargs.get("sync", True):
                    for collection_name in collection_name_array:
                        self._wait_for_flushed(collection_name,
                                               lambda: future.result().coll_segIDs[collection_name].data)

            flush_future = FlushFuture(future)
            flush_future.add_callback(_check)

            user_cb = kwargs.get("_callback", None)
            if user_cb:
                flush_future.add_callback(user_cb)

            return flush_future

        response = future.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)
        sync = kwargs.get("sync", True)
        if sync:
            for collection_name in collection_name_array:
                self._wait_for_flushed(collection_name, lambda: future.result().coll_segIDs[collection_name].data)

    @error_handler()
    def drop_index(self, collection_name, field_name, index_name, timeout=None, **kwargs):
        request = Prepare.drop_index_request(collection_name, field_name, index_name)
        future = self._stub.DropIndex.future(request, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @error_handler()
    def dummy(self, request_type, timeout=None, **kwargs):
        request = Prepare.dummy_request(request_type)
        future = self._stub.Dummy.future(request, wait_for_ready=True, timeout=timeout)
        return future.result()

    @error_handler(False)
    def fake_register_link(self, timeout=None):
        request = Prepare.register_link_request()
        future = self._stub.RegisterLink.future(request, wait_for_ready=True, timeout=timeout)
        return future.result().status

    @error_handler()
    def get(self, collection_name, ids, output_fields=None, partition_names=None, timeout=None):
        # TODO: some check
        request = Prepare.retrieve_request(collection_name, ids, output_fields, partition_names)
        future = self._stub.Retrieve.future(request, wait_for_ready=True, timeout=timeout)
        return future.result()

    @error_handler()
    def query(self, collection_name, expr, output_fields=None, partition_names=None, timeout=None):
        if output_fields is not None and not isinstance(output_fields, (list,)):
            raise ParamError("Invalid query format. 'output_fields' must be a list")
        request = Prepare.query_request(collection_name, expr, output_fields, partition_names)
        future = self._stub.Query.future(request, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.status.error_code == Status.EMPTY_COLLECTION:
            return list()
        if response.status.error_code != Status.SUCCESS:
            raise BaseException(response.status.error_code, response.status.reason)

        num_fields = len(response.fields_data)
        # check has fields
        if num_fields == 0:
            raise BaseException(0, "")

        # check if all lists are of the same length
        it = iter(response.fields_data)
        num_entities = len_of(next(it))
        if not all(len_of(field_data) == num_entities for field_data in it):
            raise BaseException(0, "The length of fields data is inconsistent")

        # transpose
        results = list()
        for index in range(0, num_entities):
            result = dict()
            for field_data in response.fields_data:
                if field_data.type == DataType.BOOL:
                    raise BaseException(0, "Not support bool yet")
                    # result[field_data.name] = field_data.field.scalars.data.bool_data[index]
                elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                    result[field_data.field_name] = field_data.scalars.int_data.data[index]
                elif field_data.type == DataType.INT64:
                    result[field_data.field_name] = field_data.scalars.long_data.data[index]
                elif field_data.type == DataType.FLOAT:
                    result[field_data.field_name] = round(field_data.scalars.float_data.data[index], 6)
                elif field_data.type == DataType.DOUBLE:
                    result[field_data.field_name] = field_data.scalars.double_data.data[index]
                elif field_data.type == DataType.STRING:
                    raise BaseException(0, "Not support string yet")
                    # result[field_data.field_name] = field_data.scalars.string_data.data[index]
                elif field_data.type == DataType.FLOAT_VECTOR:
                    dim = field_data.vectors.dim
                    start_pos = index * dim
                    end_pos = index * dim + dim
                    result[field_data.field_name] = [round(x, 6) for x in
                                                     field_data.vectors.float_vector.data[start_pos:end_pos]]
            results.append(result)

        return results

    @error_handler(None)
    def calc_distance(self,  vectors_left, vectors_right, params, timeout=30, **kwargs):
        req = Prepare.calc_distance_request(vectors_left, vectors_right, params)
        future = self._stub.CalcDistance.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)
        if len(response.int_dist.data) > 0:
            return response.int_dist.data
        elif len(response.float_dist.data) > 0:
            def is_l2(val):
                return val == "L2" or val == "l2"
            if is_l2(params["metric"]) and "sqrt" in params.keys() and params["sqrt"] == True:
                for i in range(len(response.float_dist.data)):
                    response.float_dist.data[i] = math.sqrt(response.float_dist.data[i])
            return response.float_dist.data
        raise BaseException(0, "Empty result returned")