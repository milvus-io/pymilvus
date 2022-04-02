import time
import logging
import json
import copy
import math

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import milvus_pb2 as milvus_types

from .abstract import CollectionSchema, ChunkedQueryResult, MutationResult
from .check import (
    is_legal_host,
    is_legal_port,
    check_pass_param,
    is_legal_index_metric_type,
    is_legal_binary_index_metric_type,
)
from .prepare import Prepare
from .types import (
    Status,
    ErrorCode,
    IndexState,
    DataType,
    CompactionState,
    State,
    CompactionPlans,
    Plan,
    get_consistency_level,
)

from .utils import (
    valid_index_types,
    valid_binary_index_types,
    valid_index_params_keys,
    check_invalid_binary_vector,
    len_of
)

from ..settings import DefaultConfig as config
from .configs import DefaultConfigs
from . import ts_utils

from .asynch import (
    SearchFuture,
    MutationFuture,
    CreateIndexFuture,
    CreateFlatIndexFuture,
    FlushFuture,
    LoadPartitionsFuture,
    ChunkedSearchFuture
)

from .exceptions import (
    ParamError,
    CollectionNotExistException,
    DescribeCollectionException,
    BaseException,
)

from ..decorators import retry_on_rpc_failure, error_handler, check_has_collection

LOGGER = logging.getLogger(__name__)


class GrpcHandler:
    def __init__(self, uri=config.GRPC_ADDRESS, host=None, port=None, channel=None, **kwargs):
        self._stub = None
        self._channel = channel

        if host is not None and port is not None \
                and is_legal_host(host) and is_legal_port(port):
            self._uri = f"{host}:{port}"
        else:
            self._uri = uri
        self._max_retry = kwargs.get("max_retry", 5)

        self._secure = kwargs.get("secure", False)
        self._setup_grpc_channel()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _wait_for_channel_ready(self):
        if self._channel is not None:
            try:
                grpc.channel_ready_future(self._channel).result(timeout=3)
                return
            except grpc.FutureTimeoutError:
                raise BaseException(Status.CONNECT_FAILED, f'Fail connecting to server on {self._uri}. Timeout')

        raise BaseException(Status.CONNECT_FAILED, 'No channel in handler, please setup grpc channel first')

    def close(self):
        self._channel.close()

    def _setup_grpc_channel(self):
        """ Create a ddl grpc channel """
        if self._channel is None:
            if not self._secure:
                self._channel = grpc.insecure_channel(
                    self._uri,
                    options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                             (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                             ('grpc.enable_retries', 1),
                             ('grpc.keepalive_time_ms', 55000)]
                )
            else:
                creds = grpc.ssl_channel_credentials(root_certificates=None, private_key=None, certificate_chain=None)
                self._channel = grpc.secure_channel(
                    self._uri,
                    creds,
                    options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                             (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                             ('grpc.enable_retries', 1),
                             ('grpc.keepalive_time_ms', 55000)]
                )
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)

    @property
    def server_address(self):
        """ Server network address """
        return self._uri

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def create_collection(self, collection_name, fields, shards_num=2, timeout=None, **kwargs):
        request = Prepare.create_collection_request(collection_name, fields, shards_num=shards_num, **kwargs)

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

    @retry_on_rpc_failure(retry_times=10, wait=1, retry_on_deadline=False)
    @error_handler
    def drop_collection(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.drop_collection_request(collection_name)

        rf = self._stub.DropCollection.future(request, wait_for_ready=True, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def has_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.has_collection_request(collection_name)

        rf = self._stub.HasCollection.future(request, wait_for_ready=True, timeout=timeout)
        reply = rf.result()
        if reply.status.error_code == 0:
            return reply.value

        raise BaseException(reply.status.error_code, reply.status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def describe_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status

        if status.error_code == 0:
            return CollectionSchema(raw=response).dict()

        raise DescribeCollectionException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def list_collections(self, timeout=None):
        request = Prepare.show_collections_request()
        rf = self._stub.ShowCollections.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if response.status.error_code == 0:
            return list(response.collection_names)

        raise BaseException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def create_partition(self, collection_name, partition_name, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.create_partition_request(collection_name, partition_name)
        rf = self._stub.CreatePartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def drop_partition(self, collection_name, partition_name, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.drop_partition_request(collection_name, partition_name)

        rf = self._stub.DropPartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()

        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def has_partition(self, collection_name, partition_name, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.has_partition_request(collection_name, partition_name)
        rf = self._stub.HasPartition.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.value

        raise BaseException(status.error_code, status.reason)

    # TODO: this is not inuse
    @error_handler
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

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def list_partitions(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.show_partitions_request(collection_name)

        rf = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return list(response.partition_names)

        raise BaseException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def get_partition_stats(self, collection_name, partition_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
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

        fields_info = collection_schema["fields"]

        request = insert_param if insert_param \
            else Prepare.bulk_insert_param(collection_name, entities, partition_name, fields_info)

        return request

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def bulk_insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        if not check_invalid_binary_vector(entities):
            raise ParamError("Invalid binary vector data exists")

        try:
            collection_id = self.describe_collection(collection_name, timeout, **kwargs)["collection_id"]

            request = self._prepare_bulk_insert_request(collection_name, entities, partition_name, timeout, **kwargs)
            rf = self._stub.Insert.future(request, wait_for_ready=True, timeout=timeout)
            if kwargs.get("_async", False) is True:
                cb = kwargs.get("_callback", None)
                f = MutationFuture(rf, cb)
                f.add_callback(ts_utils.update_ts_on_mutation(collection_id))
                return f

            response = rf.result()
            if response.status.error_code == 0:
                m = MutationResult(response)
                ts_utils.update_collection_ts(collection_id, m.timestamp)
                return m

            raise BaseException(response.status.error_code, response.status.reason)
        except Exception as err:
            if kwargs.get("_async", False):
                return MutationFuture(None, None, err)
            raise err

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def delete(self, collection_name, expression, partition_name=None, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        try:
            collection_id = self.describe_collection(collection_name, timeout, **kwargs)["collection_id"]

            req = Prepare.delete_request(collection_name, partition_name, expression)
            future = self._stub.Delete.future(req, wait_for_ready=True, timeout=timeout)

            response = future.result()
            if response.status.error_code == 0:
                m = MutationResult(response)
                ts_utils.update_collection_ts(collection_id, m.timestamp)
                return m

            raise BaseException(response.status.error_code, response.status.reason)
        except Exception as err:
            if kwargs.get("_async", False):
                return MutationFuture(None, None, err)
            raise err

    def _prepare_search_request(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None,
                                round_decimal=-1, **kwargs):
        rf = self._stub.HasCollection.future(Prepare.has_collection_request(collection_name), wait_for_ready=True,
                                             timeout=timeout)
        reply = rf.result()
        if reply.status.error_code != 0 or not reply.value:
            raise CollectionNotExistException(reply.status.error_code, "collection not exists")

        collection_schema = self.describe_collection(collection_name, timeout)
        auto_id = collection_schema["auto_id"]
        request = Prepare.search_request(collection_name, query_entities, partition_names, fields, round_decimal,
                                         schema=collection_schema)

        return request, auto_id

    def _divide_search_request(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None,
                               round_decimal=-1, **kwargs):
        rf = self._stub.HasCollection.future(Prepare.has_collection_request(collection_name), wait_for_ready=True,
                                             timeout=timeout)
        reply = rf.result()
        if reply.status.error_code != 0 or not reply.value:
            raise CollectionNotExistException(reply.status.error_code, "collection not exists")

        collection_schema = self.describe_collection(collection_name, timeout)
        auto_id = collection_schema["auto_id"]
        requests = Prepare.divide_search_request(collection_name, query_entities, partition_names, fields,
                                                 round_decimal,
                                                 schema=collection_schema)

        return requests, auto_id

    @error_handler
    def _execute_search_requests(self, requests, timeout=None, **kwargs):
        auto_id = kwargs.get("auto_id", True)

        try:
            raws = []
            futures = []

            # step 1: get future object
            for request in requests:
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
            round_decimal = kwargs.get("round_decimal", -1)
            return ChunkedQueryResult(raws, auto_id, round_decimal)

        except Exception as pre_err:
            if kwargs.get("_async", False):
                return SearchFuture(None, None, True, pre_err)
            raise pre_err

    def _batch_search(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None,
                      round_decimal=-1, **kwargs):
        requests, auto_id = self._divide_search_request(collection_name, query_entities, partition_names,
                                                        fields, timeout, round_decimal, **kwargs)
        kwargs["auto_id"] = auto_id
        kwargs["round_decimal"] = round_decimal
        return self._execute_search_requests(requests, timeout, **kwargs)

    @error_handler
    def _total_search(self, collection_name, query_entities, partition_names=None, fields=None, timeout=None,
                      round_decimal=-1, **kwargs):
        request, auto_id = self._prepare_search_request(collection_name, query_entities, partition_names,
                                                        fields, timeout, round_decimal, **kwargs)
        kwargs["auto_id"] = auto_id
        kwargs["round_decimal"] = round_decimal
        return self._execute_search_requests([request], timeout, **kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1, retry_on_deadline=False)
    @error_handler
    @check_has_collection
    def search(self, collection_name, data, anns_field, param, limit,
               expression=None, partition_names=None, output_fields=None,
               timeout=None, round_decimal=-1, **kwargs):
        check_pass_param(
            limit=limit,
            round_decimal=round_decimal,
            anns_field=anns_field,
            search_data=data,
            partition_name_array=partition_names,
            output_fields=output_fields,
            travel_timestamp=kwargs.get("travel_timestamp", 0),
            guarantee_timestamp=kwargs.get("guarantee_timestamp", 0)
        )

        _kwargs = copy.deepcopy(kwargs)
        collection_schema = self.describe_collection(collection_name, timeout)
        collection_id = collection_schema["collection_id"]
        auto_id = collection_schema["auto_id"]
        consistency_level = collection_schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = get_consistency_level(_kwargs.get("consistency_level", consistency_level))
        _kwargs["schema"] = collection_schema

        ts_utils.construct_guarantee_ts(consistency_level, collection_id, _kwargs)

        requests = Prepare.search_requests_with_expr(collection_name, data, anns_field, param, limit, expression,
                                                     partition_names, output_fields, round_decimal, **_kwargs)
        _kwargs.pop("schema")
        _kwargs["auto_id"] = auto_id
        _kwargs["round_decimal"] = round_decimal

        return self._execute_search_requests(requests, timeout, **_kwargs)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def get_query_segment_info(self, collection_name, timeout=30, **kwargs):
        req = Prepare.get_query_segment_info_request(collection_name)
        future = self._stub.GetQuerySegmentInfo.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of QuerySegmentInfo
        raise BaseException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def create_alias(self, collection_name, alias, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.create_alias_request(collection_name, alias)
        rf = self._stub.CreateAlias.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def drop_alias(self, alias, timeout=None, **kwargs):
        request = Prepare.drop_alias_request(alias)
        rf = self._stub.DropAlias.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def alter_alias(self, collection_name, alias, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.alter_alias_request(collection_name, alias)
        rf = self._stub.AlterAlias.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        def check_index_params(params):
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

        check_index_params(params)
        collection_desc = self.describe_collection(collection_name, timeout=timeout, **kwargs)
        valid_field = False
        for fields in collection_desc["fields"]:
            if field_name != fields["name"]:
                continue
            if fields["type"] != DataType.FLOAT_VECTOR and fields["type"] != DataType.BINARY_VECTOR:
                # TODO: add new error type
                raise BaseException(Status.UNEXPECTED_ERROR,
                                    "cannot create index on non-vector field: " + str(field_name))
            valid_field = True
            break
        if not valid_field:
            # TODO: add new error type
            raise BaseException(Status.UNEXPECTED_ERROR,
                                "cannot create index on non-existed field: " + str(field_name))
        index_type = params["index_type"].upper()
        if index_type == "FLAT":
            try:
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
                    index_success, fail_reason = self.wait_for_creating_index(collection_name=collection_name,
                                                                              field_name=field_name, timeout=timeout)
                    if not index_success:
                        raise BaseException(Status.UNEXPECTED_ERROR, fail_reason)

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
            index_success, fail_reason = self.wait_for_creating_index(collection_name=collection_name,
                                                                      field_name=field_name, timeout=timeout)
            if not index_success:
                raise BaseException(Status.UNEXPECTED_ERROR, fail_reason)

        return Status(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def describe_index(self, collection_name, index_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
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

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def get_index_build_progress(self, collection_name, index_name, timeout=None):
        request = Prepare.get_index_build_progress(collection_name, index_name)
        rf = self._stub.GetIndexBuildProgress.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return {'total_rows': response.total_rows, 'indexed_rows': response.indexed_rows}
        raise BaseException(status.error_code, status.reason)

    @error_handler
    def get_index_state(self, collection_name, field_name, timeout=None):
        request = Prepare.get_index_state_request(collection_name, field_name)
        rf = self._stub.GetIndexState.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.state, response.fail_reason
        raise BaseException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def wait_for_creating_index(self, collection_name, field_name, timeout=None):
        start = time.time()
        while True:
            time.sleep(0.5)
            state, fail_reason = self.get_index_state(collection_name, field_name, timeout)
            if state == IndexState.Finished:
                return True, fail_reason
            if state == IndexState.Failed:
                return False, fail_reason
            end = time.time()
            if timeout is not None:
                if end - start > timeout:
                    raise BaseException(1, "CreateIndex Timeout")

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def load_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.load_collection("", collection_name)
        rf = self._stub.LoadCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)
        _async = kwargs.get("_async", False)
        if not _async:
            self.wait_for_loading_collection(collection_name, timeout)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def load_collection_progress(self, collection_name, timeout=None):
        """ Return loading progress of collection """

        loaded_segments_nums = sum(info.num_rows for info in
                                   self.get_query_segment_info(collection_name, timeout))

        total_segments_nums = sum(info.num_rows for info in
                                  self.get_persistent_segment_infos(collection_name, timeout))

        progress = (loaded_segments_nums / total_segments_nums) * 100 if loaded_segments_nums < total_segments_nums else 100

        return {'loading_progress': f"{progress:.0f}%"}

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def wait_for_loading_collection(self, collection_name, timeout=None):
        return self._wait_for_loading_collection_v2(collection_name, timeout)

    # TODO seems not in use
    def _wait_for_loading_collection_v1(self, collection_name, timeout=None):
        """ Block until load collection complete. """
        unloaded_segments = {info.segmentID: info.num_rows for info in
                             self.get_persistent_segment_infos(collection_name, timeout)}

        while len(unloaded_segments) > 0:
            time.sleep(0.5)

            for info in self.get_query_segment_info(collection_name, timeout):
                if 0 <= unloaded_segments.get(info.segmentID, -1) <= info.num_rows:
                    unloaded_segments.pop(info.segmentID)

    def _wait_for_loading_collection_v2(self, collection_name, timeout=None):
        """ Block until load collection complete. """
        request = Prepare.show_collections_request([collection_name])

        while True:
            future = self._stub.ShowCollections.future(request, wait_for_ready=True, timeout=timeout)
            response = future.result()

            if response.status.error_code != 0:
                raise BaseException(response.status.error_code, response.status.reason)

            ol = len(response.collection_names)
            pl = len(response.inMemory_percentages)

            if ol != pl:
                raise BaseException(ErrorCode.UnexpectedError,
                                    f"len(collection_names) ({ol}) != len(inMemory_percentages) ({pl})")

            for i, coll_name in enumerate(response.collection_names):
                if coll_name == collection_name and response.inMemory_percentages[i] == 100:
                    return

            time.sleep(DefaultConfigs.WaitTimeDurationWhenLoad)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def release_collection(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.release_collection("", collection_name)
        rf = self._stub.ReleaseCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def load_partitions(self, collection_name, partition_names, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        request = Prepare.load_partitions("", collection_name, partition_names)
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

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def wait_for_loading_partitions(self, collection_name, partition_names, timeout=None):
        return self._wait_for_loading_partitions_v2(collection_name, partition_names, timeout)

    # TODO seems not in use
    def _wait_for_loading_partitions_v1(self, collection_name, partition_names, timeout=None):
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

            for info in self.get_query_segment_info(collection_name, timeout):
                if 0 <= unloaded_segments.get(info.segmentID, -1) <= info.num_rows:
                    unloaded_segments.pop(info.segmentID)

    def _wait_for_loading_partitions_v2(self, collection_name, partition_names, timeout=None):
        """
        Block until load partition complete.
        """
        request = Prepare.show_partitions_request(collection_name, partition_names)

        while True:
            future = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
            response = future.result()

            status = response.status
            if status.error_code != 0:
                raise BaseException(status.error_code, status.reason)

            ol = len(response.partition_names)
            pl = len(response.inMemory_percentages)

            if ol != pl:
                raise BaseException(ErrorCode.UnexpectedError,
                                    f"len(partition_names) ({ol}) != len(inMemory_percentages) ({pl})")

            loaded_histogram = dict()
            for i, par_name in enumerate(response.partition_names):
                loaded_histogram[par_name] = response.inMemory_percentages[i]

            ok = True
            for par_name in partition_names:
                if loaded_histogram.get(par_name, 0) != 100:
                    ok = False
                    break

            if ok:
                return

            time.sleep(DefaultConfigs.WaitTimeDurationWhenLoad)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def load_partitions_progress(self, collection_name, partition_names, timeout=None):
        """ Return loading progress of partitions """
        request = Prepare.show_partitions_request(collection_name)
        rf = self._stub.ShowPartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

        pIDs = []
        pNames = []
        for index, p_name in enumerate(response.partition_names):
            if p_name in partition_names:
                pIDs.append(response.partitionIDs[index])
                pNames.append((p_name))

        # all partition names must be valid, otherwise throw exception
        for name in partition_names:
            if name not in pNames:
                msg = "partitionID of partitionName:" + name + " can not be found"
                raise BaseException(1, msg)

        total_segments_nums = sum(info.num_rows for info in
                                  self.get_persistent_segment_infos(collection_name, timeout)
                                  if info.partitionID in pIDs)

        loaded_segments_nums = sum(info.num_rows for info in
                                   self.get_query_segment_info(collection_name, timeout)
                                   if info.partitionID in pIDs)

        progress = (loaded_segments_nums / total_segments_nums) * 100 if loaded_segments_nums < total_segments_nums else 100

        return {'loading_progress': f"{progress:.0f}%"}

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def release_partitions(self, collection_name, partition_names, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        request = Prepare.release_partitions("", collection_name, partition_names)
        rf = self._stub.ReleasePartitions.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        index_param = Prepare.get_collection_stats_request(collection_name)
        future = self._stub.GetCollectionStatistics.future(index_param, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise BaseException(status.error_code, status.reason)

    @error_handler
    def get_flush_state(self, segment_ids, timeout=None, **kwargs):
        req = Prepare.get_flush_state_request(segment_ids)
        future = self._stub.GetFlushState.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.flushed  # todo: A wrapper class of PersistentSegmentInfo
        raise BaseException(status.error_code, status.reason)

    # TODO seem not in use
    @error_handler
    def get_persistent_segment_infos(self, collection_name, timeout=None, **kwargs):
        req = Prepare.get_persistent_segment_info_request(collection_name)
        future = self._stub.GetPersistentSegmentInfo.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of PersistentSegmentInfo
        raise BaseException(status.error_code, status.reason)

    @error_handler
    def _wait_for_flushed(self, segment_ids, timeout=None, **kwargs):
        flush_ret = False
        while not flush_ret:
            flush_ret = self.get_flush_state(segment_ids, timeout=timeout, **kwargs)
            if not flush_ret:
                time.sleep(0.5)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def flush(self, collection_names: list, timeout=None, **kwargs):
        if collection_names in (None, []) or not isinstance(collection_names, list):
            raise ParamError("Collection name list can not be None or empty")

        for name in collection_names:
            check_pass_param(collection_name=name)

        request = Prepare.flush_param(collection_names)
        future = self._stub.Flush.future(request, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        def _check():
            for collection_name in collection_names:
                segment_ids = future.result().coll_segIDs[collection_name].data
                self._wait_for_flushed(segment_ids)

        if kwargs.get("_async", False):
            flush_future = FlushFuture(future)
            flush_future.add_callback(_check)

            user_cb = kwargs.get("_callback", None)
            if user_cb:
                flush_future.add_callback(user_cb)

            return flush_future

        _check()

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def drop_index(self, collection_name, field_name, index_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name, field_name=field_name)
        request = Prepare.drop_index_request(collection_name, field_name, index_name)
        future = self._stub.DropIndex.future(request, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.error_code != 0:
            raise BaseException(response.error_code, response.reason)

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def dummy(self, request_type, timeout=None, **kwargs):
        request = Prepare.dummy_request(request_type)
        future = self._stub.Dummy.future(request, wait_for_ready=True, timeout=timeout)
        return future.result()

    # TODO seems not in use
    @error_handler
    def fake_register_link(self, timeout=None):
        request = Prepare.register_link_request()
        future = self._stub.RegisterLink.future(request, wait_for_ready=True, timeout=timeout)
        return future.result().status

    # TODO seems not in use
    @error_handler
    def get(self, collection_name, ids, output_fields=None, partition_names=None, timeout=None):
        # TODO: some check
        request = Prepare.retrieve_request(collection_name, ids, output_fields, partition_names)
        future = self._stub.Retrieve.future(request, wait_for_ready=True, timeout=timeout)
        return future.result()

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def query(self, collection_name, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
        if output_fields is not None and not isinstance(output_fields, (list,)):
            raise ParamError("Invalid query format. 'output_fields' must be a list")

        collection_schema = self.describe_collection(collection_name, timeout)
        collection_id = collection_schema["collection_id"]
        consistency_level = collection_schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = get_consistency_level(kwargs.get("consistency_level", consistency_level))

        ts_utils.construct_guarantee_ts(consistency_level, collection_id, kwargs)
        guarantee_timestamp = kwargs.get("guarantee_timestamp", 0)
        travel_timestamp = kwargs.get("travel_timestamp", 0)

        request = Prepare.query_request(collection_name, expr, output_fields, partition_names, guarantee_timestamp,
                                        travel_timestamp)

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
                elif field_data.type == DataType.BINARY_VECTOR:
                    dim = field_data.vectors.dim
                    start_pos = index * (int(dim / 8))
                    end_pos = (index + 1) * (int(dim / 8))
                    result[field_data.field_name] = field_data.vectors.binary_vector[start_pos:end_pos]
            results.append(result)

        return results

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def calc_distance(self, vectors_left, vectors_right, params, timeout=30, **kwargs):
        # both "metric" or "metric_type" are ok
        params = params or {"metric": config.CALC_DIST_METRIC}
        if "metric_type" in params.keys():
            params["metric"] = params["metric_type"]
            params.pop("metric_type")

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

            if is_l2(params["metric"]) and "sqrt" in params.keys() and params["sqrt"] is True:
                for i in range(len(response.float_dist.data)):
                    response.float_dist.data[i] = math.sqrt(response.float_dist.data[i])
            return response.float_dist.data
        raise BaseException(0, "Empty result returned")

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def load_balance(self, src_node_id, dst_node_ids, sealed_segment_ids, timeout=None, **kwargs):
        req = Prepare.load_balance_request(src_node_id, dst_node_ids, sealed_segment_ids)
        future = self._stub.LoadBalance.future(req, wait_for_ready=True, timeout=timeout)
        status = future.result()
        if status.error_code != 0:
            raise BaseException(status.error_code, status.reason)

    @error_handler
    def compact(self, collection_name, timeout=None, **kwargs) -> int:
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, wait_for_ready=True, timeout=timeout)
        response = rf.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        req = Prepare.manual_compaction(response.collectionID, 0)
        future = self._stub.ManualCompaction.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        return response.compactionID

    @error_handler
    def get_compaction_state(self, compaction_id, timeout=None, **kwargs) -> CompactionState:
        req = Prepare.get_compaction_state(compaction_id)

        future = self._stub.GetCompactionState.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        return CompactionState(
            compaction_id,
            State.new(response.state),
            response.executingPlanNo,
            response.timeoutPlanNo,
            response.completedPlanNo
        )

    @retry_on_rpc_failure(retry_times=10, wait=1)
    @error_handler
    def wait_for_compaction_completed(self, compaction_id, timeout=None, **kwargs):
        start = time.time()
        while True:
            time.sleep(0.5)
            compaction_state = self.get_compaction_state(compaction_id, timeout, **kwargs)
            if compaction_state.state == State.Completed:
                return True
            if compaction_state == State.UndefiedState:
                return False
            end = time.time()
            if timeout is not None:
                if end - start > timeout:
                    raise BaseException(1, "Get compaction state timeout")

    @error_handler
    def get_compaction_plans(self, compaction_id, timeout=None, **kwargs) -> CompactionPlans:
        req = Prepare.get_compaction_state_with_plans(compaction_id)

        future = self._stub.GetCompactionStateWithPlans.future(req, wait_for_ready=True, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise BaseException(response.status.error_code, response.status.reason)

        cp = CompactionPlans(compaction_id, response.state)

        cp.plans = [Plan(m.sources, m.target) for m in response.mergeInfos]

        return cp
