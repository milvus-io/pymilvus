import time
import json
import copy
import base64
from urllib import parse

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import milvus_pb2 as milvus_types
from ..grpc_gen import common_pb2

from .abstract import CollectionSchema, ChunkedQueryResult, MutationResult
from .check import (
    is_legal_host,
    is_legal_port,
    check_pass_param,
    check_index_params,
)
from .prepare import Prepare
from .types import (
    Status,
    IndexState,
    DataType,
    CompactionState,
    State,
    CompactionPlans,
    Plan,
    get_consistency_level,
    Replica, Shard, Group,
    GrantInfo, UserInfo, RoleInfo,
    BulkInsertState,
)

from .utils import (
    check_invalid_binary_vector,
    len_of
)

from ..settings import DefaultConfig as config
from .configs import DefaultConfigs
from . import ts_utils
from . import interceptor

from .asynch import (
    SearchFuture,
    MutationFuture,
    CreateIndexFuture,
    FlushFuture,
    LoadPartitionsFuture,
    ChunkedSearchFuture
)

from ..exceptions import (
    ExceptionsMessage,
    ParamError,
    DescribeCollectionException,
    MilvusException,
    AmbiguousIndexName,
)

from ..decorators import retry_on_rpc_failure


class GrpcHandler:
    def __init__(self, uri=config.GRPC_URI, host="", port="", channel=None, **kwargs):
        self._stub = None
        self._channel = channel

        addr = kwargs.get("address")
        self._address = addr if addr is not None else self.__get_address(uri, host, port)
        self._log_level = None
        self._request_id = None
        self._set_authorization(**kwargs)
        self._setup_grpc_channel()

    def __get_address(self, uri: str, host: str, port: str) -> str:
        if host != "" and port != "" and is_legal_host(host) and is_legal_port(port):
            return f"{host}:{port}"

        try:
            parsed_uri = parse.urlparse(uri)
        except (Exception) as e:
            raise ParamError(message=f"Illegal uri: [{uri}], {e}") from e
        return parsed_uri.netloc

    def _set_authorization(self, **kwargs):
        secure = kwargs.get("secure", False)
        if not isinstance(secure, bool):
            raise ParamError(message="secure must be bool type")
        self._secure = secure
        self._client_pem_path = kwargs.get("client_pem_path", "")
        self._client_key_path = kwargs.get("client_key_path", "")
        self._ca_pem_path = kwargs.get("ca_pem_path", "")
        self._server_pem_path = kwargs.get("server_pem_path", "")
        self._server_name = kwargs.get("server_name", "")

        self._authorization_interceptor = None
        self._setup_authorization_interceptor(kwargs.get("user", None), kwargs.get("password", None))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _wait_for_channel_ready(self, timeout=3):
        if self._channel is not None:
            try:
                grpc.channel_ready_future(self._channel).result(timeout=timeout)
                return
            except grpc.FutureTimeoutError as e:
                raise MilvusException(Status.CONNECT_FAILED,
                                      f'Fail connecting to server on {self._address}. Timeout') from e

        raise MilvusException(Status.CONNECT_FAILED, 'No channel in handler, please setup grpc channel first')

    def close(self):
        self._channel.close()

    def _setup_authorization_interceptor(self, user, password):
        if user and password:
            authorization = base64.b64encode(f"{user}:{password}".encode('utf-8'))
            key = "authorization"
            self._authorization_interceptor = interceptor.header_adder_interceptor(key, authorization)

    def _setup_grpc_channel(self):
        """ Create a ddl grpc channel """
        if self._channel is None:
            opts = [(cygrpc.ChannelArgKey.max_send_message_length, -1),
                    (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                    ('grpc.enable_retries', 1),
                    ('grpc.keepalive_time_ms', 55000),
                    ]
            if not self._secure:
                self._channel = grpc.insecure_channel(
                    self._address,
                    options=opts,
                )
            else:
                if self._client_pem_path != "" and self._client_key_path != "" and self._ca_pem_path != "" \
                        and self._server_name != "":
                    opts.append(('grpc.ssl_target_name_override', self._server_name, ),)
                    with open(self._client_pem_path, 'rb') as f:
                        certificate_chain = f.read()
                    with open(self._client_key_path, 'rb') as f:
                        private_key = f.read()
                    with open(self._ca_pem_path, 'rb') as f:
                        root_certificates = f.read()
                    creds = grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain)
                elif self._server_pem_path != "" and self._server_name != "":
                    opts.append(('grpc.ssl_target_name_override', self._server_name,), )
                    with open(self._server_pem_path, 'rb') as f:
                        server_pem = f.read()
                    creds = grpc.ssl_channel_credentials(root_certificates=server_pem)
                else:
                    creds = grpc.ssl_channel_credentials(root_certificates=None, private_key=None,
                                                         certificate_chain=None)
                self._channel = grpc.secure_channel(
                    self._address,
                    creds,
                    options=opts
                )
        # avoid to add duplicate headers.
        self._final_channel = self._channel
        if self._authorization_interceptor:
            self._final_channel = grpc.intercept_channel(self._final_channel, self._authorization_interceptor)
        if self._log_level:
            log_level_interceptor = interceptor.header_adder_interceptor("log_level", self._log_level)
            self._final_channel = grpc.intercept_channel(self._final_channel, log_level_interceptor)
            self._log_level = None
        if self._request_id:
            request_id_interceptor = interceptor.header_adder_interceptor("client_request_id", self._request_id)
            self._final_channel = grpc.intercept_channel(self._final_channel, request_id_interceptor)
            self._request_id = None
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

    def set_onetime_loglevel(self, log_level):
        self._log_level = log_level
        self._setup_grpc_channel()

    def set_onetime_request_id(self, req_id):
        self._request_id = req_id
        self._setup_grpc_channel()

    @property
    def server_address(self):
        """ Server network address """
        return self._address

    def reset_password(self, user, old_password, new_password):
        """
        reset password and then setup the grpc channel.
        """
        self.update_password(user, old_password, new_password)
        self._setup_authorization_interceptor(user, new_password)
        self._setup_grpc_channel()

    @retry_on_rpc_failure()
    def create_collection(self, collection_name, fields, shards_num=2, timeout=None, **kwargs):
        request = Prepare.create_collection_request(collection_name, fields, shards_num=shards_num, **kwargs)

        rf = self._stub.CreateCollection.future(request, timeout=timeout)
        if kwargs.get("_async", False):
            return rf
        status = rf.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure(retry_on_deadline=False)
    def drop_collection(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.drop_collection_request(collection_name)

        rf = self._stub.DropCollection.future(request, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def alter_collection(self, collection_name, properties, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name, properties=properties)
        request = Prepare.alter_collection_request(collection_name, properties)
        rf = self._stub.AlterCollection.future(request, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def has_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, timeout=timeout)

        reply = rf.result()
        if reply.status.error_code == common_pb2.Success:
            return True

        # TODO: Workaround for unreasonable describe collection results and error_code
        if reply.status.error_code == common_pb2.UnexpectedError and "can\'t find collection" in reply.status.reason:
            return False

        raise MilvusException(reply.status.error_code, reply.status.reason)

    @retry_on_rpc_failure()
    def describe_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, timeout=timeout)
        response = rf.result()
        status = response.status

        if status.error_code == 0:
            return CollectionSchema(raw=response).dict()

        raise DescribeCollectionException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def list_collections(self, timeout=None):
        request = Prepare.show_collections_request()
        rf = self._stub.ShowCollections.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if response.status.error_code == 0:
            return list(response.collection_names)

        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def create_partition(self, collection_name, partition_name, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.create_partition_request(collection_name, partition_name)
        rf = self._stub.CreatePartition.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def drop_partition(self, collection_name, partition_name, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.drop_partition_request(collection_name, partition_name)

        rf = self._stub.DropPartition.future(request, timeout=timeout)
        response = rf.result()

        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def has_partition(self, collection_name, partition_name, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.has_partition_request(collection_name, partition_name)
        rf = self._stub.HasPartition.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.value

        raise MilvusException(status.error_code, status.reason)

    # TODO: this is not inuse
    @retry_on_rpc_failure()
    def get_partition_info(self, collection_name, partition_name, timeout=None):
        request = Prepare.partition_stats_request(collection_name, partition_name)
        rf = self._stub.DescribePartition.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            statistics = response.statistics
            info_dict = {}
            for kv in statistics:
                info_dict[kv.key] = kv.value
            return info_dict
        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def list_partitions(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.show_partitions_request(collection_name)

        rf = self._stub.ShowPartitions.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return list(response.partition_names)

        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def get_partition_stats(self, collection_name, partition_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        req = Prepare.get_partition_stats_request(collection_name, partition_name)
        future = self._stub.GetPartitionStatistics.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise MilvusException(status.error_code, status.reason)

    def _prepare_batch_insert_request(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        insert_param = kwargs.get('insert_param', None)

        if insert_param and not isinstance(insert_param, milvus_types.RowBatch):
            raise ParamError(message="The value of key 'insert_param' is invalid")
        if not isinstance(entities, list):
            raise ParamError(message="None entities, please provide valid entities.")

        collection_schema = kwargs.get("schema", None)
        if not collection_schema:
            collection_schema = self.describe_collection(collection_name, timeout=timeout, **kwargs)

        fields_info = collection_schema["fields"]

        request = insert_param if insert_param \
            else Prepare.batch_insert_param(collection_name, entities, partition_name, fields_info)

        return request

    @retry_on_rpc_failure()
    def batch_insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        if not check_invalid_binary_vector(entities):
            raise ParamError(message="Invalid binary vector data exists")

        try:
            request = self._prepare_batch_insert_request(collection_name, entities, partition_name, timeout, **kwargs)
            rf = self._stub.Insert.future(request, timeout=timeout)
            if kwargs.get("_async", False) is True:
                cb = kwargs.get("_callback", None)
                f = MutationFuture(rf, cb, timeout=timeout, **kwargs)
                f.add_callback(ts_utils.update_ts_on_mutation(collection_name))
                return f

            response = rf.result()
            if response.status.error_code == 0:
                m = MutationResult(response)
                ts_utils.update_collection_ts(collection_name, m.timestamp)
                return m

            raise MilvusException(response.status.error_code, response.status.reason)
        except Exception as err:
            if kwargs.get("_async", False):
                return MutationFuture(None, None, err)
            raise err

    @retry_on_rpc_failure()
    def delete(self, collection_name, expression, partition_name=None, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        try:
            req = Prepare.delete_request(collection_name, partition_name, expression)
            future = self._stub.Delete.future(req, timeout=timeout)

            if kwargs.get("_async", False):
                cb = kwargs.get("_callback", None)
                f = MutationFuture(future, cb, timeout=timeout, **kwargs)
                f.add_callback(ts_utils.update_ts_on_mutation(collection_name))
                return f

            response = future.result()
            if response.status.error_code == 0:
                m = MutationResult(response)
                ts_utils.update_collection_ts(collection_name, m.timestamp)
                return m

            raise MilvusException(response.status.error_code, response.status.reason)
        except Exception as err:
            if kwargs.get("_async", False):
                return MutationFuture(None, None, err)
            raise err

    def _execute_search_requests(self, requests, timeout=None, **kwargs):
        auto_id = kwargs.get("auto_id", True)

        try:
            raws = []
            futures = []

            # step 1: get future object
            for request in requests:
                ft = self._stub.Search.future(request, timeout=timeout)
                futures.append(ft)

            if kwargs.get("_async", False):
                func = kwargs.get("_callback", None)
                return ChunkedSearchFuture(futures, func, auto_id)

            # step2: get results
            for ft in futures:
                response = ft.result()

                if response.status.error_code != 0:
                    raise MilvusException(response.status.error_code, response.status.reason)

                raws.append(response)
            round_decimal = kwargs.get("round_decimal", -1)
            return ChunkedQueryResult(raws, auto_id, round_decimal)

        except Exception as pre_err:
            if kwargs.get("_async", False):
                return SearchFuture(None, None, True, pre_err)
            raise pre_err

    @retry_on_rpc_failure(retry_on_deadline=False)
    def search(self, collection_name, data, anns_field, param, limit,
               expression=None, partition_names=None, output_fields=None,
               round_decimal=-1, timeout=None, **kwargs):
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

        collection_schema = kwargs.get("schema", None)
        if not collection_schema:
            collection_schema = self.describe_collection(collection_name, timeout=timeout, **kwargs)
        auto_id = collection_schema["auto_id"]
        consistency_level = collection_schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = get_consistency_level(_kwargs.get("consistency_level", consistency_level))
        _kwargs["schema"] = collection_schema

        ts_utils.construct_guarantee_ts(consistency_level, collection_name, _kwargs)

        requests = Prepare.search_requests_with_expr(collection_name, data, anns_field, param, limit, expression,
                                                     partition_names, output_fields, round_decimal, **_kwargs)
        _kwargs.pop("schema")
        _kwargs["auto_id"] = auto_id
        _kwargs["round_decimal"] = round_decimal

        return self._execute_search_requests(requests, timeout, **_kwargs)

    @retry_on_rpc_failure()
    def get_query_segment_info(self, collection_name, timeout=30, **kwargs):
        req = Prepare.get_query_segment_info_request(collection_name)
        future = self._stub.GetQuerySegmentInfo.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of QuerySegmentInfo
        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def create_alias(self, collection_name, alias, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.create_alias_request(collection_name, alias)
        rf = self._stub.CreateAlias.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def drop_alias(self, alias, timeout=None, **kwargs):
        request = Prepare.drop_alias_request(alias)
        rf = self._stub.DropAlias.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def alter_alias(self, collection_name, alias, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.alter_alias_request(collection_name, alias)
        rf = self._stub.AlterAlias.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        # for historical reason, index_name contained in kwargs.
        index_name = kwargs.pop("index_name", DefaultConfigs.IndexName)
        copy_kwargs = copy.deepcopy(kwargs)

        collection_desc = self.describe_collection(collection_name, timeout=timeout, **copy_kwargs)

        valid_field = False
        for fields in collection_desc["fields"]:
            if field_name != fields["name"]:
                continue
            valid_field = True
            if fields["type"] != DataType.FLOAT_VECTOR and fields["type"] != DataType.BINARY_VECTOR:
                break
            # check index params on vector field.
            check_index_params(params)

        if not valid_field:
            raise MilvusException(message=f"cannot create index on non-existed field: {field_name}")

        # sync flush
        _async = kwargs.get("_async", False)
        kwargs["_async"] = False

        index_param = Prepare.create_index_request(collection_name, field_name, params, index_name=index_name)
        future = self._stub.CreateIndex.future(index_param, timeout=timeout)

        if _async:
            def _check():
                if kwargs.get("sync", True):
                    index_success, fail_reason = self.wait_for_creating_index(collection_name=collection_name,
                                                                              index_name=index_name,
                                                                              timeout=timeout, field_name=field_name)
                    if not index_success:
                        raise MilvusException(message=fail_reason)

            index_future = CreateIndexFuture(future)
            index_future.add_callback(_check)
            user_cb = kwargs.get("_callback", None)
            if user_cb:
                index_future.add_callback(user_cb)
            return index_future

        status = future.result()

        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

        if kwargs.get("sync", True):
            index_success, fail_reason = self.wait_for_creating_index(collection_name=collection_name,
                                                                      index_name=index_name,
                                                                      timeout=timeout, field_name=field_name)
            if not index_success:
                raise MilvusException(message=fail_reason)

        return Status(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def list_indexes(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_index_request(collection_name, "")

        rf = self._stub.DescribeIndex.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return response.index_descriptions
        if status.error_code == Status.INDEX_NOT_EXIST:
            return []
        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def describe_index(self, collection_name, index_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_index_request(collection_name, index_name)

        rf = self._stub.DescribeIndex.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == Status.INDEX_NOT_EXIST:
            return None
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)
        if len(response.index_descriptions) == 1:
            info_dict = {kv.key: kv.value for kv in response.index_descriptions[0].params}
            info_dict['field_name'] = response.index_descriptions[0].field_name
            info_dict['index_name'] = response.index_descriptions[0].index_name
            if info_dict.get("params", None):
                info_dict["params"] = json.loads(info_dict["params"])
            return info_dict

        raise AmbiguousIndexName(message=ExceptionsMessage.AmbiguousIndexName)

    @retry_on_rpc_failure()
    def get_index_build_progress(self, collection_name, index_name, timeout=None):
        request = Prepare.describe_index_request(collection_name, index_name)
        rf = self._stub.DescribeIndex.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            if len(response.index_descriptions) == 1:
                index_desc = response.index_descriptions[0]
                return {'total_rows': index_desc.total_rows, 'indexed_rows': index_desc.indexed_rows}
            raise AmbiguousIndexName(message=ExceptionsMessage.AmbiguousIndexName)
        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def get_index_state(self, collection_name: str, index_name: str, timeout=None, **kwargs):
        request = Prepare.describe_index_request(collection_name, index_name)
        rf = self._stub.DescribeIndex.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

        if len(response.index_descriptions) == 1:
            index_desc = response.index_descriptions[0]
            return index_desc.state, index_desc.index_state_fail_reason
        # just for create_index.
        field_name = kwargs.pop("field_name", "")
        if field_name != "":
            for index_desc in response.index_descriptions:
                if index_desc.field_name == field_name:
                    return index_desc.state, index_desc.index_state_fail_reason

        raise AmbiguousIndexName(message=ExceptionsMessage.AmbiguousIndexName)

    @retry_on_rpc_failure()
    def wait_for_creating_index(self, collection_name, index_name, timeout=None, **kwargs):
        start = time.time()
        while True:
            time.sleep(0.5)
            state, fail_reason = self.get_index_state(collection_name, index_name, timeout=timeout, **kwargs)
            if state == IndexState.Finished:
                return True, fail_reason
            if state == IndexState.Failed:
                return False, fail_reason
            end = time.time()
            if isinstance(timeout, int) and end - start > timeout:
                raise MilvusException(message=f"collection {collection_name} create index {index_name} timeout in {timeout}s")

    @retry_on_rpc_failure()
    def load_collection(self, collection_name, replica_number=1, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.load_collection("", collection_name, replica_number)
        rf = self._stub.LoadCollection.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)
        _async = kwargs.get("_async", False)
        if not _async:
            self.wait_for_loading_collection(collection_name, timeout)

    @retry_on_rpc_failure()
    def load_collection_progress(self, collection_name, timeout=None):
        """ Return loading progress of collection """
        progress = self.get_loading_progress(collection_name, timeout=timeout)
        return {
            "loading_progress": f"{progress:.0f}%",
        }

    @retry_on_rpc_failure()
    def wait_for_loading_collection(self, collection_name, timeout=None):
        start = time.time()

        def can_loop(t) -> bool:
            return True if timeout is None else t <= (start + timeout)

        while can_loop(time.time()):
            progress = self.get_loading_progress(collection_name, timeout=timeout)
            if progress >= 100:
                return
            time.sleep(DefaultConfigs.WaitTimeDurationWhenLoad)
        raise MilvusException(message=f"wait for loading collection timeout, collection: {collection_name}")

    @retry_on_rpc_failure()
    def release_collection(self, collection_name, timeout=None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.release_collection("", collection_name)
        rf = self._stub.ReleaseCollection.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def load_partitions(self, collection_name, partition_names, replica_number=1, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        request = Prepare.load_partitions("", collection_name, partition_names, replica_number)
        future = self._stub.LoadPartitions.future(request, timeout=timeout)

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
            raise MilvusException(response.error_code, response.reason)
        sync = kwargs.get("sync", True)
        if sync:
            self.wait_for_loading_partitions(collection_name, partition_names)

    @retry_on_rpc_failure()
    def wait_for_loading_partitions(self, collection_name, partition_names, timeout=None):
        start = time.time()

        def can_loop(t) -> bool:
            return True if timeout is None else t <= (start + timeout)

        while can_loop(time.time()):
            progress = self.get_loading_progress(collection_name, partition_names, timeout=timeout)
            if progress >= 100:
                return
            time.sleep(DefaultConfigs.WaitTimeDurationWhenLoad)
        raise MilvusException(message=f"wait for loading partition timeout, collection: {collection_name}, partitions: {partition_names}")

    @retry_on_rpc_failure()
    def get_loading_progress(self, collection_name, partition_names=None, timeout=None):
        request = Prepare.get_loading_progress(collection_name, partition_names)
        response = self._stub.GetLoadingProgress.future(request, timeout=timeout).result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        return response.progress

    @retry_on_rpc_failure()
    def load_partitions_progress(self, collection_name, partition_names, timeout=None):
        """ Return loading progress of partitions """
        progress = self.get_loading_progress(collection_name, partition_names, timeout)
        return {
            "loading_progress": f"{progress:.0f}%",
        }

    @retry_on_rpc_failure()
    def release_partitions(self, collection_name, partition_names, timeout=None):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        request = Prepare.release_partitions("", collection_name, partition_names)
        rf = self._stub.ReleasePartitions.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        index_param = Prepare.get_collection_stats_request(collection_name)
        future = self._stub.GetCollectionStatistics.future(index_param, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def get_flush_state(self, segment_ids, timeout=None, **kwargs):
        req = Prepare.get_flush_state_request(segment_ids)
        future = self._stub.GetFlushState.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.flushed  # todo: A wrapper class of PersistentSegmentInfo
        raise MilvusException(status.error_code, status.reason)

    # TODO seem not in use
    @retry_on_rpc_failure()
    def get_persistent_segment_infos(self, collection_name, timeout=None, **kwargs):
        req = Prepare.get_persistent_segment_info_request(collection_name)
        future = self._stub.GetPersistentSegmentInfo.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of PersistentSegmentInfo
        raise MilvusException(status.error_code, status.reason)

    def _wait_for_flushed(self, segment_ids, timeout=None, **kwargs):
        flush_ret = False
        start = time.time()
        while not flush_ret:
            flush_ret = self.get_flush_state(segment_ids, timeout, **kwargs)
            end = time.time()
            if timeout is not None:
                if end - start > timeout:
                    raise MilvusException(message=f"wait for flush timeout, segment ids: {segment_ids}")

            if not flush_ret:
                time.sleep(0.5)

    @retry_on_rpc_failure()
    def flush(self, collection_names: list, timeout=None, **kwargs):
        if collection_names in (None, []) or not isinstance(collection_names, list):
            raise ParamError(message="Collection name list can not be None or empty")

        for name in collection_names:
            check_pass_param(collection_name=name)

        request = Prepare.flush_param(collection_names)
        future = self._stub.Flush.future(request, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

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

    @retry_on_rpc_failure()
    def drop_index(self, collection_name, field_name, index_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name, field_name=field_name)
        request = Prepare.drop_index_request(collection_name, field_name, index_name)
        future = self._stub.DropIndex.future(request, timeout=timeout)
        response = future.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def dummy(self, request_type, timeout=None, **kwargs):
        request = Prepare.dummy_request(request_type)
        future = self._stub.Dummy.future(request, timeout=timeout)
        return future.result()

    # TODO seems not in use
    @retry_on_rpc_failure()
    def fake_register_link(self, timeout=None):
        request = Prepare.register_link_request()
        future = self._stub.RegisterLink.future(request, timeout=timeout)
        return future.result().status

    # TODO seems not in use
    @retry_on_rpc_failure()
    def get(self, collection_name, ids, output_fields=None, partition_names=None, timeout=None):
        # TODO: some check
        request = Prepare.retrieve_request(collection_name, ids, output_fields, partition_names)
        future = self._stub.Retrieve.future(request, timeout=timeout)
        return future.result()

    @retry_on_rpc_failure()
    def query(self, collection_name, expr, output_fields=None, partition_names=None, timeout=None, **kwargs):
        if output_fields is not None and not isinstance(output_fields, (list,)):
            raise ParamError(message="Invalid query format. 'output_fields' must be a list")
        collection_schema = kwargs.get("schema", None)
        if not collection_schema:
            collection_schema = self.describe_collection(collection_name, timeout)
        consistency_level = collection_schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = get_consistency_level(kwargs.get("consistency_level", consistency_level))

        ts_utils.construct_guarantee_ts(consistency_level, collection_name, kwargs)
        request = Prepare.query_request(collection_name, expr, output_fields, partition_names, **kwargs)

        future = self._stub.Query.future(request, timeout=timeout)
        response = future.result()
        if response.status.error_code == Status.EMPTY_COLLECTION:
            return []
        if response.status.error_code != Status.SUCCESS:
            raise MilvusException(response.status.error_code, response.status.reason)

        num_fields = len(response.fields_data)
        # check has fields
        if num_fields == 0:
            raise MilvusException(message="No fields returned")

        # check if all lists are of the same length
        it = iter(response.fields_data)
        num_entities = len_of(next(it))
        if not all(len_of(field_data) == num_entities for field_data in it):
            raise MilvusException(message="The length of fields data is inconsistent")

        # transpose
        results = []
        for index in range(0, num_entities):
            result = {}
            for field_data in response.fields_data:
                if field_data.type == DataType.BOOL:
                    result[field_data.field_name] = field_data.scalars.bool_data.data[index]
                elif field_data.type in (DataType.INT8, DataType.INT16, DataType.INT32):
                    result[field_data.field_name] = field_data.scalars.int_data.data[index]
                elif field_data.type == DataType.INT64:
                    result[field_data.field_name] = field_data.scalars.long_data.data[index]
                elif field_data.type == DataType.FLOAT:
                    result[field_data.field_name] = round(field_data.scalars.float_data.data[index], 6)
                elif field_data.type == DataType.DOUBLE:
                    result[field_data.field_name] = field_data.scalars.double_data.data[index]
                elif field_data.type == DataType.VARCHAR:
                    result[field_data.field_name] = field_data.scalars.string_data.data[index]
                elif field_data.type == DataType.STRING:
                    raise MilvusException(message="Not support string yet")
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

    @retry_on_rpc_failure()
    def load_balance(self, collection_name: str, src_node_id, dst_node_ids, sealed_segment_ids, timeout=None, **kwargs):
        req = Prepare.load_balance_request(collection_name, src_node_id, dst_node_ids, sealed_segment_ids)
        future = self._stub.LoadBalance.future(req, timeout=timeout)
        status = future.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def compact(self, collection_name, timeout=None, **kwargs) -> int:
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, timeout=timeout)
        response = rf.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        req = Prepare.manual_compaction(response.collectionID, 0)
        future = self._stub.ManualCompaction.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        return response.compactionID

    @retry_on_rpc_failure()
    def get_compaction_state(self, compaction_id, timeout=None, **kwargs) -> CompactionState:
        req = Prepare.get_compaction_state(compaction_id)

        future = self._stub.GetCompactionState.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        return CompactionState(
            compaction_id,
            State.new(response.state),
            response.executingPlanNo,
            response.timeoutPlanNo,
            response.completedPlanNo
        )

    @retry_on_rpc_failure()
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
                    raise MilvusException(message=f"get compaction state timeout, compaction id: {compaction_id}")

    @retry_on_rpc_failure()
    def get_compaction_plans(self, compaction_id, timeout=None, **kwargs) -> CompactionPlans:
        req = Prepare.get_compaction_state_with_plans(compaction_id)

        future = self._stub.GetCompactionStateWithPlans.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        cp = CompactionPlans(compaction_id, response.state)

        cp.plans = [Plan(m.sources, m.target) for m in response.mergeInfos]

        return cp

    @retry_on_rpc_failure()
    def get_replicas(self, collection_name, timeout=None, **kwargs) -> Replica:
        collection_id = self.describe_collection(collection_name, timeout, **kwargs)["collection_id"]

        req = Prepare.get_replicas(collection_id)
        future = self._stub.GetReplicas.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        groups = []
        for replica in response.replicas:
            shards = [Shard(s.dm_channel_name, s.node_ids, s.leaderID) for s in replica.shard_replicas]
            groups.append(Group(replica.replicaID, shards, replica.node_ids))

        return Replica(groups)

    @retry_on_rpc_failure()
    def do_bulk_insert(self, collection_name, partition_name, files: list, timeout=None, **kwargs) -> int:
        req = Prepare.do_bulk_insert(collection_name, partition_name, files, **kwargs)
        future = self._stub.Import.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        if len(response.tasks) == 0:
            raise MilvusException(common_pb2.UNEXPECTED_ERROR, "no task id returned from server")
        return response.tasks[0]

    @retry_on_rpc_failure()
    def get_bulk_insert_state(self, task_id, timeout=None, **kwargs) -> BulkInsertState:
        req = Prepare.get_bulk_insert_state(task_id)
        future = self._stub.GetImportState.future(req, timeout=timeout)
        resp = future.result()
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        state = BulkInsertState(task_id, resp.state, resp.row_count, resp.id_list, resp.infos, resp.create_ts)
        return state

    @retry_on_rpc_failure()
    def list_bulk_insert_tasks(self, limit, collection_name, timeout=None, **kwargs) -> list:
        req = Prepare.list_bulk_insert_tasks(limit, collection_name)
        future = self._stub.ListImportTasks.future(req, timeout=timeout)
        resp = future.result()
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        tasks = [BulkInsertState(t.id, t.state, t.row_count, t.id_list, t.infos, t.create_ts)
                 for t in resp.tasks]
        return tasks

    @retry_on_rpc_failure()
    def create_user(self, user, password, timeout=None, **kwargs):
        check_pass_param(user=user, password=password)
        req = Prepare.create_user_request(user, password)
        resp = self._stub.CreateCredential(req, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def update_password(self, user, old_password, new_password, timeout=None, **kwargs):
        req = Prepare.update_password_request(user, old_password, new_password)
        resp = self._stub.UpdateCredential(req, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def delete_user(self, user, timeout=None, **kwargs):
        req = Prepare.delete_user_request(user)
        resp = self._stub.DeleteCredential(req, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def list_usernames(self, timeout=None, **kwargs):
        req = Prepare.list_usernames_request()
        resp = self._stub.ListCredUsers(req, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return resp.usernames

    @retry_on_rpc_failure()
    def create_role(self, role_name, timeout=None, **kwargs):
        req = Prepare.create_role_request(role_name)
        resp = self._stub.CreateRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def drop_role(self, role_name, timeout=None, **kwargs):
        req = Prepare.drop_role_request(role_name)
        resp = self._stub.DropRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def add_user_to_role(self, username, role_name, timeout=None, **kwargs):
        req = Prepare.operate_user_role_request(username, role_name, milvus_types.OperateUserRoleType.AddUserToRole)
        resp = self._stub.OperateUserRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def remove_user_from_role(self, username, role_name, timeout=None, **kwargs):
        req = Prepare.operate_user_role_request(username, role_name,
                                                milvus_types.OperateUserRoleType.RemoveUserFromRole)
        resp = self._stub.OperateUserRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def select_one_role(self, role_name, include_user_info, timeout=None, **kwargs):
        req = Prepare.select_role_request(role_name, include_user_info)
        resp = self._stub.SelectRole(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return RoleInfo(resp.results)

    @retry_on_rpc_failure()
    def select_all_role(self, include_user_info, timeout=None, **kwargs):
        req = Prepare.select_role_request(None, include_user_info)
        resp = self._stub.SelectRole(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return RoleInfo(resp.results)

    @retry_on_rpc_failure()
    def select_one_user(self, username, include_role_info, timeout=None, **kwargs):
        req = Prepare.select_user_request(username, include_role_info)
        resp = self._stub.SelectUser(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return UserInfo(resp.results)

    @retry_on_rpc_failure()
    def select_all_user(self, include_role_info, timeout=None, **kwargs):
        req = Prepare.select_user_request(None, include_role_info)
        resp = self._stub.SelectUser(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return UserInfo(resp.results)

    @retry_on_rpc_failure()
    def grant_privilege(self, role_name, object, object_name, privilege, timeout=None, **kwargs):
        req = Prepare.operate_privilege_request(role_name, object, object_name, privilege,
                                                milvus_types.OperatePrivilegeType.Grant)
        resp = self._stub.OperatePrivilege(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def revoke_privilege(self, role_name, object, object_name, privilege, timeout=None, **kwargs):
        req = Prepare.operate_privilege_request(role_name, object, object_name, privilege,
                                                milvus_types.OperatePrivilegeType.Revoke)
        resp = self._stub.OperatePrivilege(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def select_grant_for_one_role(self, role_name, timeout=None, **kwargs):
        req = Prepare.select_grant_request(role_name, None, None)
        resp = self._stub.SelectGrant(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return GrantInfo(resp.entities)

    @retry_on_rpc_failure()
    def select_grant_for_role_and_object(self, role_name, object, object_name, timeout=None, **kwargs):
        req = Prepare.select_grant_request(role_name, object, object_name)
        resp = self._stub.SelectGrant(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return GrantInfo(resp.entities)
