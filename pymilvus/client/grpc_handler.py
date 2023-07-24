import base64
import copy
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib import parse

import grpc
from grpc._cython import cygrpc

from pymilvus.decorators import ignore_unimplemented, retry_on_rpc_failure, upgrade_reminder
from pymilvus.exceptions import (
    AmbiguousIndexName,
    DescribeCollectionException,
    ExceptionsMessage,
    MilvusException,
    ParamError,
)
from pymilvus.grpc_gen import common_pb2, milvus_pb2_grpc
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.settings import Config

from . import entity_helper, interceptor, ts_utils
from .abstract import ChunkedQueryResult, CollectionSchema, MutationResult
from .asynch import (
    ChunkedSearchFuture,
    CreateIndexFuture,
    FlushFuture,
    LoadPartitionsFuture,
    MutationFuture,
    SearchFuture,
)
from .check import (
    check_pass_param,
    is_legal_host,
    is_legal_port,
)
from .prepare import Prepare
from .types import (
    BulkInsertState,
    CompactionPlans,
    CompactionState,
    DataType,
    GrantInfo,
    Group,
    IndexState,
    LoadState,
    Plan,
    Replica,
    ResourceGroupInfo,
    RoleInfo,
    Shard,
    State,
    Status,
    UserInfo,
)
from .utils import (
    check_invalid_binary_vector,
    get_server_type,
    len_of,
)


class GrpcHandler:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        uri: str = Config.GRPC_URI,
        host: str = "",
        port: str = "",
        channel: Optional[grpc.Channel] = None,
        **kwargs,
    ) -> None:
        self._stub = None
        self._channel = channel

        addr = kwargs.get("address")
        self._address = addr if addr is not None else self.__get_address(uri, host, port)
        self._log_level = None
        self._request_id = None
        self._user = kwargs.get("user", None)
        self._set_authorization(**kwargs)
        self._setup_db_interceptor(kwargs.get("db_name", None))
        self._setup_grpc_channel()

    def __get_address(self, uri: str, host: str, port: str) -> str:
        if host != "" and port != "" and is_legal_host(host) and is_legal_port(port):
            return f"{host}:{port}"

        try:
            parsed_uri = parse.urlparse(uri)
        except Exception as e:
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
        self._setup_authorization_interceptor(
            kwargs.get("user", None),
            kwargs.get("password", None),
            kwargs.get("token", None),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        pass

    def _wait_for_channel_ready(self, timeout: Union[int, float] = 10):
        if self._channel is None:
            raise MilvusException(
                code=Status.CONNECT_FAILED,
                message="No channel in handler, please setup grpc channel first",
            )

        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
            self._setup_identifier_interceptor(self._user)
        except (grpc.FutureTimeoutError, MilvusException) as e:
            raise MilvusException(
                code=Status.CONNECT_FAILED,
                message=f"Fail connecting to server on {self._address}. Timeout",
            ) from e

    def close(self):
        self._channel.close()

    def reset_db_name(self, db_name: str):
        self._setup_db_interceptor(db_name)
        self._setup_grpc_channel()
        self._setup_identifier_interceptor(self._user)

    def _setup_authorization_interceptor(self, user: str, password: str, token: str):
        keys = []
        values = []
        if token:
            authorization = base64.b64encode(f"{token}".encode())
            keys.append("authorization")
            values.append(authorization)
        elif user and password:
            authorization = base64.b64encode(f"{user}:{password}".encode())
            keys.append("authorization")
            values.append(authorization)
        if len(keys) > 0 and len(values) > 0:
            self._authorization_interceptor = interceptor.header_adder_interceptor(keys, values)

    def _setup_db_interceptor(self, db_name: str):
        if db_name is None:
            self._db_interceptor = None
        else:
            check_pass_param(db_name=db_name)
            self._db_interceptor = interceptor.header_adder_interceptor(["dbname"], [db_name])

    def _setup_grpc_channel(self):
        """Create a ddl grpc channel"""
        if self._channel is None:
            opts = [
                (cygrpc.ChannelArgKey.max_send_message_length, -1),
                (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_time_ms", 55000),
            ]
            if not self._secure:
                self._channel = grpc.insecure_channel(
                    self._address,
                    options=opts,
                )
            else:
                if (
                    self._client_pem_path != ""
                    and self._client_key_path != ""
                    and self._ca_pem_path != ""
                    and self._server_name != ""
                ):
                    opts.append(("grpc.ssl_target_name_override", self._server_name))
                    with Path(self._client_pem_path).open("rb") as f:
                        certificate_chain = f.read()
                    with Path(self._client_key_path).open("rb") as f:
                        private_key = f.read()
                    with Path(self._ca_pem_path).open("rb") as f:
                        root_certificates = f.read()
                    creds = grpc.ssl_channel_credentials(
                        root_certificates, private_key, certificate_chain
                    )
                elif self._server_pem_path != "" and self._server_name != "":
                    opts.append(("grpc.ssl_target_name_override", self._server_name))
                    with Path(self._server_pem_path).open("rb") as f:
                        server_pem = f.read()
                    creds = grpc.ssl_channel_credentials(root_certificates=server_pem)
                else:
                    creds = grpc.ssl_channel_credentials(
                        root_certificates=None, private_key=None, certificate_chain=None
                    )
                self._channel = grpc.secure_channel(
                    self._address,
                    creds,
                    options=opts,
                )
        # avoid to add duplicate headers.
        self._final_channel = self._channel
        if self._authorization_interceptor:
            self._final_channel = grpc.intercept_channel(
                self._final_channel, self._authorization_interceptor
            )
        if self._db_interceptor:
            self._final_channel = grpc.intercept_channel(self._final_channel, self._db_interceptor)
        if self._log_level:
            log_level_interceptor = interceptor.header_adder_interceptor(
                ["log_level"], [self._log_level]
            )
            self._final_channel = grpc.intercept_channel(self._final_channel, log_level_interceptor)
            self._log_level = None
        if self._request_id:
            request_id_interceptor = interceptor.header_adder_interceptor(
                ["client_request_id"], [self._request_id]
            )
            self._final_channel = grpc.intercept_channel(
                self._final_channel, request_id_interceptor
            )
            self._request_id = None
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

    def set_onetime_loglevel(self, log_level: str):
        self._log_level = log_level
        self._setup_grpc_channel()

    def set_onetime_request_id(self, req_id: int):
        self._request_id = req_id
        self._setup_grpc_channel()

    def _setup_identifier_interceptor(self, user: str):
        host = socket.gethostname()
        self._identifier = self.__internal_register(user, host)
        self._identifier_interceptor = interceptor.header_adder_interceptor(
            ["identifier"], [str(self._identifier)]
        )
        self._final_channel = grpc.intercept_channel(
            self._final_channel, self._identifier_interceptor
        )
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

    @property
    def server_address(self):
        """Server network address"""
        return self._address

    def get_server_type(self):
        return get_server_type(self.server_address.split(":")[0])

    def reset_password(
        self,
        user: str,
        old_password: str,
        new_password: str,
        timeout: Optional[float] = None,
    ):
        """
        reset password and then setup the grpc channel.
        """
        self.update_password(user, old_password, new_password, timeout=timeout)
        self._setup_authorization_interceptor(user, new_password, None)
        self._setup_grpc_channel()

    @retry_on_rpc_failure()
    def create_collection(
        self, collection_name: str, fields: List, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name)
        request = Prepare.create_collection_request(collection_name, fields, **kwargs)

        rf = self._stub.CreateCollection.future(request, timeout=timeout)
        if kwargs.get("_async", False):
            return rf
        status = rf.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)
        return None

    @retry_on_rpc_failure()
    def drop_collection(self, collection_name: str, timeout: Optional[float] = None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.drop_collection_request(collection_name)

        rf = self._stub.DropCollection.future(request, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def alter_collection(
        self, collection_name: str, properties: List, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, properties=properties)
        request = Prepare.alter_collection_request(collection_name, properties)
        rf = self._stub.AlterCollection.future(request, timeout=timeout)
        status = rf.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def has_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, timeout=timeout)

        reply = rf.result()
        if reply.status.error_code == common_pb2.Success:
            return True

        # TODO: Workaround for unreasonable describe collection results and error_code
        if (
            reply.status.error_code == common_pb2.UnexpectedError
            and "can't find collection" in reply.status.reason
        ):
            return False

        raise MilvusException(reply.status.error_code, reply.status.reason)

    @retry_on_rpc_failure()
    def describe_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        rf = self._stub.DescribeCollection.future(request, timeout=timeout)
        response = rf.result()
        status = response.status

        if status.error_code == 0:
            return CollectionSchema(raw=response).dict()

        raise DescribeCollectionException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def list_collections(self, timeout: Optional[float] = None):
        request = Prepare.show_collections_request()
        rf = self._stub.ShowCollections.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if response.status.error_code == 0:
            return list(response.collection_names)

        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def rename_collections(
        self,
        old_name: str,
        new_name: str,
        new_db_name: str = "default",
        timeout: Optional[float] = None,
    ):
        check_pass_param(collection_name=new_name)
        check_pass_param(collection_name=old_name)
        if new_db_name:
            check_pass_param(db_name=new_db_name)
        request = Prepare.rename_collections_request(old_name, new_name, new_db_name)
        rf = self._stub.RenameCollection.future(request, timeout=timeout)
        response = rf.result()

        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def create_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None
    ):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.create_partition_request(collection_name, partition_name)
        rf = self._stub.CreatePartition.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def drop_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None
    ):
        check_pass_param(collection_name=collection_name, partition_name=partition_name)
        request = Prepare.drop_partition_request(collection_name, partition_name)

        rf = self._stub.DropPartition.future(request, timeout=timeout)
        response = rf.result()

        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def has_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ):
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
    def get_partition_info(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None
    ):
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
    def list_partitions(self, collection_name: str, timeout: Optional[float] = None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.show_partitions_request(collection_name)

        rf = self._stub.ShowPartitions.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            return list(response.partition_names)

        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def get_partition_stats(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name)
        req = Prepare.get_partition_stats_request(collection_name, partition_name)
        future = self._stub.GetPartitionStatistics.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise MilvusException(status.error_code, status.reason)

    def _get_info(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        schema = kwargs.get("schema", None)
        if not schema:
            schema = self.describe_collection(collection_name, timeout=timeout)

        fields_info = schema.get("fields")
        enable_dynamic = schema.get("enable_dynamic_field", False)

        return fields_info, enable_dynamic

    def _prepare_row_insert_request(
        self,
        collection_name: str,
        entity_rows: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if not isinstance(entity_rows, list):
            raise ParamError(message="None rows, please provide valid row data.")

        fields_info, enable_dynamic = self._get_info(collection_name, timeout, **kwargs)
        return Prepare.row_insert_param(
            collection_name,
            entity_rows,
            partition_name,
            fields_info,
            enable_dynamic=enable_dynamic,
        )

    @retry_on_rpc_failure()
    def insert_rows(
        self,
        collection_name: str,
        entities: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(entities, dict):
            entities = [entities]
        request = self._prepare_row_insert_request(
            collection_name, entities, partition_name, timeout, **kwargs
        )
        rf = self._stub.Insert.future(request, timeout=timeout)
        response = rf.result()
        if response.status.error_code == 0:
            m = MutationResult(response)
            ts_utils.update_collection_ts(collection_name, m.timestamp)
            return m

        raise MilvusException(response.status.error_code, response.status.reason)

    def _prepare_batch_insert_request(
        self,
        collection_name: str,
        entities: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        param = kwargs.get("insert_param")
        if param and not isinstance(param, milvus_types.InsertRequest):
            raise ParamError(message="The value of key 'insert_param' is invalid")
        if not isinstance(entities, list):
            raise ParamError(message="None entities, please provide valid entities.")

        schema = kwargs.get("schema")
        if not schema:
            schema = self.describe_collection(collection_name, timeout=timeout, **kwargs)

        fields_info = schema["fields"]

        return (
            param
            if param
            else Prepare.batch_insert_param(collection_name, entities, partition_name, fields_info)
        )

    @retry_on_rpc_failure()
    def batch_insert(
        self,
        collection_name: str,
        entities: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if not check_invalid_binary_vector(entities):
            raise ParamError(message="Invalid binary vector data exists")

        try:
            request = self._prepare_batch_insert_request(
                collection_name, entities, partition_name, timeout, **kwargs
            )
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
            raise err from err

    @retry_on_rpc_failure()
    def delete(
        self,
        collection_name: str,
        expression: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
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
            raise err from err

    def _execute_search_requests(self, requests: Any, timeout: Optional[float] = None, **kwargs):
        try:
            if kwargs.get("_async", False):
                futures = []
                for request in requests:
                    ft = self._stub.Search.future(request, timeout=timeout)
                    futures.append(ft)
                func = kwargs.get("_callback", None)
                return ChunkedSearchFuture(futures, func)

            raws = []
            for request in requests:
                response = self._stub.Search(request, timeout=timeout)

                if response.status.error_code != 0:
                    raise MilvusException(response.status.error_code, response.status.reason)

                raws.append(response)
            round_decimal = kwargs.get("round_decimal", -1)
            return ChunkedQueryResult(raws, round_decimal)

        except Exception as pre_err:
            if kwargs.get("_async", False):
                return SearchFuture(None, None, pre_err)
            raise pre_err from pre_err

    @retry_on_rpc_failure()
    def search(
        self,
        collection_name: str,
        data: List[List[float]],
        anns_field: str,
        param: Dict,
        limit: int,
        expression: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(
            limit=limit,
            round_decimal=round_decimal,
            anns_field=anns_field,
            search_data=data,
            partition_name_array=partition_names,
            output_fields=output_fields,
            travel_timestamp=kwargs.get("travel_timestamp", 0),
            guarantee_timestamp=kwargs.get("guarantee_timestamp", None),
        )

        requests = Prepare.search_requests_with_expr(
            collection_name,
            data,
            anns_field,
            param,
            limit,
            expression,
            partition_names,
            output_fields,
            round_decimal,
            **kwargs,
        )
        return self._execute_search_requests(
            requests, timeout, round_decimal=round_decimal, **kwargs
        )

    @retry_on_rpc_failure()
    def get_query_segment_info(self, collection_name: str, timeout: float = 30, **kwargs):
        req = Prepare.get_query_segment_info_request(collection_name)
        future = self._stub.GetQuerySegmentInfo.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of QuerySegmentInfo
        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def create_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name)
        request = Prepare.create_alias_request(collection_name, alias)
        rf = self._stub.CreateAlias.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def drop_alias(self, alias: str, timeout: Optional[float] = None, **kwargs):
        request = Prepare.drop_alias_request(alias)
        rf = self._stub.DropAlias.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def alter_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name)
        request = Prepare.alter_alias_request(collection_name, alias)
        rf = self._stub.AlterAlias.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def create_index(
        self,
        collection_name: str,
        field_name: str,
        params: Dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        # for historical reason, index_name contained in kwargs.
        index_name = kwargs.pop("index_name", Config.IndexName)
        copy_kwargs = copy.deepcopy(kwargs)

        collection_desc = self.describe_collection(collection_name, timeout=timeout, **copy_kwargs)

        valid_field = False
        for fields in collection_desc["fields"]:
            if field_name != fields["name"]:
                continue
            valid_field = True
            if fields["type"] not in {DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR}:
                break

        if not valid_field:
            raise MilvusException(message=f"cannot create index on non-existed field: {field_name}")

        # sync flush
        _async = kwargs.get("_async", False)
        kwargs["_async"] = False

        index_param = Prepare.create_index_request(
            collection_name, field_name, params, index_name=index_name
        )
        future = self._stub.CreateIndex.future(index_param, timeout=timeout)

        if _async:

            def _check():
                if kwargs.get("sync", True):
                    index_success, fail_reason = self.wait_for_creating_index(
                        collection_name=collection_name,
                        index_name=index_name,
                        timeout=timeout,
                        field_name=field_name,
                    )
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
            index_success, fail_reason = self.wait_for_creating_index(
                collection_name=collection_name,
                index_name=index_name,
                timeout=timeout,
                field_name=field_name,
            )
            if not index_success:
                raise MilvusException(message=fail_reason)

        return Status(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def list_indexes(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
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
    def describe_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        timestamp: Optional[int] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_index_request(collection_name, index_name, timestamp=timestamp)

        rf = self._stub.DescribeIndex.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == Status.INDEX_NOT_EXIST:
            return None
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)
        if len(response.index_descriptions) == 1:
            info_dict = {kv.key: kv.value for kv in response.index_descriptions[0].params}
            info_dict["field_name"] = response.index_descriptions[0].field_name
            info_dict["index_name"] = response.index_descriptions[0].index_name
            if info_dict.get("params", None):
                info_dict["params"] = json.loads(info_dict["params"])
            return info_dict

        raise AmbiguousIndexName(message=ExceptionsMessage.AmbiguousIndexName)

    @retry_on_rpc_failure()
    def get_index_build_progress(
        self, collection_name: str, index_name: str, timeout: Optional[float] = None
    ):
        request = Prepare.describe_index_request(collection_name, index_name)
        rf = self._stub.DescribeIndex.future(request, timeout=timeout)
        response = rf.result()
        status = response.status
        if status.error_code == 0:
            if len(response.index_descriptions) == 1:
                index_desc = response.index_descriptions[0]
                return {
                    "total_rows": index_desc.total_rows,
                    "indexed_rows": index_desc.indexed_rows,
                    "pending_index_rows": index_desc.pending_index_rows,
                }
            raise AmbiguousIndexName(message=ExceptionsMessage.AmbiguousIndexName)
        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def get_index_state(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        timestamp: Optional[int] = None,
        **kwargs,
    ):
        request = Prepare.describe_index_request(collection_name, index_name, timestamp)
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
    def wait_for_creating_index(
        self, collection_name: str, index_name: str, timeout: Optional[float] = None, **kwargs
    ):
        timestamp = self.alloc_timestamp()
        start = time.time()
        while True:
            time.sleep(0.5)
            state, fail_reason = self.get_index_state(
                collection_name, index_name, timeout=timeout, timestamp=timestamp, **kwargs
            )
            if state == IndexState.Finished:
                return True, fail_reason
            if state == IndexState.Failed:
                return False, fail_reason
            end = time.time()
            if isinstance(timeout, int) and end - start > timeout:
                msg = (
                    f"collection {collection_name} create index {index_name} "
                    f"timeout in {timeout}s"
                )
                raise MilvusException(message=msg)

    @retry_on_rpc_failure()
    def load_collection(
        self,
        collection_name: str,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name, replica_number=replica_number)
        _refresh = kwargs.get("_refresh", False)
        _resource_groups = kwargs.get("_resource_groups")
        request = Prepare.load_collection(
            "", collection_name, replica_number, _refresh, _resource_groups
        )
        rf = self._stub.LoadCollection.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)
        _async = kwargs.get("_async", False)
        if not _async:
            self.wait_for_loading_collection(collection_name, timeout)

    @retry_on_rpc_failure()
    def load_collection_progress(self, collection_name: str, timeout: Optional[float] = None):
        """Return loading progress of collection"""
        progress = self.get_loading_progress(collection_name, timeout=timeout)
        return {
            "loading_progress": f"{progress:.0f}%",
        }

    @retry_on_rpc_failure()
    def wait_for_loading_collection(self, collection_name: str, timeout: Optional[float] = None):
        start = time.time()

        def can_loop(t: int) -> bool:
            return True if timeout is None else t <= (start + timeout)

        while can_loop(time.time()):
            progress = self.get_loading_progress(collection_name, timeout=timeout)
            if progress >= 100:
                return
            time.sleep(Config.WaitTimeDurationWhenLoad)
        raise MilvusException(
            message=f"wait for loading collection timeout, collection: {collection_name}"
        )

    @retry_on_rpc_failure()
    def release_collection(self, collection_name: str, timeout: Optional[float] = None):
        check_pass_param(collection_name=collection_name)
        request = Prepare.release_collection("", collection_name)
        rf = self._stub.ReleaseCollection.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def load_partitions(
        self,
        collection_name: str,
        partition_names: List[str],
        replica_number: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(
            collection_name=collection_name,
            partition_name_array=partition_names,
            replica_number=replica_number,
        )
        _refresh = kwargs.get("_refresh", False)
        _resource_groups = kwargs.get("_resource_groups")
        request = Prepare.load_partitions(
            "", collection_name, partition_names, replica_number, _refresh, _resource_groups
        )
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
            return None
        return None

    @retry_on_rpc_failure()
    def wait_for_loading_partitions(
        self,
        collection_name: str,
        partition_names: List[str],
        timeout: Optional[float] = None,
    ):
        start = time.time()

        def can_loop(t: int) -> bool:
            return True if timeout is None else t <= (start + timeout)

        while can_loop(time.time()):
            progress = self.get_loading_progress(collection_name, partition_names, timeout=timeout)
            if progress >= 100:
                return
            time.sleep(Config.WaitTimeDurationWhenLoad)
        raise MilvusException(
            message=f"wait for loading partition timeout, collection: {collection_name}, partitions: {partition_names}"
        )

    @retry_on_rpc_failure()
    def get_loading_progress(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        request = Prepare.get_loading_progress(collection_name, partition_names)
        response = self._stub.GetLoadingProgress.future(request, timeout=timeout).result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        return response.progress

    @retry_on_rpc_failure()
    def create_database(self, db_name: str, timeout: Optional[float] = None):
        request = Prepare.create_database_req(db_name)
        status = self._stub.CreateDatabase(request, timeout=timeout)
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def drop_database(self, db_name: str, timeout: Optional[float] = None):
        request = Prepare.drop_database_req(db_name)
        status = self._stub.DropDatabase(request, timeout=timeout)
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def list_database(self, timeout: Optional[float] = None):
        request = Prepare.list_database_req()
        response = self._stub.ListDatabases(request, timeout=timeout)
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        return list(response.db_names)

    @retry_on_rpc_failure()
    def get_load_state(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        request = Prepare.get_load_state(collection_name, partition_names)
        response = self._stub.GetLoadState.future(request, timeout=timeout).result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        return LoadState(response.state)

    @retry_on_rpc_failure()
    def load_partitions_progress(
        self, collection_name: str, partition_names: List[str], timeout: Optional[float] = None
    ):
        """Return loading progress of partitions"""
        progress = self.get_loading_progress(collection_name, partition_names, timeout)
        return {
            "loading_progress": f"{progress:.0f}%",
        }

    @retry_on_rpc_failure()
    def release_partitions(
        self, collection_name: str, partition_names: List[str], timeout: Optional[float] = None
    ):
        check_pass_param(collection_name=collection_name, partition_name_array=partition_names)
        request = Prepare.release_partitions("", collection_name, partition_names)
        rf = self._stub.ReleasePartitions.future(request, timeout=timeout)
        response = rf.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def get_collection_stats(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        check_pass_param(collection_name=collection_name)
        index_param = Prepare.get_collection_stats_request(collection_name)
        future = self._stub.GetCollectionStatistics.future(index_param, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.stats

        raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def get_flush_state(self, segment_ids: List[int], timeout: Optional[float] = None, **kwargs):
        req = Prepare.get_flush_state_request(segment_ids)
        future = self._stub.GetFlushState.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.flushed  # todo: A wrapper class of PersistentSegmentInfo
        raise MilvusException(status.error_code, status.reason)

    # TODO seem not in use
    @retry_on_rpc_failure()
    def get_persistent_segment_infos(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.get_persistent_segment_info_request(collection_name)
        future = self._stub.GetPersistentSegmentInfo.future(req, timeout=timeout)
        response = future.result()
        status = response.status
        if status.error_code == 0:
            return response.infos  # todo: A wrapper class of PersistentSegmentInfo
        raise MilvusException(status.error_code, status.reason)

    def _wait_for_flushed(self, segment_ids: List[int], timeout: Optional[float] = None, **kwargs):
        flush_ret = False
        start = time.time()
        while not flush_ret:
            flush_ret = self.get_flush_state(segment_ids, timeout, **kwargs)
            end = time.time()
            if timeout is not None and end - start > timeout:
                raise MilvusException(message=f"wait for flush timeout, segment ids: {segment_ids}")

            if not flush_ret:
                time.sleep(0.5)

    @retry_on_rpc_failure()
    def flush(self, collection_names: list, timeout: Optional[float] = None, **kwargs):
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
        return None

    @retry_on_rpc_failure()
    def drop_index(
        self,
        collection_name: str,
        field_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name, field_name=field_name)
        request = Prepare.drop_index_request(collection_name, field_name, index_name)
        future = self._stub.DropIndex.future(request, timeout=timeout)
        response = future.result()
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    @retry_on_rpc_failure()
    def dummy(self, request_type: Any, timeout: Optional[float] = None, **kwargs):
        request = Prepare.dummy_request(request_type)
        future = self._stub.Dummy.future(request, timeout=timeout)
        return future.result()

    # TODO seems not in use
    @retry_on_rpc_failure()
    def fake_register_link(self, timeout: Optional[float] = None):
        request = Prepare.register_link_request()
        future = self._stub.RegisterLink.future(request, timeout=timeout)
        return future.result().status

    # TODO seems not in use
    @retry_on_rpc_failure()
    def get(
        self,
        collection_name: str,
        ids: List[int],
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        # TODO: some check
        request = Prepare.retrieve_request(collection_name, ids, output_fields, partition_names)
        future = self._stub.Retrieve.future(request, timeout=timeout)
        return future.result()

    @retry_on_rpc_failure()
    def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if output_fields is not None and not isinstance(output_fields, (list,)):
            raise ParamError(message="Invalid query format. 'output_fields' must be a list")
        request = Prepare.query_request(
            collection_name, expr, output_fields, partition_names, **kwargs
        )

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

        _, dynamic_fields = entity_helper.extract_dynamic_field_from_result(response)

        results = []
        for index in range(0, num_entities):
            entity_row_data = entity_helper.extract_row_data_from_fields_data(
                response.fields_data, index, dynamic_fields
            )
            results.append(entity_row_data)
        return results

    @retry_on_rpc_failure()
    def load_balance(
        self,
        collection_name: str,
        src_node_id: int,
        dst_node_ids: List[int],
        sealed_segment_ids: List[int],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        req = Prepare.load_balance_request(
            collection_name, src_node_id, dst_node_ids, sealed_segment_ids
        )
        future = self._stub.LoadBalance.future(req, timeout=timeout)
        status = future.result()
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    @retry_on_rpc_failure()
    def compact(self, collection_name: str, timeout: Optional[float] = None, **kwargs) -> int:
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
    def get_compaction_state(
        self, compaction_id: int, timeout: Optional[float] = None, **kwargs
    ) -> CompactionState:
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
            response.completedPlanNo,
        )

    @retry_on_rpc_failure()
    def wait_for_compaction_completed(
        self, compaction_id: int, timeout: Optional[float] = None, **kwargs
    ):
        start = time.time()
        while True:
            time.sleep(0.5)
            compaction_state = self.get_compaction_state(compaction_id, timeout, **kwargs)
            if compaction_state.state == State.Completed:
                return True
            if compaction_state == State.UndefiedState:
                return False
            end = time.time()
            if timeout is not None and end - start > timeout:
                raise MilvusException(
                    message=f"get compaction state timeout, compaction id: {compaction_id}"
                )

    @retry_on_rpc_failure()
    def get_compaction_plans(
        self, compaction_id: int, timeout: Optional[float] = None, **kwargs
    ) -> CompactionPlans:
        req = Prepare.get_compaction_state_with_plans(compaction_id)

        future = self._stub.GetCompactionStateWithPlans.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        cp = CompactionPlans(compaction_id, response.state)

        cp.plans = [Plan(m.sources, m.target) for m in response.mergeInfos]

        return cp

    @retry_on_rpc_failure()
    def get_replicas(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Replica:
        collection_id = self.describe_collection(collection_name, timeout, **kwargs)[
            "collection_id"
        ]

        req = Prepare.get_replicas(collection_id)
        future = self._stub.GetReplicas.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        groups = []
        for replica in response.replicas:
            shards = [
                Shard(s.dm_channel_name, s.node_ids, s.leaderID) for s in replica.shard_replicas
            ]
            groups.append(
                Group(
                    replica.replicaID,
                    shards,
                    replica.node_ids,
                    replica.resource_group_name,
                    replica.num_outbound_node,
                )
            )

        return Replica(groups)

    @retry_on_rpc_failure()
    def do_bulk_insert(
        self,
        collection_name: str,
        partition_name: str,
        files: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> int:
        req = Prepare.do_bulk_insert(collection_name, partition_name, files, **kwargs)
        future = self._stub.Import.future(req, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        if len(response.tasks) == 0:
            raise MilvusException(common_pb2.UNEXPECTED_ERROR, "no task id returned from server")
        return response.tasks[0]

    @retry_on_rpc_failure()
    def get_bulk_insert_state(
        self, task_id: int, timeout: Optional[float] = None, **kwargs
    ) -> BulkInsertState:
        req = Prepare.get_bulk_insert_state(task_id)
        future = self._stub.GetImportState.future(req, timeout=timeout)
        resp = future.result()
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return BulkInsertState(
            task_id, resp.state, resp.row_count, resp.id_list, resp.infos, resp.create_ts
        )

    @retry_on_rpc_failure()
    def list_bulk_insert_tasks(
        self, limit: int, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> list:
        req = Prepare.list_bulk_insert_tasks(limit, collection_name)
        future = self._stub.ListImportTasks.future(req, timeout=timeout)
        resp = future.result()
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return [
            BulkInsertState(t.id, t.state, t.row_count, t.id_list, t.infos, t.create_ts)
            for t in resp.tasks
        ]

    @retry_on_rpc_failure()
    def create_user(self, user: str, password: str, timeout: Optional[float] = None, **kwargs):
        check_pass_param(user=user, password=password)
        req = Prepare.create_user_request(user, password)
        resp = self._stub.CreateCredential(req, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def update_password(
        self,
        user: str,
        old_password: str,
        new_password: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        req = Prepare.update_password_request(user, old_password, new_password)
        resp = self._stub.UpdateCredential(req, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def delete_user(self, user: str, timeout: Optional[float] = None, **kwargs):
        req = Prepare.delete_user_request(user)
        resp = self._stub.DeleteCredential(req, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def list_usernames(self, timeout: Optional[float] = None, **kwargs):
        req = Prepare.list_usernames_request()
        resp = self._stub.ListCredUsers(req, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return resp.usernames

    @retry_on_rpc_failure()
    def create_role(self, role_name: str, timeout: Optional[float] = None, **kwargs):
        req = Prepare.create_role_request(role_name)
        resp = self._stub.CreateRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def drop_role(self, role_name: str, timeout: Optional[float] = None, **kwargs):
        req = Prepare.drop_role_request(role_name)
        resp = self._stub.DropRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def add_user_to_role(
        self, username: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.operate_user_role_request(
            username, role_name, milvus_types.OperateUserRoleType.AddUserToRole
        )
        resp = self._stub.OperateUserRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def remove_user_from_role(
        self, username: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.operate_user_role_request(
            username, role_name, milvus_types.OperateUserRoleType.RemoveUserFromRole
        )
        resp = self._stub.OperateUserRole(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def select_one_role(
        self, role_name: str, include_user_info: bool, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.select_role_request(role_name, include_user_info)
        resp = self._stub.SelectRole(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return RoleInfo(resp.results)

    @retry_on_rpc_failure()
    def select_all_role(self, include_user_info: bool, timeout: Optional[float] = None, **kwargs):
        req = Prepare.select_role_request(None, include_user_info)
        resp = self._stub.SelectRole(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return RoleInfo(resp.results)

    @retry_on_rpc_failure()
    def select_one_user(
        self, username: str, include_role_info: bool, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.select_user_request(username, include_role_info)
        resp = self._stub.SelectUser(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return UserInfo(resp.results)

    @retry_on_rpc_failure()
    def select_all_user(self, include_role_info: bool, timeout: Optional[float] = None, **kwargs):
        req = Prepare.select_user_request(None, include_role_info)
        resp = self._stub.SelectUser(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return UserInfo(resp.results)

    @retry_on_rpc_failure()
    def grant_privilege(
        self,
        role_name: str,
        object: str,
        object_name: str,
        privilege: str,
        db_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        req = Prepare.operate_privilege_request(
            role_name,
            object,
            object_name,
            privilege,
            db_name,
            milvus_types.OperatePrivilegeType.Grant,
        )
        resp = self._stub.OperatePrivilege(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def revoke_privilege(
        self,
        role_name: str,
        object: str,
        object_name: str,
        privilege: str,
        db_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        req = Prepare.operate_privilege_request(
            role_name,
            object,
            object_name,
            privilege,
            db_name,
            milvus_types.OperatePrivilegeType.Revoke,
        )
        resp = self._stub.OperatePrivilege(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def select_grant_for_one_role(
        self, role_name: str, db_name: str, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.select_grant_request(role_name, None, None, db_name)
        resp = self._stub.SelectGrant(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return GrantInfo(resp.entities)

    @retry_on_rpc_failure()
    def select_grant_for_role_and_object(
        self,
        role_name: str,
        object: str,
        object_name: str,
        db_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        req = Prepare.select_grant_request(role_name, object, object_name, db_name)
        resp = self._stub.SelectGrant(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return GrantInfo(resp.entities)

    @retry_on_rpc_failure()
    def get_server_version(self, timeout: Optional[float] = None, **kwargs) -> str:
        req = Prepare.get_server_version()
        resp = self._stub.GetVersion(req, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.version

    @retry_on_rpc_failure()
    def create_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        req = Prepare.create_resource_group(name)
        resp = self._stub.CreateResourceGroup(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def drop_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        req = Prepare.drop_resource_group(name)
        resp = self._stub.DropResourceGroup(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def list_resource_groups(self, timeout: Optional[float] = None, **kwargs):
        req = Prepare.list_resource_groups()
        resp = self._stub.ListResourceGroups(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return list(resp.resource_groups)

    @retry_on_rpc_failure()
    def describe_resource_group(
        self, name: str, timeout: Optional[float] = None, **kwargs
    ) -> ResourceGroupInfo:
        req = Prepare.describe_resource_group(name)
        resp = self._stub.DescribeResourceGroup(req, wait_for_ready=True, timeout=timeout)
        if resp.status.error_code != 0:
            raise MilvusException(resp.status.error_code, resp.status.reason)
        return ResourceGroupInfo(resp.resource_group)

    @retry_on_rpc_failure()
    def transfer_node(
        self, source: str, target: str, num_node: int, timeout: Optional[float] = None, **kwargs
    ):
        req = Prepare.transfer_node(source, target, num_node)
        resp = self._stub.TransferNode(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def transfer_replica(
        self,
        source: str,
        target: str,
        collection_name: str,
        num_replica: int,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        req = Prepare.transfer_replica(source, target, collection_name, num_replica)
        resp = self._stub.TransferReplica(req, wait_for_ready=True, timeout=timeout)
        if resp.error_code != 0:
            raise MilvusException(resp.error_code, resp.reason)

    @retry_on_rpc_failure()
    def get_flush_all_state(self, flush_all_ts: int, timeout: Optional[float] = None, **kwargs):
        req = Prepare.get_flush_all_state_request(flush_all_ts)
        response = self._stub.GetFlushAllState(req, timeout=timeout)
        status = response.status
        if status.error_code == 0:
            return response.flushed
        raise MilvusException(status.error_code, status.reason)

    def _wait_for_flush_all(self, flush_all_ts: int, timeout: Optional[float] = None, **kwargs):
        flush_ret = False
        start = time.time()
        while not flush_ret:
            flush_ret = self.get_flush_all_state(flush_all_ts, timeout, **kwargs)
            end = time.time()
            if timeout is not None and end - start > timeout:
                raise MilvusException(
                    message=f"wait for flush all timeout, flush_all_ts: {flush_all_ts}"
                )

            if not flush_ret:
                time.sleep(5)

    @retry_on_rpc_failure()
    def flush_all(self, timeout: Optional[float] = None, **kwargs):
        request = Prepare.flush_all_request()
        future = self._stub.FlushAll.future(request, timeout=timeout)
        response = future.result()
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)

        def _check():
            self._wait_for_flush_all(response.flush_all_ts, timeout, **kwargs)

        if kwargs.get("_async", False):
            flush_future = FlushFuture(future)
            flush_future.add_callback(_check)

            user_cb = kwargs.get("_callback", None)
            if user_cb:
                flush_future.add_callback(user_cb)

            return flush_future

        _check()
        return None

    @retry_on_rpc_failure()
    @upgrade_reminder
    def __internal_register(self, user: str, host: str) -> int:
        req = Prepare.register_request(user, host)
        response = self._stub.Connect(request=req)
        if response.status.error_code != common_pb2.Success:
            raise MilvusException(response.status.error_code, response.status.reason)
        return response.identifier

    @retry_on_rpc_failure()
    @ignore_unimplemented(0)
    def alloc_timestamp(self, timeout: Optional[float] = None) -> int:
        request = milvus_types.AllocTimestampRequest()
        response = self._stub.AllocTimestamp(request, timeout=timeout)
        if response.status.error_code != common_pb2.Success:
            raise MilvusException(response.status.error_code, response.status.reason)
        return response.timestamp
