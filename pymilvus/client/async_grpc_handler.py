import asyncio
import base64
import copy
import json
import socket
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
from urllib import parse

import grpc
from grpc._cython import cygrpc

from pymilvus.decorators import ignore_unimplemented, retry_on_rpc_failure, upgrade_reminder
from pymilvus.exceptions import (
    AmbiguousIndexName,
    DescribeCollectionException,
    ErrorCode,
    ExceptionsMessage,
    MilvusException,
    ParamError,
)
from pymilvus.grpc_gen import common_pb2, milvus_pb2_grpc
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.settings import Config

from . import entity_helper, interceptor, ts_utils, utils
from .abstract import AnnSearchRequest, BaseRanker, CollectionSchema, MutationResult, SearchResult
from .async_interceptor import async_header_adder_interceptor
from .asynch import (
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
from .constants import ITERATOR_SESSION_TS_FIELD
from .prepare import Prepare
from .types import (
    BulkInsertState,
    CompactionPlans,
    CompactionState,
    DatabaseInfo,
    DataType,
    ExtraList,
    GrantInfo,
    Group,
    IndexState,
    LoadState,
    Plan,
    PrivilegeGroupInfo,
    Replica,
    ResourceGroupConfig,
    ResourceGroupInfo,
    RoleInfo,
    Shard,
    State,
    Status,
    UserInfo,
    get_cost_extra,
)
from .utils import (
    check_invalid_binary_vector,
    check_status,
    get_server_type,
    is_successful,
    len_of,
)

class AsyncGrpcHandler:
    def __init__(
        self,
        uri: str = Config.GRPC_URI,
        host: str = "",
        port: str = "",
        channel: Optional[grpc.aio.Channel] = None,
        **kwargs,
    ) -> None:
        self._async_stub = None
        self._async_channel = channel

        addr = kwargs.get("address")
        self._address = addr if addr is not None else self.__get_address(uri, host, port)
        self._log_level = None
        self._request_id = None
        self._user = kwargs.get("user")
        self._set_authorization(**kwargs)
        self._setup_db_interceptor(kwargs.get("db_name"))
        self._setup_grpc_channel()  # init channel and stub
        self.callbacks = []

    def register_state_change_callback(self, callback: Callable):
        self.callbacks.append(callback)
        self._async_channel.subscribe(callback, try_to_connect=True)

    def deregister_state_change_callbacks(self):
        for callback in self.callbacks:
            self._async_channel.unsubscribe(callback)
        self.callbacks = []

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
            kwargs.get("user"),
            kwargs.get("password"),
            kwargs.get("token"),
        )

    def __enter__(self):
        return self

    def __exit__(self: object, exc_type: object, exc_val: object, exc_tb: object):
        pass

    def _wait_for_channel_ready(self, timeout: Union[float] = 10, retry_interval: float = 1):
        try:
            async def wait_for_async_channel_ready():
                await self._async_channel.channel_ready()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(wait_for_async_channel_ready())

            self._setup_identifier_interceptor(self._user, timeout=timeout)
        except grpc.FutureTimeoutError as e:
            raise MilvusException(
                code=Status.CONNECT_FAILED,
                message=f"Fail connecting to server on {self._address}, illegal connection params or server unavailable",
            ) from e
        except Exception as e:
            raise e from e

    def close(self):
        self.deregister_state_change_callbacks()
        self._async_channel.close()

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
        if self._async_channel is None:
            opts = [
                (cygrpc.ChannelArgKey.max_send_message_length, -1),
                (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                ("grpc.enable_retries", 1),
                ("grpc.keepalive_time_ms", 55000),
            ]
            if not self._secure:
                self._async_channel = grpc.aio.insecure_channel(
                    self._address,
                    options=opts,
                )
            else:
                if self._server_name != "":
                    opts.append(("grpc.ssl_target_name_override", self._server_name))

                root_cert, private_k, cert_chain = None, None, None
                if self._server_pem_path != "":
                    with Path(self._server_pem_path).open("rb") as f:
                        root_cert = f.read()
                elif (
                    self._client_pem_path != ""
                    and self._client_key_path != ""
                    and self._ca_pem_path != ""
                ):
                    with Path(self._ca_pem_path).open("rb") as f:
                        root_cert = f.read()
                    with Path(self._client_key_path).open("rb") as f:
                        private_k = f.read()
                    with Path(self._client_pem_path).open("rb") as f:
                        cert_chain = f.read()

                creds = grpc.ssl_channel_credentials(
                    root_certificates=root_cert,
                    private_key=private_k,
                    certificate_chain=cert_chain,
                )
                self._async_channel = grpc.aio.secure_channel(
                    self._address,
                    creds,
                    options=opts,
                )

        # avoid to add duplicate headers.
        self._final_channel = self._async_channel
        if self._log_level:

            async_log_level_interceptor = async_header_adder_interceptor(
                ["log_level"], [self._log_level]
            )
            self._final_channel._unary_unary_interceptors.append(async_log_level_interceptor)

            self._log_level = None
        if self._request_id:

            async_request_id_interceptor = async_header_adder_interceptor(
                ["client_request_id"], [self._request_id]
            )
            self._final_channel._unary_unary_interceptors.append(async_request_id_interceptor)

            self._request_id = None
        self._async_stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

    def _setup_identifier_interceptor(self, user: str, timeout: int = 10):
        host = socket.gethostname()
        self._identifier = self.__async_internal_register(user, host, timeout=timeout)
        _async_identifier_interceptor = async_header_adder_interceptor(
            ["identifier"], [str(self._identifier)]
        )
        self._async_channel._unary_unary_interceptors.append(_async_identifier_interceptor)
        self._async_stub = milvus_pb2_grpc.MilvusServiceStub(self._async_channel)

    @property
    def server_address(self):
        return self._address

    def get_server_type(self):
        return get_server_type(self.server_address.split(":")[0])

    @retry_on_rpc_failure()
    async def async_create_collection(
        self, collection_name: str, fields: List, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.create_collection_request(collection_name, fields, **kwargs)
        response = await self._async_stub.CreateCollection(request, timeout=timeout)
        check_status(response)

    @retry_on_rpc_failure()
    async def async_drop_collection(self, collection_name: str, timeout: Optional[float] = None):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.drop_collection_request(collection_name)
        response = await self._async_stub.DropCollection(request, timeout=timeout)
        check_status(response)

    @retry_on_rpc_failure()
    async def async_load_collection(
        self,
        collection_name: str,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(
            collection_name=collection_name, replica_number=replica_number, timeout=timeout
        )
        refresh = kwargs.get("refresh", kwargs.get("_refresh", False))
        resource_groups = kwargs.get("resource_groups", kwargs.get("_resource_groups"))
        load_fields = kwargs.get("load_fields", kwargs.get("_load_fields"))
        skip_load_dynamic_field = kwargs.get(
            "skip_load_dynamic_field", kwargs.get("_skip_load_dynamic_field", False)
        )

        request = Prepare.load_collection(
            "",
            collection_name,
            replica_number,
            refresh,
            resource_groups,
            load_fields,
            skip_load_dynamic_field,
        )
        response = await self._async_stub.LoadCollection(request, timeout=timeout)
        check_status(response)

    @retry_on_rpc_failure()
    async def async_describe_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        request = Prepare.describe_collection_request(collection_name)
        response = await self._async_stub.DescribeCollection(request, timeout=timeout)
        status = response.status

        if is_successful(status):
            return CollectionSchema(raw=response).dict()

        raise DescribeCollectionException(status.code, status.reason, status.error_code)

    async def _async_get_info(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        schema = kwargs.get("schema")
        if not schema:
            schema = await self.async_describe_collection(collection_name, timeout=timeout)

        fields_info = schema.get("fields")
        enable_dynamic = schema.get("enable_dynamic_field", False)

        return fields_info, enable_dynamic

    @retry_on_rpc_failure()
    async def async_insert_rows(
        self,
        collection_name: str,
        entities: Union[Dict, List[Dict]],
        partition_name: Optional[str] = None,
        schema: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        request = await self._async_prepare_row_insert_request(
            collection_name, entities, partition_name, schema, timeout, **kwargs
        )
        resp = await self._async_stub.Insert(request=request, timeout=timeout)
        check_status(resp.status)
        ts_utils.update_collection_ts(collection_name, resp.timestamp)
        return MutationResult(resp)

    async def _async_prepare_row_insert_request(
        self,
        collection_name: str,
        entity_rows: Union[List[Dict], Dict],
        partition_name: Optional[str] = None,
        schema: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(entity_rows, dict):
            entity_rows = [entity_rows]

        if not isinstance(schema, dict):
            schema = await self.async_describe_collection(collection_name, timeout=timeout)

        fields_info = schema.get("fields")
        enable_dynamic = schema.get("enable_dynamic_field", False)

        return Prepare.row_insert_param(
            collection_name,
            entity_rows,
            partition_name,
            fields_info,
            enable_dynamic=enable_dynamic,
        )

    async def async_delete(
        self,
        collection_name: str,
        expression: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(collection_name=collection_name, timeout=timeout)
        try:
            req = Prepare.delete_request(
                collection_name=collection_name,
                filter=expression,
                partition_name=partition_name,
                consistency_level=kwargs.pop("consistency_level", 0),
                **kwargs,
            )

            response = await self._async_stub.Delete(req, timeout=timeout)

            m = MutationResult(response)
            ts_utils.update_collection_ts(collection_name, m.timestamp)
        except Exception as err:
            raise err from err
        else:
            return m

    async def _async_prepare_batch_upsert_request(
        self,
        collection_name: str,
        entities: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        param = kwargs.get("upsert_param")
        if param and not isinstance(param, milvus_types.UpsertRequest):
            raise ParamError(message="The value of key 'upsert_param' is invalid")
        if not isinstance(entities, list):
            raise ParamError(message="'entities' must be a list, please provide valid entity data.")

        schema = kwargs.get("schema")
        if not schema:
            schema = await self.async_describe_collection(collection_name, timeout=timeout, **kwargs)

        fields_info = schema["fields"]

        return (
            param
            if param
            else Prepare.batch_upsert_param(collection_name, entities, partition_name, fields_info)
        )

    @retry_on_rpc_failure()
    async def async_upsert(
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
            request = await self._async_prepare_batch_upsert_request(
                collection_name, entities, partition_name, timeout, **kwargs
            )
            response = await self._async_stub.Upsert(request, timeout=timeout)
            check_status(response.status)
            m = MutationResult(response)
            ts_utils.update_collection_ts(collection_name, m.timestamp)
        except Exception as err:
            raise err from err
        else:
            return m

    async def _async_prepare_row_upsert_request(
        self,
        collection_name: str,
        rows: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if not isinstance(rows, list):
            raise ParamError(message="'rows' must be a list, please provide valid row data.")

        fields_info, enable_dynamic = await self._async_get_info(collection_name, timeout, **kwargs)
        return Prepare.row_upsert_param(
            collection_name,
            rows,
            partition_name,
            fields_info,
            enable_dynamic=enable_dynamic,
        )

    @retry_on_rpc_failure()
    async def async_upsert_rows(
        self,
        collection_name: str,
        entities: List,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(entities, dict):
            entities = [entities]
        request = await self._async_prepare_row_upsert_request(
            collection_name, entities, partition_name, timeout, **kwargs
        )
        response = await self._async_stub.Upsert(request, timeout=timeout)
        check_status(response.status)
        m = MutationResult(response)
        ts_utils.update_collection_ts(collection_name, m.timestamp)
        return m

    async def _async_execute_search(
        self, request: milvus_types.SearchRequest, timeout: Optional[float] = None, **kwargs
    ):
        try:
            response = await self._async_stub.Search(request, timeout=timeout)
            check_status(response.status)
            round_decimal = kwargs.get("round_decimal", -1)
            return SearchResult(
                response.results,
                round_decimal,
                status=response.status,
                session_ts=response.session_ts,
            )
        except Exception as e:
            raise e from e

    async def _async_execute_hybrid_search(
        self, request: milvus_types.HybridSearchRequest, timeout: Optional[float] = None, **kwargs
    ):
        try:
            response = await self._async_stub.HybridSearch(request, timeout=timeout)
            check_status(response.status)
            round_decimal = kwargs.get("round_decimal", -1)
            return SearchResult(response.results, round_decimal, status=response.status)

        except Exception as e:
            raise e from e

    @retry_on_rpc_failure()
    async def async_search(
        self,
        collection_name: str,
        data: Union[List[List[float]], utils.SparseMatrixInputType],
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
            guarantee_timestamp=kwargs.get("guarantee_timestamp"),
            timeout=timeout,
        )
        request = Prepare.search_requests_with_expr(
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
        return await self._async_execute_search(
            request, timeout, round_decimal=round_decimal, **kwargs
        )

    @retry_on_rpc_failure()
    async def async_hybrid_search(
        self,
        collection_name: str,
        reqs: List[AnnSearchRequest],
        rerank: BaseRanker,
        limit: int,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        round_decimal: int = -1,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        check_pass_param(
            limit=limit,
            round_decimal=round_decimal,
            partition_name_array=partition_names,
            output_fields=output_fields,
            guarantee_timestamp=kwargs.get("guarantee_timestamp"),
            timeout=timeout,
        )

        requests = []
        for req in reqs:
            search_request = Prepare.search_requests_with_expr(
                collection_name,
                req.data,
                req.anns_field,
                req.param,
                req.limit,
                req.expr,
                partition_names=partition_names,
                round_decimal=round_decimal,
                **kwargs,
            )
            requests.append(search_request)

        hybrid_search_request = Prepare.hybrid_search_request_with_ranker(
            collection_name,
            requests,
            rerank.dict(),
            limit,
            partition_names,
            output_fields,
            round_decimal,
            **kwargs,
        )
        return await self._async_execute_hybrid_search(
            hybrid_search_request, timeout, round_decimal=round_decimal, **kwargs
        )

    @retry_on_rpc_failure()
    async def async_create_index(
        self,
        collection_name: str,
        field_name: str,
        params: Dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        index_name = kwargs.pop("index_name", Config.IndexName)
        copy_kwargs = copy.deepcopy(kwargs)

        collection_desc = await self.async_describe_collection(collection_name, timeout=timeout, **copy_kwargs)

        valid_field = False
        for fields in collection_desc["fields"]:
            if field_name != fields["name"]:
                continue
            valid_field = True
            if fields["type"] not in {
                DataType.FLOAT_VECTOR,
                DataType.BINARY_VECTOR,
                DataType.FLOAT16_VECTOR,
                DataType.BFLOAT16_VECTOR,
                DataType.SPARSE_FLOAT_VECTOR,
            }:
                break

        if not valid_field:
            raise MilvusException(message=f"cannot create index on non-existed field: {field_name}")

        index_param = Prepare.create_index_request(
            collection_name, field_name, params, index_name=index_name
        )

        status = await self._async_stub.CreateIndex(index_param, timeout=timeout)
        check_status(status)

        return Status(status.code, status.reason)

    @retry_on_rpc_failure()
    async def async_get(
        self,
        collection_name: str,
        ids: List[int],
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ):
        # TODO: some check
        request = Prepare.retrieve_request(collection_name, ids, output_fields, partition_names)
        return await self._async_stub.Retrieve.async_get(request, timeout=timeout)

    @retry_on_rpc_failure()
    async def async_query(
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
        response = await self._async_stub.Query(request, timeout=timeout)
        check_status(response.status)

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
        for index in range(num_entities):
            entity_row_data = entity_helper.extract_row_data_from_fields_data(
                response.fields_data, index, dynamic_fields
            )
            results.append(entity_row_data)

        extra_dict = get_cost_extra(response.status)
        extra_dict[ITERATOR_SESSION_TS_FIELD] = response.session_ts
        return ExtraList(results, extra=extra_dict)

    @retry_on_rpc_failure()
    @upgrade_reminder
    def __async_internal_register(self, user: str, host: str, **kwargs) -> int:
        req = Prepare.register_request(user, host)
        async def wait_for_connect_response():
            return await self._async_stub.Connect(request=req)

        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(wait_for_connect_response())

        check_status(response.status)
        return response.identifier