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
from .grpc_handler import GrpcHandler
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

class AsyncGrpcHandler(GrpcHandler):
    def __init__(
        self,
        uri: str = Config.GRPC_URI,
        host: str = "",
        port: str = "",
        channel: Optional[grpc.Channel] = None,
        **kwargs,
    ) -> None:
        self._async_stub = None
        self._async_channel = None
        super().__init__(uri, host, port, channel, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self: object, exc_type: object, exc_val: object, exc_tb: object):
        pass

    def _wait_for_channel_ready(self, timeout: Union[float] = 10, retry_interval: float = 1):
        if self._channel is None:
            raise MilvusException(
                code=Status.CONNECT_FAILED,
                message="No channel in handler, please setup grpc channel first",
            )
        try:
            async def wait_for_async_channel_ready():
                await self._async_channel.channel_ready()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(wait_for_async_channel_ready())

            grpc.channel_ready_future(self._channel).result(timeout=timeout)

            self._setup_identifier_interceptor(self._user, timeout=timeout)
        except grpc.FutureTimeoutError as e:
            raise MilvusException(
                code=Status.CONNECT_FAILED,
                message=f"Fail connecting to server on {self._address}, illegal connection params or server unavailable",
            ) from e
        except Exception as e:
            raise e from e

    def _setup_grpc_channel(self):
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
                self._channel = grpc.secure_channel(
                    self._address,
                    creds,
                    options=opts,
                )
                self._async_channel = grpc.aio.secure_channel(
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

            async_log_level_interceptor = async_header_adder_interceptor(
                ["log_level"], [self._log_level]
            )
            self._async_channel._unary_unary_interceptors.append(async_log_level_interceptor)

            self._log_level = None
        if self._request_id:
            request_id_interceptor = interceptor.header_adder_interceptor(
                ["client_request_id"], [self._request_id]
            )
            self._final_channel = grpc.intercept_channel(
                self._final_channel, request_id_interceptor
            )

            async_request_id_interceptor = async_header_adder_interceptor(
                ["client_request_id"], [self._request_id]
            )
            self._async_channel._unary_unary_interceptors.append(async_request_id_interceptor)

            self._request_id = None
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)
        self._async_stub = milvus_pb2_grpc.MilvusServiceStub(self._async_channel)

    def _setup_identifier_interceptor(self, user: str, timeout: int = 10):
        host = socket.gethostname()
        self._identifier = self.__internal_register(user, host, timeout=timeout)
        self._identifier_interceptor = interceptor.header_adder_interceptor(
            ["identifier"], [str(self._identifier)]
        )
        self._final_channel = grpc.intercept_channel(
            self._final_channel, self._identifier_interceptor
        )
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._final_channel)

        _async_identifier_interceptor = async_header_adder_interceptor(
            ["identifier"], [str(self._identifier)]
        )
        self._async_channel._unary_unary_interceptors.append(_async_identifier_interceptor)
        self._async_stub = milvus_pb2_grpc.MilvusServiceStub(self._async_channel)

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
        request = self._prepare_row_insert_request(
            collection_name, entities, partition_name, schema, timeout, **kwargs
        )
        resp = await self._async_stub.Insert(request=request, timeout=timeout)
        check_status(resp.status)
        ts_utils.update_collection_ts(collection_name, resp.timestamp)
        return MutationResult(resp)

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
            request = self._prepare_batch_upsert_request(
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
        request = self._prepare_row_upsert_request(
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
    def __internal_register(self, user: str, host: str, **kwargs) -> int:
        req = Prepare.register_request(user, host)
        response = self._stub.Connect(request=req)
        check_status(response.status)
        return response.identifier