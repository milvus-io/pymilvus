import asyncio
import copy

import grpc.aio

from ...client.grpc_handler import (
    AbstractGrpcHandler,
    Status,
    MilvusException,
    # retry_on_rpc_failure,
    check_pass_param,
    get_consistency_level,
    ts_utils,
    Prepare,
    CollectionSchema,
    DescribeCollectionException,
    ChunkedQueryResult,
    common_pb2,
    check_invalid_binary_vector,
    ParamError,
    milvus_types,
    MutationResult,
    DefaultConfigs,
    DataType,
    check_index_params,
)


class GrpcHandler(AbstractGrpcHandler[grpc.aio.Channel]):
    _insecure_channel = staticmethod(grpc.aio.insecure_channel)
    _secure_channel = staticmethod(grpc.aio.secure_channel)

    async def _channel_ready(self):
        if self._channel is None:
            raise MilvusException(
                Status.CONNECT_FAILED,
                'No channel in handler, please setup grpc channel first',
            )
        await self._channel.channel_ready()

    def _header_adder_interceptor(self, header, value):
        raise NotImplementedError  # TODO

    # TODO: @retry_on_rpc_failure()
    async def create_collection(self, collection_name, fields, shards_num=2, timeout=None, **kwargs):
        request = Prepare.create_collection_request(collection_name, fields, shards_num=shards_num, **kwargs)

        status = await self._stub.CreateCollection(request, timeout=timeout)
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

    # TODO: @retry_on_rpc_failure()
    async def has_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        reply = await self._stub.DescribeCollection(request, timeout=timeout)

        if reply.status.error_code == common_pb2.Success:
            return True

        # TODO: Workaround for unreasonable describe collection results and error_code
        if reply.status.error_code == common_pb2.UnexpectedError and "can\'t find collection" in reply.status.reason:
            return False

        raise MilvusException(reply.status.error_code, reply.status.reason)

    # TODO: @retry_on_rpc_failure()
    async def describe_collection(self, collection_name, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name)
        request = Prepare.describe_collection_request(collection_name)
        response = await self._stub.DescribeCollection(request, timeout=timeout)

        status = response.status
        if status.error_code != 0:
            raise DescribeCollectionException(status.error_code, status.reason)

        return CollectionSchema(raw=response).dict()

    # TODO: @retry_on_rpc_failure()
    async def batch_insert(self, collection_name, entities, partition_name=None, timeout=None, **kwargs):
        if not check_invalid_binary_vector(entities):
            raise ParamError(message="Invalid binary vector data exists")
        insert_param = kwargs.get('insert_param', None)
        if insert_param and not isinstance(insert_param, milvus_types.RowBatch):
            raise ParamError(message="The value of key 'insert_param' is invalid")
        if not isinstance(entities, list):
            raise ParamError(message="None entities, please provide valid entities.")

        collection_schema = kwargs.get("schema", None)
        if not collection_schema:
            collection_schema = await self.describe_collection(collection_name, timeout=timeout, **kwargs)

        fields_info = collection_schema["fields"]
        request = insert_param or Prepare.batch_insert_param(collection_name, entities, partition_name, fields_info)
        response = await self._stub.Insert(request, timeout=timeout)
        if response.status.error_code != 0:
            raise MilvusException(response.status.error_code, response.status.reason)
        m = MutationResult(response)
        ts_utils.update_collection_ts(collection_name, m.timestamp)
        return m

    async def _execute_search_requests(self, requests, timeout=None, *, auto_id=True, round_decimal=-1, **kwargs):
        async def _raise_milvus_exception_on_error_response(awaitable_response):
            response = await awaitable_response
            if response.status.error_code != 0:
                raise MilvusException(response.status.error_code, response.status.reason)
            return response

        raws: list = await asyncio.gather(*(
            _raise_milvus_exception_on_error_response(
                self._stub.Search(request, timeout=timeout)
            )
            for request in requests
        ))
        return ChunkedQueryResult(raws, auto_id, round_decimal)

    # TODO: @retry_on_rpc_failure(retry_on_deadline=False)
    async def search(
        self, collection_name, data, anns_field, param, limit,
        expression=None, partition_names=None, output_fields=None,
        round_decimal=-1, timeout=None, schema=None, **kwargs,
    ):
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

        if schema is None:
            schema = await self.describe_collection(collection_name, timeout=timeout, **kwargs)

        consistency_level = schema["consistency_level"]
        # overwrite the consistency level defined when user created the collection
        consistency_level = get_consistency_level(kwargs.get("consistency_level", consistency_level))

        ts_utils.construct_guarantee_ts(consistency_level, collection_name, kwargs)

        requests = Prepare.search_requests_with_expr(collection_name, data, anns_field, param, limit, schema,
                                                     expression, partition_names, output_fields, round_decimal,
                                                     **kwargs)

        auto_id = schema["auto_id"]
        return await self._execute_search_requests(
            requests, timeout, round_decimal=round_decimal, auto_id=auto_id, **kwargs,
        )

    # TODO: @retry_on_rpc_failure()
    async def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        # for historical reason, index_name contained in kwargs.
        index_name = kwargs.pop("index_name", DefaultConfigs.IndexName)
        copy_kwargs = copy.deepcopy(kwargs)

        collection_desc = await self.describe_collection(collection_name, timeout=timeout, **copy_kwargs)

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

        index_param = Prepare.create_index_request(collection_name, field_name, params, index_name=index_name)

        status = await self._stub.CreateIndex(index_param, timeout=timeout)
        if status.error_code != 0:
            raise MilvusException(status.error_code, status.reason)

        return Status(status.error_code, status.reason)

    # TODO: @retry_on_rpc_failure()
    async def load_collection(self, collection_name, replica_number=1, timeout=None, **kwargs):
        check_pass_param(collection_name=collection_name, replica_number=replica_number)
        _refresh = kwargs.get("_refresh", False)
        _resource_groups = kwargs.get("_resource_groups")
        request = Prepare.load_collection("", collection_name, replica_number, _refresh, _resource_groups)
        response = await self._stub.LoadCollection(request, timeout=timeout)
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)

    # TODO: @retry_on_rpc_failure()
    async def load_partitions(self, collection_name, partition_names, replica_number=1, timeout=None, **kwargs):
        check_pass_param(
            collection_name=collection_name,
            partition_name_array=partition_names,
            replica_number=replica_number)
        _refresh = kwargs.get("_refresh", False)
        _resource_groups = kwargs.get("_resource_groups")
        request = Prepare.load_partitions("", collection_name, partition_names, replica_number, _refresh,
                                          _resource_groups)
        response = await self._stub.LoadPartitions(request, timeout=timeout)
        if response.error_code != 0:
            raise MilvusException(response.error_code, response.reason)
