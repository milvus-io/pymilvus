from abc import ABCMeta, abstractmethod

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import common_pb2 as common_types
from ..grpc_gen import milvus_pb2 as milvus_types
from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import schema_pb2 as schema_types


class IServer(metaclass=ABCMeta):
    """
    Abstraction for the function of milvus server, to makes the unit tests not depend on a milvus server.
    """

    def __init__(self):
        pass

    @abstractmethod
    def create_collection(self, collection_name, fields, shards_num) -> common_types.Status:
        pass

    @abstractmethod
    def drop_collection(self, collection_name) -> common_types.Status:
        pass

    @abstractmethod
    def has_collection(self, collection_name) -> milvus_types.BoolResponse:
        pass

    @abstractmethod
    def describe_collection(self, collection_name) -> milvus_types.DescribeCollectionResponse:
        pass

    @abstractmethod
    def list_collections(self) -> milvus_types.ShowCollectionsResponse:
        pass

    @abstractmethod
    def create_partition(self, collection_name, partition_name) -> common_types.Status:
        pass

    @abstractmethod
    def drop_partition(self, collection_name, partition_name) -> common_types.Status:
        pass

    @abstractmethod
    def has_partition(self, collection_name, partition_name) -> milvus_types.BoolResponse:
        pass


class GrpcServer(IServer):
    """
    Methods in this class cannot be covered by unit tests(unit tests should not depends on the milvus server), so that
    keep them as simple as possible.
    """

    def __init__(self, host="localhost", port="19530"):
        super().__init__()
        self._channel = grpc.insecure_channel(
            f"{host}:{port}",
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1),
                     ('grpc.enable_retries', 1),
                     ('grpc.keepalive_time_ms', 55000)]
        )
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)

    def create_collection(self, collection_name, fields, shards_num) -> common_types.Status:
        assert isinstance(fields, dict)
        assert "fields" in fields
        assert sum(1 for field in fields["fields"] if "is_primary" in field) == 1
        assert sum(1 for field in fields["fields"] if "auto_id" in field) <= 1

        schema = schema_types.CollectionSchema(name=collection_name)
        for field in fields["fields"]:
            field_schema = schema_types.FieldSchema()
            assert "name" in field
            field_schema.name = field["name"]
            assert "type" in field
            field_schema.data_type = field["type"]

            field_schema.is_primary_key = field.get("is_primary", False)
            field_schema.autoID = field.get('auto_id', False)

            if "params" in field:
                assert isinstance(field["params"], dict)
                assert "dim" in field["params"]
                kv_pair = common_types.KeyValuePair(key="dim", value=str(int(field["params"]["dim"])))
                field_schema.type_params.append(kv_pair)

            schema.fields.append(field_schema)

        request = milvus_types.CreateCollectionRequest(collection_name=collection_name,
                                                       schema=bytes(schema.SerializeToString()), shards_num=shards_num)
        return self._stub.CreateCollection(request)

    def drop_collection(self, collection_name) -> common_types.Status:
        request = milvus_types.DropCollectionRequest(collection_name=collection_name)
        return self._stub.DropCollection(request)

    def has_collection(self, collection_name) -> milvus_types.BoolResponse:
        request = milvus_types.HasCollectionRequest(collection_name=collection_name)
        return self._stub.HasCollection(request)

    def describe_collection(self, collection_name) -> milvus_types.DescribeCollectionResponse:
        request = milvus_types.DescribeCollectionRequest(collection_name=collection_name)
        return self._stub.DescribeCollection(request)

    def list_collections(self) -> milvus_types.ShowCollectionsResponse:
        request = milvus_types.ShowCollectionsRequest()
        return self._stub.ShowCollections(request)

    def create_partition(self, collection_name, partition_name) -> common_types.Status:
        request = milvus_types.CreatePartitionRequest(collection_name=collection_name, partition_name=partition_name)
        return self._stub.CreatePartition(request)

    def drop_partition(self, collection_name, partition_name) -> common_types.Status:
        request = milvus_types.DropPartitionRequest(collection_name=collection_name, partition_name=partition_name)
        return self._stub.DropPartition(request)

    def has_partition(self, collection_name, partition_name) -> milvus_types.BoolResponse:
        request = milvus_types.HasPartitionRequest(collection_name=collection_name, partition_name=partition_name)
        resp = self._stub.HasPartition(request)
        return resp
