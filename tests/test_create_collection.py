import grpc
import grpc_testing
import pytest
import random

from pymilvus.grpc_gen import milvus_pb2, schema_pb2, common_pb2
from pymilvus import Milvus, DataType


class Fields:
    class NormalizedField:
        def __init__(self, **kwargs):
            self.name = kwargs.get("name", None)
            self.is_primary_key = kwargs.get("is_primary_key", False)
            self.data_type = kwargs.get("data_type", None)
            self.type_params = kwargs.get("type_params", dict())
            self.autoID = kwargs.get("autoID", False)

        def __eq__(self, other):
            if isinstance(other, Fields.NormalizedField):
                return self.name == other.name and \
                       self.is_primary_key == other.is_primary_key and \
                       self.data_type == other.data_type and \
                       self.type_params == other.type_params and \
                       self.autoID == other.autoID
            return False

        def __repr__(self):
            dump = f"(name: {self.name}"
            dump += f", id_primary_key:{self.is_primary_key}"
            dump += f", data_type:{self.data_type}"
            dump += f", type_params:{self.type_params}"
            dump += f", autoID:{self.autoID})"
            return dump

    @classmethod
    def equal(cls, grpc_fields, dict_fields):
        n_grpc_fields = {
            field.name: Fields.NormalizedField(name=field.name,
                                               is_primary_key=field.is_primary_key,
                                               data_type=field.data_type,
                                               type_params={pair.key: pair.value for pair in field.type_params},
                                               autoID=field.autoID
                                               )
            for field in grpc_fields}
        n_dict_fields = {
            field["name"]: Fields.NormalizedField(name=field["name"],
                                                  is_primary_key=field.get("is_primary", False),
                                                  data_type=field["type"],
                                                  type_params=field.get("params", dict()),
                                                  autoID=field.get("auto_id", False)
                                                  )
            for field in dict_fields
        }
        return n_grpc_fields == n_dict_fields


class TestCreateCollection:
    @pytest.fixture(scope="function")
    def collection_name(self):
        return f"test_collection_{random.randint(100000, 999999)}"

    def setup(self) -> None:
        self._real_time = grpc_testing.strict_real_time()
        self._real_time_channel = grpc_testing.channel(
            milvus_pb2.DESCRIPTOR.services_by_name.values(), self._real_time)
        self._servicer = milvus_pb2.DESCRIPTOR.services_by_name['MilvusService']
        self._milvus = Milvus(channel=self._real_time_channel, try_connect=False, pre_ping=False)

    def teardown(self) -> None:
        pass

    def test_create_collection(self, collection_name):
        id_field = {
            "name": "my_id",
            "type": DataType.INT64,
            "auto_id": True,
            "is_primary": True,
        }
        vector_field = {
            "name": "embedding",
            "type": DataType.FLOAT_VECTOR,
            "metric_type": "L2",
            "params": {"dim": "4"},
        }
        fields = {"fields": [id_field, vector_field]}
        future = self._milvus.create_collection(collection_name=collection_name, fields=fields, _async=True)

        invocation_metadata, request, rpc = self._real_time_channel.take_unary_unary(
            self._servicer.methods_by_name['CreateCollection']
        )
        rpc.send_initial_metadata(())
        rpc.terminate(common_pb2.Status(error_code=common_pb2.Success, reason="success"), (), grpc.StatusCode.OK, '')

        request_schema = schema_pb2.CollectionSchema()
        request_schema.ParseFromString(request.schema)

        assert request.collection_name == collection_name
        assert Fields.equal(request_schema.fields, fields["fields"])

        return_value = future.result()
        assert return_value.error_code == common_pb2.Success
        assert return_value.reason == "success"
