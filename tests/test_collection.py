import logging
from unittest import mock

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

LOGGER = logging.getLogger(__name__)


class TestCollections:
    def test_collection_by_DataFrame(self):
        coll_name = "ut_collection_test_collection_by_DataFrame"
        fields = [
            FieldSchema("int64", DataType.INT64),
            FieldSchema("float", DataType.FLOAT),
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=128),
            FieldSchema("binary_vector", DataType.BINARY_VECTOR, dim=128),
            FieldSchema("float16_vector", DataType.FLOAT16_VECTOR, dim=128),
            FieldSchema("bfloat16_vector", DataType.BFLOAT16_VECTOR, dim=128),
            FieldSchema("int8_vector", DataType.INT8_VECTOR, dim=128),
            FieldSchema("timestamptz", DataType.TIMESTAMPTZ),
        ]

        prefix = "pymilvus.client.grpc_handler.GrpcHandler"

        collection_schema = CollectionSchema(fields, primary_field="int64")
        with mock.patch(f"{prefix}.__init__", return_value=None), mock.patch(f"{prefix}._wait_for_channel_ready", return_value=None):
                connections.connect(keep_alive=False)

        with mock.patch(f"{prefix}.create_collection", return_value=None), mock.patch(f"{prefix}.has_collection", return_value=False):
                collection = Collection(name=coll_name, schema=collection_schema)

        with mock.patch(f"{prefix}.create_collection", return_value=None), mock.patch(f"{prefix}.has_collection", return_value=True), mock.patch(f"{prefix}.describe_collection", return_value=collection_schema.to_dict()):
                    collection = Collection(name=coll_name)

        with mock.patch(f"{prefix}.drop_collection", return_value=None), mock.patch(f"{prefix}.list_indexes", return_value=[]), mock.patch(f"{prefix}.release_collection", return_value=None):
                    collection.drop()

        with mock.patch(f"{prefix}.close", return_value=None):
            connections.disconnect("default")
