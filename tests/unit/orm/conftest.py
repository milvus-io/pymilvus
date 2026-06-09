"""Common fixtures for ORM tests."""

from unittest import mock

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, connections

GRPC_PREFIX = "pymilvus.client.grpc_handler.GrpcHandler"


@pytest.fixture
def basic_schema():
    """Basic schema with primary key and vector field."""
    return CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
    )


@pytest.fixture
def schema_with_all_field_types():
    """Schema with various field types."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("varchar_field", DataType.VARCHAR, max_length=256),
            FieldSchema("float_vec", DataType.FLOAT_VECTOR, dim=128),
            FieldSchema("binary_vec", DataType.BINARY_VECTOR, dim=128),
            FieldSchema("sparse_vec", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("json_field", DataType.JSON),
            FieldSchema(
                "array_field",
                DataType.ARRAY,
                element_type=DataType.INT64,
                max_capacity=100,
            ),
        ]
    )


@pytest.fixture
def raw_dict_schema():
    """Raw dict for schema construction."""
    return {
        "description": "Test collection",
        "enable_dynamic_field": True,
        "fields": [
            {
                "name": "id",
                "description": "Primary key",
                "type": DataType.INT64,
                "is_primary": True,
                "auto_id": False,
            },
            {
                "name": "vec",
                "description": "Vector field",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 128},
            },
        ],
    }


@pytest.fixture
def mock_grpc_connect():
    """Mock gRPC init + channel ready so connections.connect() succeeds without a server."""
    with mock.patch(f"{GRPC_PREFIX}.__init__", return_value=None) as m_init, mock.patch(
        f"{GRPC_PREFIX}._wait_for_channel_ready", return_value=None
    ) as m_ready:
        yield m_init, m_ready


@pytest.fixture
def mock_grpc_close():
    """Mock gRPC close for disconnect/remove_connection."""
    with mock.patch(f"{GRPC_PREFIX}.close", return_value=None) as m_close:
        yield m_close


@pytest.fixture(autouse=True)
def _cleanup_connections():
    """Clean up any connections created during tests to avoid cross-test pollution."""
    yield
    # Always reset to a pristine default config (no db_name) to avoid
    # duplicate-kwarg errors when a subsequent test calls connections.connect().
    connections._alias_handlers.clear()
    connections._alias_config.clear()
    connections.add_connection(default={"address": "localhost:19530", "user": ""})
