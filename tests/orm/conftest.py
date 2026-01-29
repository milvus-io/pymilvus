"""Common fixtures for ORM tests."""

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema


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
