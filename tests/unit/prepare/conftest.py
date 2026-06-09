"""Shared fixtures and test data for Prepare tests."""

from typing import Optional

import numpy as np
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType
from pymilvus.orm.schema import FunctionScore, LexicalHighlighter, StructFieldSchema

# ============================================================
# Common Fixtures
# ============================================================


@pytest.fixture
def basic_schema():
    """Basic schema with primary key and vector field."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
        ]
    )


@pytest.fixture
def basic_schema_auto_id():
    """Schema with auto_id enabled."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
        ]
    )


@pytest.fixture
def schema_with_nullable():
    """Schema with nullable and default fields."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
            FieldSchema("nullable_field", DataType.VARCHAR, nullable=True, max_length=100),
            FieldSchema("default_field", DataType.INT64, default_value=0),
        ]
    )


@pytest.fixture
def schema_with_dynamic():
    """Schema with dynamic field enabled."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
        ],
        enable_dynamic_field=True,
    )


@pytest.fixture
def schema_with_function():
    """Schema with function output field."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=1000),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=128, is_function_output=True),
        ],
        functions=[
            Function(
                "text_embedding",
                FunctionType.TEXTEMBEDDING,
                input_field_names=["text"],
                output_field_names=["embedding"],
            )
        ],
    )


@pytest.fixture
def struct_field_schema():
    """Struct field schema for testing."""
    struct = StructFieldSchema()
    struct.name = "metadata"
    struct.add_field("score", DataType.FLOAT)
    struct.add_field("label", DataType.VARCHAR, max_length=100)
    struct.max_capacity = 10
    return struct


@pytest.fixture
def schema_with_struct(struct_field_schema):
    """Schema with struct array field."""
    schema = CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4),
        ]
    )
    schema.add_struct_field(struct_field_schema)
    return schema


@pytest.fixture
def sample_vectors():
    """Sample float vectors for testing."""
    return [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


@pytest.fixture
def sample_rows(sample_vectors):
    """Sample row data for insert."""
    return [
        {"pk": 1, "vector": sample_vectors[0]},
        {"pk": 2, "vector": sample_vectors[1]},
    ]


# ============================================================
# Test Data Tables
# ============================================================

INVALID_COLLECTION_NAMES = [
    pytest.param(None, id="none"),
    pytest.param("", id="empty"),
    pytest.param(123, id="int"),
    pytest.param([], id="list"),
    pytest.param({}, id="dict"),
]

INVALID_PARTITION_NAMES = [
    pytest.param(None, id="none"),
    pytest.param("", id="empty"),
    pytest.param(123, id="int"),
]

INVALID_INDEX_NAMES = [
    pytest.param(None, id="none"),
    pytest.param("", id="empty"),
]

VALID_DATA_TYPES = [
    pytest.param(DataType.INT8, id="int8"),
    pytest.param(DataType.INT16, id="int16"),
    pytest.param(DataType.INT32, id="int32"),
    pytest.param(DataType.INT64, id="int64"),
    pytest.param(DataType.FLOAT, id="float"),
    pytest.param(DataType.DOUBLE, id="double"),
    pytest.param(DataType.VARCHAR, id="varchar"),
    pytest.param(DataType.BOOL, id="bool"),
]

VECTOR_TYPES = [
    pytest.param(DataType.FLOAT_VECTOR, id="float_vector"),
    pytest.param(DataType.BINARY_VECTOR, id="binary_vector"),
    pytest.param(DataType.FLOAT16_VECTOR, id="float16_vector"),
    pytest.param(DataType.BFLOAT16_VECTOR, id="bfloat16_vector"),
    pytest.param(DataType.INT8_VECTOR, id="int8_vector"),
    pytest.param(DataType.SPARSE_FLOAT_VECTOR, id="sparse_float_vector"),
]

CONSISTENCY_LEVELS = [
    pytest.param(0, id="strong"),
    pytest.param(1, id="session"),
    pytest.param(2, id="bounded"),
    pytest.param(3, id="eventually"),
]

SEARCH_OFFSET_CASES = [
    pytest.param(0, id="zero"),
    pytest.param(10, id="positive"),
    pytest.param(100, id="large"),
]


# ============================================================
# Helper Functions
# ============================================================


def make_fields_info(schema: CollectionSchema):
    """Convert CollectionSchema to fields_info dict format."""
    return schema.to_dict()["fields"]


def make_struct_fields_info(schema: CollectionSchema):
    """Extract struct fields info from CollectionSchema."""
    schema_dict = schema.to_dict()
    return schema_dict.get("struct_fields", [])


def generate_vectors(count: int, dim: int, dtype: str = "float32") -> list:
    """Generate random vectors of specified type."""
    rng = np.random.default_rng(seed=42)
    if dtype == "float32":
        return rng.random((count, dim)).astype(np.float32).tolist()
    if dtype == "float16":
        return rng.random((count, dim)).astype(np.float16)
    if dtype == "bfloat16":
        # numpy doesn't have bfloat16, simulate with float32
        return rng.random((count, dim)).astype(np.float32)
    if dtype == "int8":
        return rng.integers(-128, 127, size=(count, dim), dtype=np.int8)
    if dtype == "binary":
        return rng.bytes(count * (dim // 8))
    return rng.random((count, dim)).tolist()


def make_function(
    name: str = "test_func",
    func_type: FunctionType = FunctionType.TEXTEMBEDDING,
    input_fields: Optional[list] = None,
    output_fields: Optional[list] = None,
    params: Optional[dict] = None,
) -> Function:
    """Create a Function for testing."""
    return Function(
        name=name,
        function_type=func_type,
        input_field_names=input_fields or ["text"],
        output_field_names=output_fields or ["embedding"],
        params=params or {},
    )


def make_function_score(
    functions: Optional[list] = None, params: Optional[dict] = None
) -> FunctionScore:
    """Create a FunctionScore for testing."""
    return FunctionScore(
        functions=functions or [],
        params=params or {},
    )


def make_highlighter(
    pre_tags: Optional[list] = None, post_tags: Optional[list] = None
) -> LexicalHighlighter:
    """Create a LexicalHighlighter for testing."""
    return LexicalHighlighter(
        pre_tags=pre_tags or ["<em>"],
        post_tags=post_tags or ["</em>"],
    )
