# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.client.types import FunctionType
from pymilvus.orm.schema import Function

# ── Helpers / factories ───────────────────────────────────────────────────────


def _make_schema(
    output_type=DataType.BINARY_VECTOR,
    output_dim=512,
    input_type=DataType.VARCHAR,
    check_fields=False,
):
    input_field = (
        FieldSchema("text", input_type, max_length=65535)
        if input_type == DataType.VARCHAR
        else FieldSchema("text", input_type)
    )
    return CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, is_primary=True),
            input_field,
            FieldSchema("minhash_vector", output_type, dim=output_dim),
        ],
        check_fields=check_fields,
    )


def _make_minhash_func(
    name="text_to_minhash", input_fields=None, output_fields=None, params=None, description=None
):
    if input_fields is None:
        input_fields = ["text"]
    if output_fields is None:
        output_fields = ["minhash_vector"]
    kwargs = {
        "name": name,
        "function_type": FunctionType.MINHASH,
        "input_field_names": input_fields,
        "output_field_names": output_fields,
    }
    if params is not None:
        kwargs["params"] = params
    if description is not None:
        kwargs["description"] = description
    return Function(**kwargs)


# ── TestMinHashFunctionType ───────────────────────────────────────────────────


class TestMinHashFunctionType:
    """Test MinHash FunctionType is correctly defined."""

    def test_minhash_function_type_value(self):
        """Test MINHASH enum value is 4."""
        assert FunctionType.MINHASH == 4
        assert FunctionType.MINHASH.value == 4

    def test_minhash_function_type_in_enum(self):
        """Test MINHASH is a valid FunctionType."""
        assert hasattr(FunctionType, "MINHASH")
        assert FunctionType.MINHASH in FunctionType


# ── TestMinHashFunction ───────────────────────────────────────────────────────


class TestMinHashFunction:
    """Test MinHash Function creation and validation."""

    @pytest.fixture
    def valid_schema(self):
        return _make_schema()

    @pytest.fixture
    def invalid_dim_schema(self):
        return _make_schema(output_dim=500)  # Not multiple of 32

    @pytest.fixture
    def invalid_input_type_schema(self):
        return _make_schema(input_type=DataType.INT64)

    @pytest.fixture
    def invalid_output_type_schema(self):
        return _make_schema(output_type=DataType.FLOAT_VECTOR)

    def test_create_minhash_function(self):
        """Test creating a MinHash function with valid parameters."""
        func = _make_minhash_func(
            params={
                "num_hashes": 16,
                "shingle_size": 3,
                "hash_function": "xxhash64",
                "token_level": "word",
            },
        )
        assert func.name == "text_to_minhash"
        assert func.type == FunctionType.MINHASH
        assert func.input_field_names == ["text"]
        assert func.output_field_names == ["minhash_vector"]
        assert func.params["num_hashes"] == 16
        assert func.params["shingle_size"] == 3

    def test_minhash_function_to_dict(self):
        """Test MinHash function serialization to dict."""
        func = _make_minhash_func(
            name="minhash_func",
            input_fields=["content"],
            output_fields=["signature"],
            description="MinHash signature generator",
            params={"num_hashes": 8, "shingle_size": 5},
        )
        result = func.to_dict()
        assert result["name"] == "minhash_func"
        assert result["type"] == FunctionType.MINHASH
        assert result["input_field_names"] == ["content"]
        assert result["output_field_names"] == ["signature"]
        assert result["description"] == "MinHash signature generator"
        assert result["params"]["num_hashes"] == 8

    def test_minhash_function_verify_valid_schema(self, valid_schema):
        """Test MinHash function verification with valid schema."""
        _make_minhash_func().verify(valid_schema)  # Should not raise

    def test_minhash_function_verify_skips_validation(self, valid_schema):
        """Test MinHash function verify does not perform client-side validation."""
        # Multiple inputs - server validates
        _make_minhash_func(input_fields=["text", "extra_text"]).verify(valid_schema)
        # Multiple outputs - server validates
        _make_minhash_func(output_fields=["vec1", "vec2"]).verify(valid_schema)

    def test_minhash_function_verify_no_type_validation(
        self, invalid_input_type_schema, invalid_output_type_schema
    ):
        """Test MinHash function verify does not validate field types."""
        func = _make_minhash_func()
        func.verify(invalid_input_type_schema)  # Should not raise
        func.verify(invalid_output_type_schema)  # Should not raise

    def test_minhash_function_verify_no_dim_validation(self, invalid_dim_schema):
        """Test MinHash function verify does not validate dimension."""
        _make_minhash_func().verify(invalid_dim_schema)  # Should not raise


# ── TestMinHashSchemaIntegration ──────────────────────────────────────────────


class TestMinHashSchemaIntegration:
    """Test MinHash function integration with CollectionSchema."""

    def test_add_minhash_function_to_schema(self):
        """Test adding MinHash function to CollectionSchema."""
        schema = CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("minhash_vector", DataType.BINARY_VECTOR, dim=256),
            ]
        )
        schema.add_function(_make_minhash_func(params={"num_hashes": 8, "shingle_size": 3}))
        assert len(schema.functions) == 1
        assert schema.functions[0].name == "text_to_minhash"
        assert schema.functions[0].type == FunctionType.MINHASH

    def test_schema_with_minhash_function_to_dict(self):
        """Test CollectionSchema with MinHash function serialization."""
        func = _make_minhash_func(name="minhash_func", output_fields=["signature"])
        schema = CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("signature", DataType.BINARY_VECTOR, dim=512),
            ],
            functions=[func],
        )
        result = schema.to_dict()
        assert "functions" in result
        assert len(result["functions"]) == 1
        assert result["functions"][0]["type"] == FunctionType.MINHASH

    def test_minhash_output_field_marked_as_function_output(self):
        """Test that MinHash output field is marked as function output."""
        func = _make_minhash_func(name="minhash_func", output_fields=["signature"])
        schema = CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("signature", DataType.BINARY_VECTOR, dim=512),
            ],
            functions=[func],
        )
        output_field = next(f for f in schema.fields if f.name == "signature")
        assert output_field.is_function_output is True


# ── TestMinHashFunctionParams ─────────────────────────────────────────────────


class TestMinHashFunctionParams:
    """Test MinHash function parameter handling."""

    def test_minhash_function_default_params(self):
        """Test MinHash function with minimal params."""
        func = _make_minhash_func(name="minhash", output_fields=["vector"])
        assert func.params == {}

    def test_minhash_function_full_params(self):
        """Test MinHash function with all parameters."""
        params = {
            "num_hashes": 32,
            "shingle_size": 5,
            "hash_function": "sha1",
            "token_level": "char",
            "seed": 42,
        }
        func = _make_minhash_func(name="minhash", output_fields=["vector"], params=params)
        assert func.params["num_hashes"] == 32
        assert func.params["shingle_size"] == 5
        assert func.params["hash_function"] == "sha1"
        assert func.params["token_level"] == "char"
        assert func.params["seed"] == 42

    def test_minhash_function_equality(self):
        """Test MinHash function equality comparison."""
        func1 = _make_minhash_func(
            name="minhash", output_fields=["vector"], params={"num_hashes": 16}
        )
        func2 = _make_minhash_func(
            name="minhash", output_fields=["vector"], params={"num_hashes": 16}
        )
        func3 = _make_minhash_func(
            name="minhash", output_fields=["vector"], params={"num_hashes": 32}
        )
        assert func1 == func2
        assert func1 != func3
