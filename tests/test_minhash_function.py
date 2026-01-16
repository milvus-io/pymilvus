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
from pymilvus.exceptions import ParamError
from pymilvus.orm.schema import Function


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


class TestMinHashFunction:
    """Test MinHash Function creation and validation."""

    @pytest.fixture
    def valid_schema(self):
        """Create a valid schema with VARCHAR input and BINARY_VECTOR output."""
        return CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("minhash_vector", DataType.BINARY_VECTOR, dim=512),  # 512 = 16 * 32
            ],
            check_fields=False,
        )

    @pytest.fixture
    def invalid_dim_schema(self):
        """Create a schema with invalid dim (not multiple of 32)."""
        return CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("minhash_vector", DataType.BINARY_VECTOR, dim=500),  # Not multiple of 32
            ],
            check_fields=False,
        )

    @pytest.fixture
    def invalid_input_type_schema(self):
        """Create a schema with non-VARCHAR input field."""
        return CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.INT64),  # Wrong type
                FieldSchema("minhash_vector", DataType.BINARY_VECTOR, dim=512),
            ],
            check_fields=False,
        )

    @pytest.fixture
    def invalid_output_type_schema(self):
        """Create a schema with non-BINARY_VECTOR output field."""
        return CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("minhash_vector", DataType.FLOAT_VECTOR, dim=512),  # Wrong type
            ],
            check_fields=False,
        )

    def test_create_minhash_function(self):
        """Test creating a MinHash function with valid parameters."""
        func = Function(
            name="text_to_minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_vector"],
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
        func = Function(
            name="minhash_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["content"],
            output_field_names=["signature"],
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
        func = Function(
            name="text_to_minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_vector"],
        )

        # Should not raise any exception
        func.verify(valid_schema)

    def test_minhash_function_verify_invalid_input_count(self, valid_schema):
        """Test MinHash function with multiple input fields raises error."""
        func = Function(
            name="invalid_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text", "extra_text"],  # Multiple inputs
            output_field_names=["minhash_vector"],
        )

        with pytest.raises(ParamError) as exc_info:
            func.verify(valid_schema)
        assert "exact 1 input and 1 output" in str(exc_info.value)

    def test_minhash_function_verify_invalid_output_count(self, valid_schema):
        """Test MinHash function with multiple output fields raises error."""
        func = Function(
            name="invalid_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["vec1", "vec2"],  # Multiple outputs
        )

        with pytest.raises(ParamError) as exc_info:
            func.verify(valid_schema)
        assert "exact 1 input and 1 output" in str(exc_info.value)

    def test_minhash_function_verify_invalid_input_type(self, invalid_input_type_schema):
        """Test MinHash function with non-VARCHAR input raises error."""
        func = Function(
            name="invalid_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_vector"],
        )

        with pytest.raises(ParamError) as exc_info:
            func.verify(invalid_input_type_schema)
        assert "VARCHAR" in str(exc_info.value)

    def test_minhash_function_verify_invalid_output_type(self, invalid_output_type_schema):
        """Test MinHash function with non-BINARY_VECTOR output raises error."""
        func = Function(
            name="invalid_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_vector"],
        )

        with pytest.raises(ParamError) as exc_info:
            func.verify(invalid_output_type_schema)
        assert "BINARY_VECTOR" in str(exc_info.value)

    def test_minhash_function_verify_invalid_dim(self, invalid_dim_schema):
        """Test MinHash function with invalid dim (not multiple of 32) raises error."""
        func = Function(
            name="invalid_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_vector"],
        )

        with pytest.raises(ParamError) as exc_info:
            func.verify(invalid_dim_schema)
        assert "multiple of 32" in str(exc_info.value)


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

        minhash_func = Function(
            name="text_to_minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["minhash_vector"],
            params={"num_hashes": 8, "shingle_size": 3},
        )

        schema.add_function(minhash_func)

        assert len(schema.functions) == 1
        assert schema.functions[0].name == "text_to_minhash"
        assert schema.functions[0].type == FunctionType.MINHASH

    def test_schema_with_minhash_function_to_dict(self):
        """Test CollectionSchema with MinHash function serialization."""
        minhash_func = Function(
            name="minhash_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["signature"],
        )

        schema = CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("signature", DataType.BINARY_VECTOR, dim=512),
            ],
            functions=[minhash_func],
        )

        result = schema.to_dict()
        assert "functions" in result
        assert len(result["functions"]) == 1
        assert result["functions"][0]["type"] == FunctionType.MINHASH

    def test_minhash_output_field_marked_as_function_output(self):
        """Test that MinHash output field is marked as function output."""
        minhash_func = Function(
            name="minhash_func",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["signature"],
        )

        schema = CollectionSchema(
            fields=[
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("signature", DataType.BINARY_VECTOR, dim=512),
            ],
            functions=[minhash_func],
        )

        # Find the output field
        output_field = next(f for f in schema.fields if f.name == "signature")
        assert output_field.is_function_output is True


class TestMinHashFunctionParams:
    """Test MinHash function parameter handling."""

    def test_minhash_function_default_params(self):
        """Test MinHash function with minimal params."""
        func = Function(
            name="minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["vector"],
        )

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

        func = Function(
            name="minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["vector"],
            params=params,
        )

        assert func.params["num_hashes"] == 32
        assert func.params["shingle_size"] == 5
        assert func.params["hash_function"] == "sha1"
        assert func.params["token_level"] == "char"
        assert func.params["seed"] == 42

    def test_minhash_function_equality(self):
        """Test MinHash function equality comparison."""
        func1 = Function(
            name="minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["vector"],
            params={"num_hashes": 16},
        )

        func2 = Function(
            name="minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["vector"],
            params={"num_hashes": 16},
        )

        func3 = Function(
            name="minhash",
            function_type=FunctionType.MINHASH,
            input_field_names=["text"],
            output_field_names=["vector"],
            params={"num_hashes": 32},  # Different
        )

        assert func1 == func2
        assert func1 != func3
