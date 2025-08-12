import json
import pytest
import numpy as np
import tempfile
from unittest.mock import patch
from pathlib import Path

from pymilvus.bulk_writer.buffer import Buffer
from pymilvus.bulk_writer.constants import BulkFileType, DYNAMIC_FIELD_NAME
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema
from pymilvus.exceptions import MilvusException


class TestBuffer:
    @pytest.fixture
    def simple_schema(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        ]
        return CollectionSchema(fields=fields)

    @pytest.fixture
    def complex_schema(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="bool_field", dtype=DataType.BOOL),
            FieldSchema(name="int8_field", dtype=DataType.INT8),
            FieldSchema(name="int16_field", dtype=DataType.INT16),
            FieldSchema(name="int32_field", dtype=DataType.INT32),
            FieldSchema(name="float_field", dtype=DataType.FLOAT),
            FieldSchema(name="double_field", dtype=DataType.DOUBLE),
            FieldSchema(name="json_field", dtype=DataType.JSON),
            FieldSchema(name="array_field", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=128),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=128),
            FieldSchema(name="bfloat16_vector", dtype=DataType.BFLOAT16_VECTOR, dim=128),
            FieldSchema(name="int8_vector", dtype=DataType.INT8_VECTOR, dim=128),
        ]
        return CollectionSchema(fields=fields)

    @pytest.fixture
    def dynamic_schema(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        ]
        return CollectionSchema(fields=fields, enable_dynamic_field=True)

    def test_buffer_init_numpy(self, simple_schema: CollectionSchema):
        buffer = Buffer(simple_schema, BulkFileType.NUMPY)
        assert buffer._file_type == BulkFileType.NUMPY
        assert len(buffer._buffer) == 3
        assert "id" in buffer._buffer
        assert "vector" in buffer._buffer
        assert "text" in buffer._buffer

    def test_buffer_init_json(self, simple_schema: CollectionSchema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        assert buffer._file_type == BulkFileType.JSON
        assert len(buffer._buffer) == 3

    def test_buffer_init_parquet(self, simple_schema: CollectionSchema):
        buffer = Buffer(simple_schema, BulkFileType.PARQUET)
        assert buffer._file_type == BulkFileType.PARQUET
        assert len(buffer._buffer) == 3

    def test_buffer_init_csv(self, simple_schema: CollectionSchema):
        buffer = Buffer(simple_schema, BulkFileType.CSV)
        assert buffer._file_type == BulkFileType.CSV
        assert len(buffer._buffer) == 3

    def test_buffer_init_with_dynamic_field(self, dynamic_schema: CollectionSchema):
        buffer = Buffer(dynamic_schema, BulkFileType.JSON)
        assert DYNAMIC_FIELD_NAME in buffer._buffer
        assert len(buffer._buffer) == 3  # id, vector, $meta

    def test_buffer_init_auto_id_field(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.JSON)
        assert "id" not in buffer._buffer
        assert "vector" in buffer._buffer

    def test_buffer_init_function_output_field(self):
        # Test that fields marked with is_function_output=True are skipped in buffer
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="output", dtype=DataType.FLOAT),
        ]
        # Manually mark the output field as function output
        # (This would normally be done by CollectionSchema when functions are provided)
        fields[2].is_function_output = True

        schema = CollectionSchema(fields=fields)

        # Verify that the field is marked as function output
        output_field = None
        for field in schema.fields:
            if field.name == "output":
                output_field = field
                break

        assert output_field is not None
        assert output_field.is_function_output is True

        # Create buffer and verify function output field is NOT included
        buffer = Buffer(schema, BulkFileType.JSON)
        assert "output" not in buffer._buffer
        assert "id" in buffer._buffer
        assert "vector" in buffer._buffer

    def test_buffer_init_only_auto_id_field(self):
        # Test with only an auto_id primary field (which gets skipped in buffer)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        ]
        schema = CollectionSchema(fields=fields)
        # Buffer should raise exception because no fields remain after filtering
        with pytest.raises(MilvusException, match="fields list is empty"):
            Buffer(schema, BulkFileType.JSON)

    def test_append_row_simple(self, simple_schema: CollectionSchema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        row = {
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test text"
        }
        buffer.append_row(row)
        assert buffer.row_count == 1
        assert buffer._buffer["id"][0] == 1
        assert buffer._buffer["vector"][0] == [1.0] * 128
        assert buffer._buffer["text"][0] == "test text"

    def test_append_row_with_numpy(self, simple_schema: CollectionSchema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        row = {
            "id": np.int64(1),
            "vector": np.array([1.0] * 128, dtype=np.float32),
            "text": "test text"
        }
        buffer.append_row(row)
        assert buffer.row_count == 1
        assert buffer._buffer["id"][0] == 1
        assert len(buffer._buffer["vector"][0]) == 128

    def test_append_row_dynamic_field(self, dynamic_schema: CollectionSchema):
        buffer = Buffer(dynamic_schema, BulkFileType.JSON)
        row = {
            "id": 1,
            "vector": [1.0] * 128,
            "extra_field": "extra_value",
            "another_field": 123
        }
        buffer.append_row(row)
        assert buffer.row_count == 1
        assert buffer._buffer[DYNAMIC_FIELD_NAME][0]["extra_field"] == "extra_value"
        assert buffer._buffer[DYNAMIC_FIELD_NAME][0]["another_field"] == 123

    def test_append_row_dynamic_field_dict(self, dynamic_schema: CollectionSchema):
        buffer = Buffer(dynamic_schema, BulkFileType.JSON)
        row = {
            "id": 1,
            "vector": [1.0] * 128,
            DYNAMIC_FIELD_NAME: {"field1": "value1", "field2": 2}
        }
        buffer.append_row(row)
        assert buffer.row_count == 1
        assert buffer._buffer[DYNAMIC_FIELD_NAME][0]["field1"] == "value1"
        assert buffer._buffer[DYNAMIC_FIELD_NAME][0]["field2"] == 2

    def test_append_row_invalid_dynamic_field(self, dynamic_schema: CollectionSchema):
        buffer = Buffer(dynamic_schema, BulkFileType.JSON)
        row = {
            "id": 1,
            "vector": [1.0] * 128,
            DYNAMIC_FIELD_NAME: "not_a_dict"
        }
        with pytest.raises(MilvusException):
            buffer.append_row(row)

    def test_persist_npy(self, simple_schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = Buffer(simple_schema, BulkFileType.NUMPY)
            buffer.append_row({
                "id": 1,
                "vector": [1.0] * 128,
                "text": "test"
            })

            files = buffer.persist(temp_dir)
            assert len(files) == 3
            # Check files were created
            for f in files:
                pf = Path(f)
                assert pf.exists()
                assert pf.suffix == '.npy'

    def test_persist_json_rows(self, simple_schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = Buffer(simple_schema, BulkFileType.JSON)
            buffer.append_row({
                "id": 1,
                "vector": [1.0] * 128,
                "text": "test"
            })

            files = buffer.persist(temp_dir)
            assert len(files) == 1
            assert files[0].endswith(".json")
            # Verify file was created and contains correct data
            file = Path(files[0])
            assert file.exists()
            with file.open('r') as f:
                data = json.load(f)
                assert 'rows' in data
                assert len(data['rows']) == 1
                assert data['rows'][0]['id'] == 1

    @patch('pandas.DataFrame.to_parquet')
    def test_persist_parquet(self, mock_to_parquet, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.PARQUET)
        buffer.append_row({
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test"
        })

        files = buffer.persist("/tmp/test")
        assert len(files) == 1
        assert files[0].endswith(".parquet")
        assert mock_to_parquet.called

    @patch('pandas.DataFrame.to_csv')
    def test_persist_csv(self, mock_to_csv, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.CSV)
        buffer.append_row({
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test"
        })

        with patch.object(buffer, '_persist_csv', return_value=["/tmp/test.csv"]):
            files = buffer.persist("/tmp/test")
            assert len(files) == 1
            assert files[0].endswith(".csv")

    def test_persist_unsupported_type(self, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        buffer._file_type = 999  # Invalid type
        buffer.append_row({"id": 1, "vector": [1.0] * 128, "text": "test"})

        with pytest.raises(MilvusException):
            buffer.persist("/tmp/test")

    def test_persist_mismatched_row_count(self, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        buffer._buffer["id"].append(1)
        buffer._buffer["vector"].append([1.0] * 128)
        # Missing text field value

        with pytest.raises(MilvusException):
            buffer.persist("/tmp/test")

    def test_row_count_empty(self, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        assert buffer.row_count == 0

    def test_row_count_multiple(self, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)
        for i in range(5):
            buffer.append_row({
                "id": i,
                "vector": [1.0] * 128,
                "text": f"test {i}"
            })
        assert buffer.row_count == 5

    def test_complex_data_types(self, complex_schema):
        buffer = Buffer(complex_schema, BulkFileType.JSON)
        row = {
            "id": 1,
            "bool_field": True,
            "int8_field": 127,
            "int16_field": 32767,
            "int32_field": 2147483647,
            "float_field": 3.14,
            "double_field": 2.718281828,
            "json_field": {"key": "value"},
            "array_field": [1, 2, 3, 4, 5],
            "vector": [1.0] * 128,
            "binary_vector": [1] * 128,
            "sparse_vector": {1: 0.5, 10: 0.3},
            "float16_vector": bytes([1] * 32),
            "bfloat16_vector": bytes([1] * 32),
            "int8_vector": [1] * 128,
        }
        buffer.append_row(row)
        assert buffer.row_count == 1

    @pytest.mark.xfail(reason='numpy.dtype("bfloat16") is not supported')
    def test_persist_json_with_special_vectors(self, complex_schema):
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = Buffer(complex_schema, BulkFileType.JSON)
            row = {
                "id": 1,
                "bool_field": True,
                "int8_field": 127,
                "int16_field": 32767,
                "int32_field": 2147483647,
                "float_field": 3.14,
                "double_field": 2.718281828,
                "json_field": {"key": "value"},
                "array_field": [1, 2, 3],
                "vector": [1.0] * 128,
                "binary_vector": [1] * 128,
                "sparse_vector": {1: 0.5},
                "float16_vector": bytes([1] * 32),
                "bfloat16_vector": bytes([1] * 32),
                "int8_vector": [1] * 128,
            }
            buffer.append_row(row)

            files = buffer.persist(temp_dir)
            assert len(files) == 1
            assert Path(files[0]).exists()

    def test_persist_with_kwargs(self, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.PARQUET)
        buffer.append_row({
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test"
        })

        with patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            files = buffer.persist(
                "/tmp/test",
                buffer_size=1024,
                buffer_row_count=1,
                row_group_bytes=32 * 1024 * 1024
            )
            assert len(files) == 1
            mock_to_parquet.assert_called_once()

    def test_raw_obj_conversion(self, simple_schema):
        buffer = Buffer(simple_schema, BulkFileType.JSON)

        # Test numpy array conversion
        arr = np.array([1, 2, 3])
        result = buffer._raw_obj(arr)
        assert result == [1, 2, 3]

        # Test numpy scalar conversion
        scalar = np.int64(42)
        result = buffer._raw_obj(scalar)
        assert result == 42

        # Test regular object
        obj = {"key": "value"}
        result = buffer._raw_obj(obj)
        assert result == {"key": "value"}
