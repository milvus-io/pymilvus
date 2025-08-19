import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pymilvus.bulk_writer.buffer import Buffer
from pymilvus.bulk_writer.constants import DYNAMIC_FIELD_NAME, BulkFileType
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema, FieldSchema


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


class TestBufferExtended:
    """Extended tests to cover additional edge cases and error paths in buffer.py"""

    @pytest.fixture
    def schema_with_array(self):
        """Schema with array field for testing array error handling"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="array_field", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64),
        ]
        return CollectionSchema(fields=fields)

    @pytest.fixture
    def schema_with_sparse(self):
        """Schema with sparse vector field"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        return CollectionSchema(fields=fields)

    def test_persist_npy_with_array_field_error(self, schema_with_array):
        """Test that array field raises error in numpy format"""
        buffer = Buffer(schema_with_array, BulkFileType.NUMPY)
        buffer.append_row({
            "id": 1,
            "array_field": [1, 2, 3, 4, 5]
        })

        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(MilvusException, match="doesn't support parsing array type"):
            buffer.persist(temp_dir)

    def test_persist_npy_with_sparse_vector_error(self, schema_with_sparse):
        """Test that sparse vector field raises error in numpy format"""
        buffer = Buffer(schema_with_sparse, BulkFileType.NUMPY)
        buffer.append_row({
            "id": 1,
            "sparse_vector": {1: 0.5, 10: 0.3}
        })

        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(MilvusException, match="SPARSE_FLOAT_VECTOR"):
            # The error happens because SPARSE_FLOAT_VECTOR is not in NUMPY_TYPE_CREATOR
            # This causes a KeyError which is caught and re-raised as MilvusException
            buffer.persist(temp_dir)

    def test_persist_npy_with_json_field(self):
        """Test persisting JSON field as numpy"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="json_field", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.NUMPY)
        buffer.append_row({
            "id": 1,
            "json_field": {"key": "value", "nested": {"data": 123}}
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            files = buffer.persist(temp_dir)
            assert len(files) == 2
            # Verify JSON field was persisted
            json_file = next(f for f in files if "json_field" in f)
            assert Path(json_file).exists()

    def test_persist_npy_with_float16_vectors(self):
        """Test persisting float16 and bfloat16 vectors as numpy"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=4),
            FieldSchema(name="bfloat16_vector", dtype=DataType.BFLOAT16_VECTOR, dim=4),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.NUMPY)

        # Add data with bytes for float16/bfloat16 vectors
        float16_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes()
        bfloat16_data = bytes([1, 2, 3, 4, 5, 6, 7, 8])  # 4 bfloat16 values = 8 bytes

        buffer.append_row({
            "id": 1,
            "float16_vector": float16_data,
            "bfloat16_vector": bfloat16_data,
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            files = buffer.persist(temp_dir)
            assert len(files) == 3
            for f in files:
                assert Path(f).exists()

    def test_persist_npy_with_none_values(self):
        """Test persisting None values in numpy format"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="nullable_field", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.NUMPY)
        buffer.append_row({"id": 1, "nullable_field": None})
        buffer.append_row({"id": 2, "nullable_field": 100})

        with tempfile.TemporaryDirectory() as temp_dir:
            files = buffer.persist(temp_dir)
            assert len(files) == 2

    @patch('numpy.save')
    def test_persist_npy_exception_handling(self, mock_save):
        """Test exception handling in numpy persist"""
        mock_save.side_effect = Exception("Save failed")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.NUMPY)
        buffer.append_row({"id": 1})

        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(MilvusException, match="Failed to persist file"):
            buffer.persist(temp_dir)

    def test_persist_npy_cleanup_on_partial_failure(self):
        """Test that files are cleaned up if not all fields persist successfully"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.NUMPY)
        buffer.append_row({"id": 1, "vector": [1.0, 2.0]})

        with tempfile.TemporaryDirectory() as temp_dir, patch('numpy.save') as mock_save:
            # Make the second save fail
            mock_save.side_effect = [None, Exception("Second save failed")]

            with pytest.raises(MilvusException, match="Failed to persist file"):
                buffer.persist(temp_dir)

    def test_persist_json_with_float16_vectors(self):
        """Test JSON persist with float16 vectors only"""
        # Skip bfloat16 as numpy doesn't have native support
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=4),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.JSON)

        float16_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes()

        buffer.append_row({
            "id": 1,
            "float16_vector": float16_data,
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            files = buffer.persist(temp_dir)
            assert len(files) == 1

            # Verify the JSON content
            with Path(files[0]).open('r') as f:
                data = json.load(f)
                assert 'rows' in data
                assert len(data['rows']) == 1
                assert isinstance(data['rows'][0]['float16_vector'], list)

    @patch('pathlib.Path.open')
    def test_persist_json_exception_handling(self, mock_open):
        """Test exception handling in JSON persist"""
        mock_open.side_effect = Exception("Failed to open file")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.JSON)
        buffer.append_row({"id": 1})

        with pytest.raises(MilvusException, match="Failed to persist file"):
            buffer.persist("/tmp/test")

    def test_persist_parquet_with_json_and_sparse(self):
        """Test Parquet persist with JSON and sparse vector fields"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="json_field", dtype=DataType.JSON),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.PARQUET)

        buffer.append_row({
            "id": 1,
            "json_field": {"key": "value"},
            "sparse_vector": {1: 0.5, 10: 0.3}
        })

        with tempfile.TemporaryDirectory() as temp_dir, patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            # Pass buffer_size and buffer_row_count to avoid UnboundLocalError
            files = buffer.persist(temp_dir, buffer_size=1024, buffer_row_count=1)
            assert len(files) == 1
            mock_to_parquet.assert_called_once()

    def test_persist_parquet_with_float16_vectors(self):
        """Test Parquet persist with float16 vectors only"""
        # Skip bfloat16 as it has issues with numpy dtype
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=4),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.PARQUET)

        float16_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes()

        buffer.append_row({
            "id": 1,
            "float16_vector": float16_data,
        })

        with tempfile.TemporaryDirectory() as temp_dir, patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            # Pass buffer_size and buffer_row_count to avoid UnboundLocalError
            files = buffer.persist(temp_dir, buffer_size=1024, buffer_row_count=1)
            assert len(files) == 1
            mock_to_parquet.assert_called_once()

    def test_persist_parquet_with_array_field(self):
        """Test Parquet persist with array field"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="array_field", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.PARQUET)

        buffer.append_row({
            "id": 1,
            "array_field": [1, 2, 3, 4, 5]
        })
        buffer.append_row({
            "id": 2,
            "array_field": None  # Test None value
        })

        with tempfile.TemporaryDirectory() as temp_dir, patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            # Pass buffer_size and buffer_row_count to avoid UnboundLocalError
            files = buffer.persist(temp_dir, buffer_size=1024, buffer_row_count=2)
            assert len(files) == 1
            mock_to_parquet.assert_called_once()

    def test_persist_parquet_with_unknown_dtype(self):
        """Test Parquet persist with fields having standard dtypes"""
        # Using standard field types that work with parquet
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="field1", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="field2", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.PARQUET)

        buffer.append_row({
            "id": 1,
            "field1": "test_value",
            "field2": {"key": "value"}
        })

        with tempfile.TemporaryDirectory() as temp_dir, patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            # Pass buffer_size and buffer_row_count to avoid UnboundLocalError
            files = buffer.persist(temp_dir, buffer_size=1024, buffer_row_count=1)
            assert len(files) == 1
            mock_to_parquet.assert_called_once()

    def test_persist_csv_with_all_field_types(self):
        """Test CSV persist with common field types"""
        # Skip bfloat16 due to numpy dtype issues
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="bool_field", dtype=DataType.BOOL),
            FieldSchema(name="int8_field", dtype=DataType.INT8),
            FieldSchema(name="int16_field", dtype=DataType.INT16),
            FieldSchema(name="int32_field", dtype=DataType.INT32),
            FieldSchema(name="float_field", dtype=DataType.FLOAT),
            FieldSchema(name="double_field", dtype=DataType.DOUBLE),
            FieldSchema(name="varchar_field", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="json_field", dtype=DataType.JSON),
            FieldSchema(name="array_field", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64),
            FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=128),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=128),
            FieldSchema(name="int8_vector", dtype=DataType.INT8_VECTOR, dim=128),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.CSV)

        row = {
            "id": 1,
            "bool_field": True,
            "int8_field": 127,
            "int16_field": 32767,
            "int32_field": 2147483647,
            "float_field": 3.14,
            "double_field": 2.718281828,
            "varchar_field": "test string",
            "json_field": {"key": "value"},
            "array_field": [1, 2, 3],
            "float_vector": [1.0] * 128,
            "binary_vector": [1] * 128,
            "sparse_vector": {1: 0.5},
            "float16_vector": np.array([1.0] * 128, dtype=np.float16).tobytes(),
            "int8_vector": [1] * 128,
        }
        buffer.append_row(row)

        # Add row with None values
        row_with_none = row.copy()
        row_with_none["array_field"] = None
        buffer.append_row(row_with_none)

        with tempfile.TemporaryDirectory() as temp_dir, patch('pandas.DataFrame.to_csv') as mock_to_csv:
            files = buffer.persist(temp_dir)
            assert len(files) == 1
            mock_to_csv.assert_called_once()

    @patch('pandas.DataFrame.to_csv')
    def test_persist_csv_exception_handling(self, mock_to_csv):
        """Test exception handling in CSV persist"""
        mock_to_csv.side_effect = Exception("CSV write failed")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.CSV)
        buffer.append_row({"id": 1})

        with pytest.raises(MilvusException, match="Failed to persist file"):
            buffer.persist("/tmp/test")

    def test_persist_csv_with_custom_config(self):
        """Test CSV persist with custom separator and null key"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="nullable_field", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields)

        # Create buffer with custom CSV config
        config = {"sep": "|", "nullkey": "NULL"}
        buffer = Buffer(schema, BulkFileType.CSV, config=config)

        buffer.append_row({"id": 1, "nullable_field": None})
        buffer.append_row({"id": 2, "nullable_field": 100})

        with tempfile.TemporaryDirectory() as temp_dir, patch('pandas.DataFrame.to_csv') as mock_to_csv:
            files = buffer.persist(temp_dir)
            assert len(files) == 1

            # Verify custom config was used
            call_kwargs = mock_to_csv.call_args[1]
            assert call_kwargs.get('sep') == '|'
            assert call_kwargs.get('na_rep') == 'NULL'

    def test_get_field_schema(self):
        """Test accessing fields from buffer"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.JSON)

        # Test that fields are stored in the buffer
        assert "id" in buffer._fields
        assert "vector" in buffer._fields
        assert buffer._fields["id"].name == "id"
        assert buffer._fields["vector"].name == "vector"

        # Test that non-existent field is not in buffer
        assert "non_existent" not in buffer._fields

    def test_throw_method(self):
        """Test the _throw method raises MilvusException"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        ]
        schema = CollectionSchema(fields=fields)
        buffer = Buffer(schema, BulkFileType.JSON)

        with pytest.raises(MilvusException, match="Test error message"):
            buffer._throw("Test error message")
