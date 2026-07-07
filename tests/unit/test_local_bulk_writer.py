import csv
import json
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import ml_dtypes
import numpy as np
import pytest
from pymilvus.bulk_writer.constants import MB, BulkFileType
from pymilvus.bulk_writer.local_bulk_writer import LocalBulkWriter
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client import MilvusClient
from pymilvus.orm.schema import CollectionSchema, FieldSchema, StructFieldSchema


class TestLocalBulkWriter:
    @pytest.fixture
    def simple_schema(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        ]
        return CollectionSchema(fields=fields)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory that gets cleaned up after test"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        if Path(temp_path).exists():
            shutil.rmtree(temp_path)

    @pytest.fixture
    def writer(self, simple_schema, temp_dir):
        return LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        )

    def _mock_row_count(self, writer, value):
        """Context manager: mock buffer row_count property."""
        return patch.object(
            type(writer._buffer), "row_count", new_callable=PropertyMock, return_value=value
        )

    @patch("uuid.uuid4")
    def test_init(self, mock_uuid, simple_schema, temp_dir):
        mock_uuid.return_value = "test-uuid"
        w = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        )
        expected_path = Path(temp_dir) / "test-uuid"
        assert w._local_path == expected_path
        assert expected_path.exists()
        assert w._uuid == "test-uuid"
        assert w._chunk_size == 128 * MB
        assert w._file_type == BulkFileType.JSON
        assert w._flush_count == 0

    def test_init_with_segment_size(self, simple_schema, temp_dir):
        """Test backward compatibility with segment_size parameter"""
        w = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=64 * MB,
            segment_size=256 * MB,  # Should override chunk_size
            file_type=BulkFileType.PARQUET,
        )
        assert w._chunk_size == 256 * MB

    def test_init_with_describe_collection_struct_array_schema(self, temp_dir):
        raw_schema = {
            "description": "",
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True},
                {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
                {
                    "name": "metadata",
                    "type": DataType.ARRAY,
                    "element_type": DataType.STRUCT,
                    "params": {"max_capacity": 2},
                    "struct_fields": [
                        {"name": "score", "type": DataType.FLOAT},
                        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 2}},
                    ],
                },
            ],
        }
        schema = CollectionSchema.construct_from_dict(raw_schema)
        writer = LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.PARQUET,
        )

        writer.append_row(
            {
                "id": 1,
                "vector": [0.1, 0.2],
                "metadata": [
                    {"score": 1.0, "embedding": [0.3, 0.4]},
                    {"score": 2.0, "embedding": [0.5, 0.6]},
                ],
            }
        )

        assert writer._buffer.row_count == 1

    def test_init_with_milvus_client_schema(self, temp_dir):
        schema = MilvusClient.create_schema(auto_id=False)
        struct_schema = MilvusClient.create_struct_field_schema()
        struct_schema.add_field("score", DataType.FLOAT)
        schema.add_field(
            "chunks",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=2,
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)

        with LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.PARQUET,
        ) as writer:
            writer.append_row(
                {
                    "id": 1,
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "chunks": [{"score": 0.9}],
                }
            )
            writer.commit()

            assert writer.batch_files

    def test_context_manager(self, simple_schema, temp_dir):
        with LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        ) as w:
            assert w is not None
            uuid_dir = w._local_path
            assert uuid_dir.exists()
        assert not uuid_dir.exists()

    def test_del_method(self, simple_schema, temp_dir):
        w = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        )
        uuid_dir = w._local_path
        assert uuid_dir.exists()
        del w
        assert not uuid_dir.exists()

    @patch("pymilvus.bulk_writer.local_bulk_writer.Thread")
    def test_append_row_triggers_flush(self, mock_thread, simple_schema, temp_dir):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        w = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=1,  # Very small to trigger flush
            file_type=BulkFileType.JSON,
        )
        with patch.object(type(w), "buffer_size", new_callable=PropertyMock, return_value=2):
            w.append_row({"id": 1, "vector": [1.0] * 128, "text": "test"})
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()

    @patch("pymilvus.bulk_writer.local_bulk_writer.Thread")
    def test_commit_sync(self, mock_thread, writer):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        writer.append_row({"id": 1, "vector": [1.0] * 128, "text": "test"})
        writer.commit(_async=False)
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()
        mock_thread_instance.join.assert_called()

    @patch("time.sleep")
    @patch("pymilvus.bulk_writer.local_bulk_writer.Thread")
    def test_commit_async(self, mock_thread, mock_sleep, writer):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        writer.append_row({"id": 1, "vector": [1.0] * 128, "text": "test"})
        writer.commit(_async=True)
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()
        mock_thread_instance.join.assert_not_called()

    def test_append_row_fp16_vector_list_uses_fp32_size(self, temp_dir):
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT16_VECTOR, dim=4),
            ]
        )
        writer = LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        )

        writer.append_row({"id": 1, "vector": [1.0, 2.0, 3.0, 4.0]})

        assert writer.buffer_size == 24
        assert writer._buffer._buffer["vector"] == [[1.0, 2.0, 3.0, 4.0]]

    def test_append_row_text_scalar_array_and_struct(self, temp_dir):
        struct_schema = StructFieldSchema()
        struct_schema.add_field("chunk", DataType.TEXT)
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="body", dtype=DataType.TEXT),
                FieldSchema(
                    name="tags",
                    dtype=DataType.ARRAY,
                    element_type=DataType.TEXT,
                    max_capacity=4,
                ),
            ]
        )
        schema.add_field(
            "chunks",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=2,
        )
        writer = LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        )
        long_text = "x" * 70000

        writer.append_row(
            {
                "id": 1,
                "body": long_text,
                "tags": ["short", long_text],
                "chunks": [{"chunk": "a"}, {"chunk": long_text}],
            }
        )

        assert writer.total_row_count == 1
        assert writer._buffer._buffer["body"] == [long_text]
        assert writer._buffer._buffer["tags"] == [["short", long_text]]

    @pytest.mark.parametrize(
        ("vector_type", "dim"),
        [
            pytest.param(DataType.FLOAT_VECTOR, 4, id="float"),
            pytest.param(DataType.FLOAT16_VECTOR, 4, id="float16"),
            pytest.param(DataType.BFLOAT16_VECTOR, 4, id="bfloat16"),
            pytest.param(DataType.INT8_VECTOR, 4, id="int8"),
            pytest.param(DataType.BINARY_VECTOR, 8, id="binary"),
        ],
    )
    @pytest.mark.parametrize(
        "file_type",
        [
            pytest.param(BulkFileType.JSON, id="json"),
            pytest.param(BulkFileType.JSONL, id="jsonl"),
            pytest.param(BulkFileType.CSV, id="csv"),
            pytest.param(BulkFileType.PARQUET, id="parquet"),
        ],
    )
    def test_append_row_struct_vector_subfield_all_file_types(
        self, temp_dir, file_type, vector_type, dim
    ):
        def vector_value():
            if vector_type in {
                DataType.FLOAT_VECTOR,
                DataType.FLOAT16_VECTOR,
                DataType.BFLOAT16_VECTOR,
            }:
                return [1.0, 2.0, 3.0, 4.0]
            if vector_type == DataType.INT8_VECTOR:
                return [-1, 0, 1, 2]
            return [0, 1, 0, 1, 1, 0, 1, 0]

        struct_schema = StructFieldSchema()
        struct_schema.add_field("vector", vector_type, dim=dim)
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            ]
        )
        schema.add_field(
            "chunks",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=2,
        )
        writer = LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=file_type,
        )

        writer.append_row(
            {
                "id": 1,
                "chunks": [
                    {"vector": vector_value()},
                    {"vector": vector_value()},
                ],
            }
        )
        file_prefix = str(Path(temp_dir) / f"{file_type.name.lower()}_{vector_type.name.lower()}")

        files = writer._buffer.persist(file_prefix)

        assert files
        expected_suffix = {
            BulkFileType.JSON: ".json",
            BulkFileType.JSONL: ".jsonl",
            BulkFileType.CSV: ".csv",
            BulkFileType.PARQUET: ".parquet",
        }[file_type]
        for file_path in files:
            path = Path(file_path)
            assert path.exists()
            assert path.suffix == expected_suffix

    @pytest.mark.parametrize(
        "file_type",
        [
            pytest.param(BulkFileType.JSON, id="json"),
            pytest.param(BulkFileType.JSONL, id="jsonl"),
            pytest.param(BulkFileType.CSV, id="csv"),
        ],
    )
    @pytest.mark.parametrize(
        "vector_type",
        [
            pytest.param(DataType.FLOAT16_VECTOR, id="float16"),
            pytest.param(DataType.BFLOAT16_VECTOR, id="bfloat16"),
        ],
    )
    def test_append_row_struct_fp16_bf16_ndarray_subfield_non_parquet_outputs_numeric_lists(
        self, temp_dir, file_type, vector_type
    ):
        struct_schema = StructFieldSchema()
        struct_schema.add_field("vector", vector_type, dim=4)
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            ]
        )
        schema.add_field(
            "chunks",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct_schema,
            max_capacity=2,
        )
        writer = LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=file_type,
        )
        dtype = np.float16 if vector_type == DataType.FLOAT16_VECTOR else ml_dtypes.bfloat16
        expected_vector = [1.0, 2.0, 3.0, 4.0]

        writer.append_row(
            {
                "id": 1,
                "chunks": [
                    {"vector": np.array(expected_vector, dtype=dtype)},
                ],
            }
        )
        file_prefix = str(Path(temp_dir) / f"{file_type.name.lower()}_{vector_type.name.lower()}")

        file_path = Path(writer._buffer.persist(file_prefix)[0])

        if file_type == BulkFileType.JSON:
            rows = json.loads(file_path.read_text())["rows"]
        elif file_type == BulkFileType.JSONL:
            rows = [json.loads(line) for line in file_path.read_text().splitlines()]
        else:
            with file_path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            rows[0]["chunks"] = json.loads(rows[0]["chunks"])

        vector = rows[0]["chunks"][0]["vector"]
        assert vector == expected_vector
        assert all(isinstance(value, float) for value in vector)

    def test_append_row_text_rejects_non_string(self, temp_dir):
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="body", dtype=DataType.TEXT),
            ]
        )
        writer = LocalBulkWriter(
            schema=schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON,
        )

        with pytest.raises(MilvusException, match="Illegal text value"):
            writer.append_row({"id": 1, "body": 1})

    def test_commit_with_callback(self, writer):
        callback_mock = Mock()
        with self._mock_row_count(writer, 1):
            with patch.object(writer._buffer, "persist") as mock_persist:
                test_file = str(writer._local_path / "1" / "test.json")
                mock_persist.return_value = [test_file]
                writer._working_thread[threading.current_thread().name] = threading.current_thread()
                writer._flush(callback_mock)
                callback_mock.assert_called_once_with([test_file])

    def test_flush_with_empty_buffer(self, writer):
        with self._mock_row_count(writer, 0):
            writer._working_thread[threading.current_thread().name] = threading.current_thread()
            writer._flush()
        assert len(writer._local_files) == 0

    def test_flush_exception_handling(self, writer):
        with self._mock_row_count(writer, 1):
            with patch.object(writer._buffer, "persist", side_effect=Exception("Test error")):
                writer._working_thread[threading.current_thread().name] = threading.current_thread()
                with pytest.raises(Exception, match="Test error"):
                    writer._flush()

    def test_properties(self, writer):
        assert writer.uuid == writer._uuid
        assert writer.data_path == writer._local_path
        assert writer.batch_files == []
        writer._local_files = [["file1.json"], ["file2.json"]]
        assert writer.batch_files == [["file1.json"], ["file2.json"]]

    def test_multiple_flushes(self, writer):
        with self._mock_row_count(writer, 1):
            with patch.object(writer._buffer, "persist") as mock_persist:
                mock_persist.side_effect = [
                    [str(writer._local_path / "1" / "1.json")],
                    [str(writer._local_path / "2" / "2.json")],
                    [str(writer._local_path / "3" / "3.json")],
                ]
                current_thread = threading.current_thread()
                for expected_count in range(1, 4):
                    writer._working_thread[current_thread.name] = current_thread
                    writer._flush()
                    assert writer._flush_count == expected_count
                assert len(writer._local_files) == 3

    @patch("time.sleep")
    def test_wait_for_previous_flush(self, mock_sleep, writer):
        mock_thread = MagicMock()
        writer._working_thread = {"thread1": mock_thread}

        def sleep_side_effect(duration):
            if mock_sleep.call_count == 1:
                writer._working_thread.clear()

        mock_sleep.side_effect = sleep_side_effect
        writer.commit(_async=False)
        mock_sleep.assert_called()

    def test_rm_dir(self, writer):
        uuid_dir = writer._local_path
        assert uuid_dir.exists()
        writer._rm_dir()
        assert not uuid_dir.exists()

    def test_rm_dir_not_empty(self, writer):
        uuid_dir = writer._local_path
        test_file = uuid_dir / "test.json"
        test_file.write_text("{}")
        writer._rm_dir()
        # Non-empty directory should NOT be removed
        assert uuid_dir.exists()
        assert test_file.exists()

    def test_exit_waits_for_threads(self, writer):
        mock_thread1 = MagicMock()
        mock_thread2 = MagicMock()
        writer._working_thread = {"thread1": mock_thread1, "thread2": mock_thread2}
        writer._exit()
        mock_thread1.join.assert_called_once()
        mock_thread2.join.assert_called_once()
