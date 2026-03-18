import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from pymilvus.bulk_writer.constants import MB, BulkFileType
from pymilvus.bulk_writer.local_bulk_writer import LocalBulkWriter
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema


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
