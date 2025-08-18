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
        # Cleanup after test
        if Path(temp_path).exists():
            shutil.rmtree(temp_path)

    @patch('uuid.uuid4')
    def test_init(self, mock_uuid, simple_schema, temp_dir):
        mock_uuid.return_value = "test-uuid"

        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        # Check the UUID directory was created
        expected_path = Path(temp_dir) / "test-uuid"
        assert writer._local_path == expected_path
        assert expected_path.exists()
        assert writer._uuid == "test-uuid"
        assert writer._chunk_size == 128 * MB
        assert writer._file_type == BulkFileType.JSON
        assert writer._flush_count == 0

    def test_init_with_segment_size(self, simple_schema, temp_dir):
        """Test backward compatibility with segment_size parameter"""
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=64 * MB,
            segment_size=256 * MB,  # Should override chunk_size
            file_type=BulkFileType.PARQUET
        )

        assert writer._chunk_size == 256 * MB

    def test_context_manager(self, simple_schema, temp_dir):
        uuid_dir = None
        with LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        ) as writer:
            assert writer is not None
            uuid_dir = writer._local_path
            assert uuid_dir.exists()

        # After context exit, empty UUID directory should be removed
        assert not uuid_dir.exists()

    def test_del_method(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )
        uuid_dir = writer._local_path
        assert uuid_dir.exists()

        del writer
        # Empty UUID directory should be removed through __del__
        assert not uuid_dir.exists()

    @patch('pymilvus.bulk_writer.local_bulk_writer.Thread')
    def test_append_row_triggers_flush(self, mock_thread, simple_schema, temp_dir):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=1,  # Very small size to trigger flush
            file_type=BulkFileType.JSON
        )

        # Use PropertyMock to mock the buffer_size property
        with patch.object(type(writer), 'buffer_size', new_callable=PropertyMock) as mock_buffer_size:
            mock_buffer_size.return_value = 2  # Larger than chunk_size
            writer.append_row({
                "id": 1,
                "vector": [1.0] * 128,
                "text": "test"
            })

        # Verify flush thread was started
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()

    @patch('pymilvus.bulk_writer.local_bulk_writer.Thread')
    def test_commit_sync(self, mock_thread, simple_schema, temp_dir):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        writer.append_row({
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test"
        })

        # Commit synchronously
        writer.commit(_async=False)

        # Thread should be started and joined
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()
        mock_thread_instance.join.assert_called()

    @patch('time.sleep')
    @patch('pymilvus.bulk_writer.local_bulk_writer.Thread')
    def test_commit_async(self, mock_thread, mock_sleep, simple_schema, temp_dir):
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        writer.append_row({
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test"
        })

        # Commit asynchronously
        writer.commit(_async=True)

        # Thread should be started but not joined
        mock_thread.assert_called()
        mock_thread_instance.start.assert_called()
        mock_thread_instance.join.assert_not_called()

    def test_commit_with_callback(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        callback_mock = Mock()

        # Mock the buffer to have data using PropertyMock
        with patch.object(type(writer._buffer), 'row_count', new_callable=PropertyMock) as mock_row_count:
            mock_row_count.return_value = 1
            with patch.object(writer._buffer, 'persist') as mock_persist:
                test_file = str(writer._local_path / "1" / "test.json")
                mock_persist.return_value = [test_file]
                # Add current thread to _working_thread to avoid KeyError
                writer._working_thread[threading.current_thread().name] = threading.current_thread()
                writer._flush(callback_mock)

                # Callback should be called with file list
                callback_mock.assert_called_once_with([test_file])

    def test_flush_with_empty_buffer(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        # Mock empty buffer using PropertyMock
        with patch.object(type(writer._buffer), 'row_count', new_callable=PropertyMock) as mock_row_count:
            mock_row_count.return_value = 0
            # Add current thread to _working_thread to avoid KeyError
            writer._working_thread[threading.current_thread().name] = threading.current_thread()
            writer._flush()

        # No files should be added
        assert len(writer._local_files) == 0

    def test_flush_exception_handling(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        # Mock buffer to raise exception using PropertyMock
        with patch.object(type(writer._buffer), 'row_count', new_callable=PropertyMock) as mock_row_count:
            mock_row_count.return_value = 1
            with patch.object(writer._buffer, 'persist', side_effect=Exception("Test error")):
                # Add current thread to _working_thread to avoid KeyError
                writer._working_thread[threading.current_thread().name] = threading.current_thread()
                with pytest.raises(Exception, match="Test error"):
                    writer._flush()

    def test_properties(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        assert writer.uuid == writer._uuid
        assert writer.data_path == writer._local_path
        assert writer.batch_files == []

        # Add some files
        writer._local_files = [["file1.json"], ["file2.json"]]
        assert writer.batch_files == [["file1.json"], ["file2.json"]]

    def test_multiple_flushes(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        # Simulate multiple flushes using PropertyMock
        with patch.object(type(writer._buffer), 'row_count', new_callable=PropertyMock) as mock_row_count:
            mock_row_count.return_value = 1
            with patch.object(writer._buffer, 'persist') as mock_persist:
                mock_persist.side_effect = [
                    [str(writer._local_path / "1" / "1.json")],
                    [str(writer._local_path / "2" / "2.json")],
                    [str(writer._local_path / "3" / "3.json")]
                ]

                # Add current thread to _working_thread before each flush
                current_thread = threading.current_thread()

                writer._working_thread[current_thread.name] = current_thread
                writer._flush()
                assert writer._flush_count == 1

                writer._working_thread[current_thread.name] = current_thread
                writer._flush()
                assert writer._flush_count == 2

                writer._working_thread[current_thread.name] = current_thread
                writer._flush()
                assert writer._flush_count == 3

                assert len(writer._local_files) == 3

    @patch('time.sleep')
    def test_wait_for_previous_flush(self, mock_sleep, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        # Simulate a working thread
        mock_thread = MagicMock()
        writer._working_thread = {"thread1": mock_thread}

        # Create a side effect that clears the working thread after first call
        def sleep_side_effect(duration):
            if mock_sleep.call_count == 1:
                writer._working_thread.clear()

        mock_sleep.side_effect = sleep_side_effect

        writer.commit(_async=False)

        # Should have waited for previous thread
        mock_sleep.assert_called()

    def test_rm_dir(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        uuid_dir = writer._local_path
        assert uuid_dir.exists()

        # Test removing empty directory
        writer._rm_dir()

        # Directory should be removed
        assert not uuid_dir.exists()

    def test_rm_dir_not_empty(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        uuid_dir = writer._local_path

        # Create a file in the directory
        test_file = uuid_dir / "test.json"
        test_file.write_text("{}")

        assert uuid_dir.exists()
        assert test_file.exists()

        # Test removing non-empty directory
        writer._rm_dir()

        # Directory should NOT be removed (still has files)
        assert uuid_dir.exists()
        assert test_file.exists()

    def test_exit_waits_for_threads(self, simple_schema, temp_dir):
        writer = LocalBulkWriter(
            schema=simple_schema,
            local_path=temp_dir,
            chunk_size=128 * MB,
            file_type=BulkFileType.JSON
        )

        # Add mock working threads
        mock_thread1 = MagicMock()
        mock_thread2 = MagicMock()
        writer._working_thread = {
            "thread1": mock_thread1,
            "thread2": mock_thread2
        }

        writer._exit()

        # Should wait for all threads
        mock_thread1.join.assert_called_once()
        mock_thread2.join.assert_called_once()
