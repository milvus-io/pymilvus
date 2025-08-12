import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from pymilvus.stage.stage_operation import StageOperation
from pymilvus.stage.file_utils import FileUtils
from minio.error import S3Error


class TestStageOperation:
    @pytest.fixture
    def mock_response(self):
        response = Mock()
        response.json.return_value = {
            "data": {
                "stageName": "test-stage",
                "endpoint": "minio.example.com",
                "uploadPath": "uploads/",
                "bucketName": "test-bucket",
                "region": "us-east-1",
                "credentials": {
                    "tmpAK": "access-key",
                    "tmpSK": "secret-key",
                    "sessionToken": "session-token",
                    "expireTime": "2024-12-31T23:59:59Z"
                },
                "condition": {
                    "maxContentLength": 1073741824  # 1GB
                }
            }
        }
        return response

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_init(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage",
            path="/test/path"
        )
        
        assert stage_op.cloud_endpoint == "https://api.example.com"
        assert stage_op.api_key == "test-api-key"
        assert stage_op.stage_name == "test-stage"
        assert stage_op.path == "/test/path/"
        assert stage_op.stage_info == mock_response.json()["data"]
        
        mock_apply_stage.assert_called_once()
        mock_minio.assert_called_once()

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_convert_dir_path(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        
        # Test path without trailing slash
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage",
            path="/test/path"
        )
        assert stage_op.path == "/test/path/"
        
        # Test path with trailing slash
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage",
            path="/test/path/"
        )
        assert stage_op.path == "/test/path/"
        
        # Test None path
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage",
            path=None
        )
        assert stage_op.path is None
        
        # Test empty path
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage",
            path=""
        )
        assert stage_op.path == ""

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_refresh_stage_and_client(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        # Refresh again
        stage_op._refresh_stage_and_client()
        
        # Should be called twice (once in init, once in refresh)
        assert mock_apply_stage.call_count == 2
        assert mock_minio.call_count == 2

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_validate_size_success(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        stage_op.total_bytes = 500000000  # 500MB
        stage_op._validate_size()  # Should not raise

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_validate_size_exceeds_limit(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        stage_op.total_bytes = 2000000000  # 2GB
        
        with pytest.raises(ValueError, match="exceeds.*maximum contentLength limit"):
            stage_op._validate_size()

    @patch('pymilvus.stage.stage_operation.FileUtils')
    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    @patch('pymilvus.stage.stage_operation.ThreadPoolExecutor')
    def test_upload_file_to_stage(self, mock_executor, mock_minio, mock_apply_stage, mock_file_utils, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        # Mock FileUtils
        mock_file_utils.process_local_path.return_value = (
            ["/tmp/file1.txt", "/tmp/file2.txt"],
            1024000  # 1MB total
        )
        
        # Mock ThreadPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_future = Mock()
        mock_future.result.return_value = None
        mock_executor_instance.submit.return_value = mock_future
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        with patch.object(stage_op, '_put_object'):
            result = stage_op.upload_file_to_stage("/tmp/test", concurrency=5)
        
        assert result["stageName"] == "test-stage"
        assert result["path"] == stage_op.path
        assert mock_executor_instance.submit.call_count == 2

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_put_object_refresh_on_expiry(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        # Set expired time
        stage_op.stage_info["credentials"]["expireTime"] = "2020-01-01T00:00:00Z"
        
        with patch.object(stage_op, '_upload_with_retry') as mock_upload:
            with patch.object(stage_op, '_refresh_stage_and_client') as mock_refresh:
                stage_op._put_object("/tmp/file.txt", "remote/file.txt")
                
                mock_refresh.assert_called_once()
                mock_upload.assert_called_once()

    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_put_object_no_refresh_on_valid_token(self, mock_minio, mock_apply_stage, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        # Set future expiry time
        stage_op.stage_info["credentials"]["expireTime"] = "2099-12-31T23:59:59Z"
        
        with patch.object(stage_op, '_upload_with_retry') as mock_upload:
            with patch.object(stage_op, '_refresh_stage_and_client') as mock_refresh:
                stage_op._put_object("/tmp/file.txt", "remote/file.txt")
                
                mock_refresh.assert_not_called()
                mock_upload.assert_called_once()

    @patch('time.sleep')
    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_upload_with_retry_success(self, mock_minio, mock_apply_stage, mock_sleep, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        stage_op._upload_with_retry("/tmp/file.txt", "remote/file.txt")
        
        mock_minio_instance.fput_object.assert_called_once_with(
            bucket_name="test-bucket",
            object_name="remote/file.txt",
            file_path="/tmp/file.txt"
        )
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_upload_with_retry_fails_then_succeeds(self, mock_minio, mock_apply_stage, mock_sleep, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        # First call fails, second succeeds
        mock_minio_instance.fput_object.side_effect = [
            Exception("Network error"),
            None
        ]
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        with patch.object(stage_op, '_refresh_stage_and_client'):
            stage_op._upload_with_retry("/tmp/file.txt", "remote/file.txt")
        
        assert mock_minio_instance.fput_object.call_count == 2
        mock_sleep.assert_called_once_with(5)

    @patch('time.sleep')
    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_upload_with_retry_max_retries(self, mock_minio, mock_apply_stage, mock_sleep, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        # All calls fail
        mock_minio_instance.fput_object.side_effect = Exception("Network error")
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        with patch.object(stage_op, '_refresh_stage_and_client'):
            with pytest.raises(RuntimeError, match="Upload failed after 3 attempts"):
                stage_op._upload_with_retry("/tmp/file.txt", "remote/file.txt")
        
        assert mock_minio_instance.fput_object.call_count == 3
        assert mock_sleep.call_count == 2

    @patch('pymilvus.stage.stage_operation.Path')
    @patch('pymilvus.stage.stage_operation.FileUtils')
    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_upload_task_file_path(self, mock_minio, mock_apply_stage, mock_file_utils, mock_path, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        # Mock FileUtils
        mock_file_utils.process_local_path.return_value = (
            ["/tmp/dir/file1.txt"],
            1024
        )
        
        # Mock Path operations
        mock_file_path = Mock()
        mock_file_path.resolve.return_value = mock_file_path
        mock_file_path.is_file.return_value = True
        mock_file_path.name = "file1.txt"
        mock_file_path.stat.return_value.st_size = 1024
        mock_path.return_value = mock_file_path
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        with patch.object(stage_op, '_put_object') as mock_put:
            with patch('pymilvus.stage.stage_operation.ThreadPoolExecutor') as mock_executor:
                mock_executor_instance = MagicMock()
                mock_executor.return_value.__enter__.return_value = mock_executor_instance
                mock_future = Mock()
                mock_future.result.return_value = None
                mock_executor_instance.submit.return_value = mock_future
                
                result = stage_op.upload_file_to_stage("/tmp/dir/file1.txt")
        
        assert result["stageName"] == "test-stage"

    @patch('pymilvus.stage.stage_operation.Path')
    @patch('pymilvus.stage.stage_operation.FileUtils')
    @patch('pymilvus.stage.stage_operation.apply_stage')
    @patch('pymilvus.stage.stage_operation.Minio')
    def test_upload_task_directory_path(self, mock_minio, mock_apply_stage, mock_file_utils, mock_path, mock_response):
        mock_apply_stage.return_value = mock_response
        mock_minio_instance = Mock()
        mock_minio.return_value = mock_minio_instance
        
        # Mock FileUtils
        mock_file_utils.process_local_path.return_value = (
            ["/tmp/dir/subdir/file1.txt", "/tmp/dir/file2.txt"],
            2048
        )
        
        # Mock Path operations for root directory
        mock_root_path = Mock()
        mock_root_path.resolve.return_value = mock_root_path
        mock_root_path.is_file.return_value = False
        
        # Mock Path operations for files
        mock_file1_path = Mock()
        mock_file1_path.resolve.return_value = mock_file1_path
        mock_file1_path.relative_to.return_value.as_posix.return_value = "subdir/file1.txt"
        mock_file1_path.stat.return_value.st_size = 1024
        
        mock_file2_path = Mock()
        mock_file2_path.resolve.return_value = mock_file2_path
        mock_file2_path.relative_to.return_value.as_posix.return_value = "file2.txt"
        mock_file2_path.stat.return_value.st_size = 1024
        
        mock_path.side_effect = [mock_root_path, mock_file1_path, mock_root_path, mock_file2_path]
        
        stage_op = StageOperation(
            cloud_endpoint="https://api.example.com",
            api_key="test-api-key",
            stage_name="test-stage"
        )
        
        with patch.object(stage_op, '_put_object') as mock_put:
            with patch('pymilvus.stage.stage_operation.ThreadPoolExecutor') as mock_executor:
                mock_executor_instance = MagicMock()
                mock_executor.return_value.__enter__.return_value = mock_executor_instance
                mock_future = Mock()
                mock_future.result.return_value = None
                mock_executor_instance.submit.return_value = mock_future
                
                result = stage_op.upload_file_to_stage("/tmp/dir")
        
        assert result["stageName"] == "test-stage"
        assert mock_executor_instance.submit.call_count == 2
