import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from pymilvus.bulk_writer.constants import BulkFileType
from pymilvus.bulk_writer.stage_bulk_writer import StageBulkWriter
from pymilvus.bulk_writer.stage_file_manager import StageFileManager
from pymilvus.bulk_writer.stage_manager import StageManager
from pymilvus.bulk_writer.stage_restful import (
    apply_stage,
    create_stage,
    delete_stage,
    list_stages,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema, FieldSchema


class TestStageRestful:
    """Test stage RESTful API functions."""

    @pytest.fixture
    def mock_response(self) -> Mock:
        """Create a mock response object."""
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"code": 0, "message": "success", "data": {}}
        return response

    @pytest.fixture
    def api_params(self) -> Dict[str, str]:
        """Common API parameters."""
        return {
            "url": "https://api.cloud.zilliz.com",
            "api_key": "test_api_key",
        }

    @patch("pymilvus.bulk_writer.stage_restful.requests.get")
    def test_list_stages_success(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful list_stages call."""
        mock_get.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"stages": ["stage1", "stage2"]},
        }

        response = list_stages(
            **api_params,
            project_id="test_project",
            current_page=1,
            page_size=10,
        )

        assert response.status_code == 200
        assert response.json()["data"]["stages"] == ["stage1", "stage2"]
        mock_get.assert_called_once()

    @patch("pymilvus.bulk_writer.stage_restful.requests.get")
    def test_list_stages_failure(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test failed list_stages call."""
        mock_response.json.return_value = {
            "code": 1001,
            "message": "Invalid API key",
            "data": {},
        }
        mock_get.return_value = mock_response

        with pytest.raises(MilvusException, match="Invalid API key"):
            list_stages(**api_params, project_id="test_project")

    @patch("pymilvus.bulk_writer.stage_restful.requests.post")
    def test_create_stage_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful create_stage call."""
        mock_post.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"stageId": "stage123"},
        }

        response = create_stage(
            **api_params,
            project_id="test_project",
            region_id="us-west-2",
            stage_name="test_stage",
        )

        assert response.status_code == 200
        assert response.json()["data"]["stageId"] == "stage123"
        mock_post.assert_called_once()

    @patch("pymilvus.bulk_writer.stage_restful.requests.delete")
    def test_delete_stage_success(
        self, mock_delete: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful delete_stage call."""
        mock_delete.return_value = mock_response

        response = delete_stage(**api_params, stage_name="test_stage")

        assert response.status_code == 200
        mock_delete.assert_called_once()

    @patch("pymilvus.bulk_writer.stage_restful.requests.post")
    def test_apply_stage_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful apply_stage call."""
        mock_post.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {
                "stageName": "test_stage",
                "stagePrefix": "prefix/",
                "endpoint": "s3.amazonaws.com",
                "bucketName": "test-bucket",
                "region": "us-west-2",
                "condition": {"maxContentLength": 1073741824},
                "credentials": {
                    "tmpAK": "test_access_key",
                    "tmpSK": "test_secret_key",
                    "sessionToken": "test_token",
                    "expireTime": "2024-12-31T23:59:59Z",
                },
            },
        }

        response = apply_stage(
            **api_params,
            stage_name="test_stage",
            path="data/",
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["stageName"] == "test_stage"
        assert data["endpoint"] == "s3.amazonaws.com"
        mock_post.assert_called_once()

    @patch("pymilvus.bulk_writer.stage_restful.requests.get")
    def test_http_error_handling(
        self, mock_get: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test HTTP error handling."""
        mock_get.return_value.status_code = 404

        with pytest.raises(MilvusException, match="status code: 404"):
            list_stages(**api_params, project_id="test_project")

    @patch("pymilvus.bulk_writer.stage_restful.requests.get")
    def test_network_error_handling(
        self, mock_get: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test network error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(MilvusException, match="Network error"):
            list_stages(**api_params, project_id="test_project")


class TestStageManager:
    """Test StageManager class."""

    @pytest.fixture
    def stage_manager(self) -> StageManager:
        """Create a StageManager instance."""
        return StageManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
        )

    @patch("pymilvus.bulk_writer.stage_manager.create_stage")
    def test_create_stage(self, mock_create: Mock, stage_manager: StageManager) -> None:
        """Test creating a stage."""
        stage_manager.create_stage(
            project_id="test_project",
            region_id="us-west-2",
            stage_name="test_stage",
        )

        mock_create.assert_called_once_with(
            stage_manager.cloud_endpoint,
            stage_manager.api_key,
            "test_project",
            "us-west-2",
            "test_stage",
        )

    @patch("pymilvus.bulk_writer.stage_manager.delete_stage")
    def test_delete_stage(self, mock_delete: Mock, stage_manager: StageManager) -> None:
        """Test deleting a stage."""
        stage_manager.delete_stage(stage_name="test_stage")

        mock_delete.assert_called_once_with(
            stage_manager.cloud_endpoint,
            stage_manager.api_key,
            "test_stage",
        )

    @patch("pymilvus.bulk_writer.stage_manager.list_stages")
    def test_list_stages(self, mock_list: Mock, stage_manager: StageManager) -> None:
        """Test listing stages."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"stages": ["stage1", "stage2"]}}
        mock_list.return_value = mock_response

        result = stage_manager.list_stages(project_id="test_project", current_page=1, page_size=10)

        assert result.json()["data"]["stages"] == ["stage1", "stage2"]
        mock_list.assert_called_once_with(
            stage_manager.cloud_endpoint,
            stage_manager.api_key,
            "test_project",
            1,
            10,
        )


class TestStageFileManager:
    """Test StageFileManager class."""

    @pytest.fixture
    def stage_file_manager(self) -> StageFileManager:
        """Create a StageFileManager instance."""
        return StageFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            stage_name="test_stage",
        )

    @pytest.fixture
    def mock_stage_info(self) -> Dict[str, Any]:
        """Mock stage information."""
        return {
            "stageName": "test_stage",
            "stagePrefix": "prefix/",
            "endpoint": "s3.amazonaws.com",
            "bucketName": "test-bucket",
            "region": "us-west-2",
            "condition": {"maxContentLength": 1073741824},
            "credentials": {
                "tmpAK": "test_access_key",
                "tmpSK": "test_secret_key",
                "sessionToken": "test_token",
                "expireTime": "2099-12-31T23:59:59Z",
            },
        }

    def test_convert_dir_path(self, stage_file_manager: StageFileManager) -> None:
        """Test directory path conversion."""
        assert stage_file_manager._convert_dir_path("") == ""
        assert stage_file_manager._convert_dir_path("/") == ""
        assert stage_file_manager._convert_dir_path("data") == "data/"
        assert stage_file_manager._convert_dir_path("data/") == "data/"

    @patch("pymilvus.bulk_writer.stage_file_manager.apply_stage")
    @patch("pymilvus.bulk_writer.stage_file_manager.Minio")
    def test_refresh_stage_and_client(
        self,
        mock_minio: Mock,
        mock_apply: Mock,
        stage_file_manager: StageFileManager,
        mock_stage_info: Dict[str, Any],
    ) -> None:
        """Test refreshing stage info and client."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": mock_stage_info}
        mock_apply.return_value = mock_response

        stage_file_manager._refresh_stage_and_client("data/")

        assert stage_file_manager.stage_info == mock_stage_info
        mock_apply.assert_called_once()
        mock_minio.assert_called_once()

    def test_validate_size_success(
        self, stage_file_manager: StageFileManager, mock_stage_info: Dict[str, Any]
    ) -> None:
        """Test successful size validation."""
        stage_file_manager.stage_info = mock_stage_info
        stage_file_manager.total_bytes = 1000000  # 1MB

        # Should not raise any exception
        stage_file_manager._validate_size()

    def test_validate_size_failure(
        self, stage_file_manager: StageFileManager, mock_stage_info: Dict[str, Any]
    ) -> None:
        """Test size validation failure."""
        stage_file_manager.stage_info = mock_stage_info
        stage_file_manager.total_bytes = 2147483648  # 2GB

        with pytest.raises(ValueError, match="exceeds the maximum contentLength limit"):
            stage_file_manager._validate_size()

    @patch("pymilvus.bulk_writer.stage_file_manager.FileUtils.process_local_path")
    @patch.object(StageFileManager, "_refresh_stage_and_client")
    @patch.object(StageFileManager, "_validate_size")
    @patch.object(StageFileManager, "_put_object")
    def test_upload_file_to_stage(
        self,
        mock_put_object: Mock,
        mock_validate: Mock,
        mock_refresh: Mock,
        mock_process: Mock,
        stage_file_manager: StageFileManager,
        mock_stage_info: Dict[str, Any],
    ) -> None:
        """Test uploading file to stage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            mock_process.return_value = ([str(test_file)], 12)
            stage_file_manager.stage_info = mock_stage_info

            result = stage_file_manager.upload_file_to_stage(str(test_file), "data/")

            assert result["stageName"] == "test_stage"
            assert result["path"] == "data/"
            mock_refresh.assert_called_once_with("data/")
            mock_validate.assert_called_once()

    @patch.object(StageFileManager, "_upload_with_retry")
    @patch.object(StageFileManager, "_refresh_stage_and_client")
    def test_put_object_refresh_on_expiry(
        self,
        mock_refresh: Mock,
        mock_upload: Mock,
        stage_file_manager: StageFileManager,
        mock_stage_info: Dict[str, Any],
    ) -> None:
        """Test that credentials are refreshed when expired."""
        # Set expired credentials
        expired_info = mock_stage_info.copy()
        expired_info["credentials"]["expireTime"] = "2020-01-01T00:00:00Z"
        stage_file_manager.stage_info = expired_info

        stage_file_manager._put_object("test.txt", "remote/test.txt", "data/")

        mock_refresh.assert_called_once_with("data/")
        mock_upload.assert_called_once()

    @patch("pymilvus.bulk_writer.stage_file_manager.Minio")
    def test_upload_with_retry_success(
        self, mock_minio: Mock, stage_file_manager: StageFileManager, mock_stage_info: Dict[str, Any]
    ) -> None:
        """Test successful upload with retry."""
        stage_file_manager.stage_info = mock_stage_info
        stage_file_manager._client = mock_minio.return_value

        stage_file_manager._upload_with_retry("test.txt", "remote/test.txt", "data/")

        stage_file_manager._client.fput_object.assert_called_once_with(
            bucket_name="test-bucket",
            object_name="remote/test.txt",
            file_path="test.txt",
        )

    @patch("pymilvus.bulk_writer.stage_file_manager.Minio")
    @patch.object(StageFileManager, "_refresh_stage_and_client")
    def test_upload_with_retry_failure(
        self,
        mock_refresh: Mock,
        mock_minio: Mock,
        stage_file_manager: StageFileManager,
        mock_stage_info: Dict[str, Any],
    ) -> None:
        """Test upload failure after max retries."""
        stage_file_manager.stage_info = mock_stage_info
        mock_client = mock_minio.return_value
        mock_client.fput_object.side_effect = Exception("Upload failed")
        stage_file_manager._client = mock_client

        with pytest.raises(RuntimeError, match="Upload failed after 2 attempts"):
            stage_file_manager._upload_with_retry("test.txt", "remote/test.txt", "data/", max_retries=2)

        assert mock_client.fput_object.call_count == 2
        assert mock_refresh.call_count == 2  # Refreshed on each retry


class TestStageBulkWriter:
    """Test StageBulkWriter class."""

    @pytest.fixture
    def simple_schema(self) -> CollectionSchema:
        """Create a simple collection schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        ]
        return CollectionSchema(fields=fields)

    @pytest.fixture
    def stage_bulk_writer(self, simple_schema: CollectionSchema) -> StageBulkWriter:
        """Create a StageBulkWriter instance."""
        with patch("pymilvus.bulk_writer.stage_bulk_writer.StageFileManager"):
            return StageBulkWriter(
                schema=simple_schema,
                remote_path="test/data",
                cloud_endpoint="https://api.cloud.zilliz.com",
                api_key="test_api_key",
                stage_name="test_stage",
                chunk_size=1024,
                file_type=BulkFileType.PARQUET,
            )

    def test_init(self, stage_bulk_writer: StageBulkWriter) -> None:
        """Test StageBulkWriter initialization."""
        assert stage_bulk_writer._remote_path.endswith("/")
        assert stage_bulk_writer._stage_name == "test_stage"
        assert isinstance(stage_bulk_writer._stage_file_manager, MagicMock)

    def test_append_row(self, stage_bulk_writer: StageBulkWriter) -> None:
        """Test appending a row."""
        row = {
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test text",
        }
        stage_bulk_writer.append_row(row)
        assert stage_bulk_writer.total_row_count == 1

    @patch.object(StageBulkWriter, "_upload")
    def test_commit(self, mock_upload: Mock, stage_bulk_writer: StageBulkWriter) -> None:
        """Test committing data."""
        # Add some data
        for i in range(10):
            stage_bulk_writer.append_row({
                "id": i,
                "vector": [float(i)] * 128,
                "text": f"text_{i}",
            })

        # Mock the upload to return file paths
        mock_upload.return_value = ["file1.parquet", "file2.parquet"]

        # Commit the data
        stage_bulk_writer.commit()

        # Upload should have been called during commit
        assert mock_upload.called

    def test_data_path_property(self, stage_bulk_writer: StageBulkWriter) -> None:
        """Test data_path property."""
        assert isinstance(stage_bulk_writer.data_path, str)
        assert "/" in stage_bulk_writer.data_path

    def test_batch_files_property(self, stage_bulk_writer: StageBulkWriter) -> None:
        """Test batch_files property."""
        assert stage_bulk_writer.batch_files == []
        stage_bulk_writer._remote_files = [["file1.parquet"], ["file2.parquet"]]
        assert stage_bulk_writer.batch_files == [["file1.parquet"], ["file2.parquet"]]

    def test_get_stage_upload_result(self, stage_bulk_writer: StageBulkWriter) -> None:
        """Test getting stage upload result."""
        result = stage_bulk_writer.get_stage_upload_result()
        assert result["stage_name"] == "test_stage"
        assert "path" in result

    @patch("pymilvus.bulk_writer.stage_bulk_writer.Path")
    def test_local_rm(self, mock_path: Mock, stage_bulk_writer: StageBulkWriter) -> None:
        """Test local file removal."""
        # Test successful removal
        mock_file = mock_path.return_value
        mock_file.parent.iterdir.return_value = []

        stage_bulk_writer._local_rm("test_file.parquet")

        mock_file.unlink.assert_called_once()

    @patch.object(StageBulkWriter, "_upload_object")
    @patch.object(StageBulkWriter, "_local_rm")
    @patch("pymilvus.bulk_writer.stage_bulk_writer.Path")
    def test_upload(
        self, mock_path_class: Mock, mock_rm: Mock, mock_upload_object: Mock, stage_bulk_writer: StageBulkWriter
    ) -> None:
        """Test uploading files."""
        # Mock Path behavior
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.relative_to.return_value = Path("test.parquet")

        file_list = ["test_file.parquet"]
        result = stage_bulk_writer._upload(file_list)

        assert len(result) == 1
        mock_upload_object.assert_called_once()
        # The actual call will be with the mock path object
        assert mock_rm.called

    @patch.object(StageFileManager, "upload_file_to_stage")
    def test_upload_object(
        self, mock_upload_to_stage: Mock, stage_bulk_writer: StageBulkWriter
    ) -> None:
        """Test uploading a single object."""
        stage_bulk_writer._upload_object("local_file.parquet", "remote_file.parquet")

        stage_bulk_writer._stage_file_manager.upload_file_to_stage.assert_called_once_with(
            "local_file.parquet", stage_bulk_writer._remote_path
        )

    def test_context_manager(self, simple_schema: CollectionSchema) -> None:
        """Test StageBulkWriter as context manager."""
        with patch("pymilvus.bulk_writer.stage_bulk_writer.StageFileManager"), StageBulkWriter(
                schema=simple_schema,
                remote_path="test/data",
                cloud_endpoint="https://api.cloud.zilliz.com",
                api_key="test_api_key",
                stage_name="test_stage",
            ) as writer:
                assert writer is not None
                writer.append_row({
                    "id": 1,
                    "vector": [1.0] * 128,
                    "text": "test",
                })

    @patch.object(StageBulkWriter, "_upload_object")
    @patch("pymilvus.bulk_writer.stage_bulk_writer.Path")
    def test_upload_error_handling(
        self, mock_path_class: Mock, mock_upload_object: Mock, stage_bulk_writer: StageBulkWriter
    ) -> None:
        """Test error handling during upload."""
        mock_upload_object.side_effect = Exception("Upload error")

        # Mock Path behavior
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.relative_to.return_value = Path("test.parquet")

        with pytest.raises(MilvusException, match="Failed to upload file"):
            stage_bulk_writer._upload(["test_file.parquet"])


class TestIntegration:
    """Integration tests for stage operations."""

    @pytest.fixture
    def mock_server_responses(self) -> Dict[str, Any]:
        """Mock server responses for integration testing."""
        return {
            "apply_stage": {
                "code": 0,
                "message": "success",
                "data": {
                    "stageName": "test_stage",
                    "stagePrefix": "prefix/",
                    "endpoint": "s3.amazonaws.com",
                    "bucketName": "test-bucket",
                    "region": "us-west-2",
                    "condition": {"maxContentLength": 1073741824},
                    "credentials": {
                        "tmpAK": "test_access_key",
                        "tmpSK": "test_secret_key",
                        "sessionToken": "test_token",
                        "expireTime": "2099-12-31T23:59:59Z",
                    },
                },
            },
            "list_stages": {
                "code": 0,
                "message": "success",
                "data": {"stages": ["stage1", "stage2", "test_stage"]},
            },
        }

    @patch("pymilvus.bulk_writer.stage_restful.requests.post")
    @patch("pymilvus.bulk_writer.stage_restful.requests.get")
    def test_full_stage_workflow(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_server_responses: Dict[str, Any],
    ) -> None:
        """Test complete stage workflow from creation to upload."""
        # Setup mock responses
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = mock_server_responses["apply_stage"]
        mock_post.return_value = mock_post_response

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = mock_server_responses["list_stages"]
        mock_get.return_value = mock_get_response

        # Create stage manager
        stage_manager = StageManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
        )

        # List stages
        result = stage_manager.list_stages(project_id="test_project")
        assert "test_stage" in result.json()["data"]["stages"]

        # Create stage file manager
        file_manager = StageFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            stage_name="test_stage",
        )

        # Verify stage info can be refreshed
        file_manager._refresh_stage_and_client("data/")
        assert file_manager.stage_info["stageName"] == "test_stage"
