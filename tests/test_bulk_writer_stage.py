import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from pymilvus.bulk_writer.constants import BulkFileType, ConnectType
from pymilvus.bulk_writer.volume_bulk_writer import VolumeBulkWriter
from pymilvus.bulk_writer.volume_file_manager import VolumeFileManager
from pymilvus.bulk_writer.volume_manager import VolumeManager
from pymilvus.bulk_writer.volume_restful import (
    apply_volume,
    create_volume,
    delete_volume,
    list_volumes,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema, FieldSchema


class TestVolumeRestful:
    """Test volume RESTful API functions."""

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

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_list_volumes_success(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful list_volumes call."""
        mock_get.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"volumes": ["volume1", "volume2"]},
        }

        response = list_volumes(
            **api_params,
            project_id="test_project",
            current_page=1,
            page_size=10,
        )

        assert response.status_code == 200
        assert response.json()["data"]["volumes"] == ["volume1", "volume2"]
        mock_get.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_list_volumes_failure(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test failed list_volumes call."""
        mock_response.json.return_value = {
            "code": 1001,
            "message": "Invalid API key",
            "data": {},
        }
        mock_get.return_value = mock_response

        with pytest.raises(MilvusException, match="Invalid API key"):
            list_volumes(**api_params, project_id="test_project")

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    def test_create_volume_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful create_volume call."""
        mock_post.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"volumeId": "volume123"},
        }

        response = create_volume(
            **api_params,
            project_id="test_project",
            region_id="us-west-2",
            volume_name="test_volume",
        )

        assert response.status_code == 200
        assert response.json()["data"]["volumeId"] == "volume123"
        mock_post.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.delete")
    def test_delete_volume_success(
        self, mock_delete: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful delete_volume call."""
        mock_delete.return_value = mock_response

        response = delete_volume(**api_params, volume_name="test_volume")

        assert response.status_code == 200
        mock_delete.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    def test_apply_volume_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test successful apply_volume call."""
        mock_post.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {
                "volumeName": "test_volume",
                "volumePrefix": "prefix/",
                "endpoint": "s3.amazonaws.com",
                "bucketName": "test-bucket",
                "region": "us-west-2",
                "cloud": "aws",
                "condition": {"maxContentLength": 1073741824},
                "credentials": {
                    "tmpAK": "test_access_key",
                    "tmpSK": "test_secret_key",
                    "sessionToken": "test_token",
                    "expireTime": "2024-12-31T23:59:59Z",
                },
            },
        }

        response = apply_volume(
            **api_params,
            volume_name="test_volume",
            path="data/",
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["volumeName"] == "test_volume"
        assert data["endpoint"] == "s3.amazonaws.com"
        mock_post.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_http_error_handling(
        self, mock_get: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test HTTP error handling."""
        mock_get.return_value.status_code = 404

        with pytest.raises(MilvusException, match="status code: 404"):
            list_volumes(**api_params, project_id="test_project")

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_network_error_handling(
        self, mock_get: Mock, api_params: Dict[str, str]
    ) -> None:
        """Test network error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(MilvusException, match="Network error"):
            list_volumes(**api_params, project_id="test_project")


class TestVolumeManager:
    """Test VolumeManager class."""

    @pytest.fixture
    def volume_manager(self) -> VolumeManager:
        """Create a VolumeManager instance."""
        return VolumeManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
        )

    @patch("pymilvus.bulk_writer.volume_manager.create_volume")
    def test_create_volume(self, mock_create: Mock, volume_manager: VolumeManager) -> None:
        """Test creating a volume."""
        volume_manager.create_volume(
            project_id="test_project",
            region_id="us-west-2",
            volume_name="test_volume",
        )

        mock_create.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_project",
            "us-west-2",
            "test_volume",
        )

    @patch("pymilvus.bulk_writer.volume_manager.delete_volume")
    def test_delete_volume(self, mock_delete: Mock, volume_manager: VolumeManager) -> None:
        """Test deleting a volume."""
        volume_manager.delete_volume(volume_name="test_volume")

        mock_delete.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_volume",
        )

    @patch("pymilvus.bulk_writer.volume_manager.list_volumes")
    def test_list_volumes(self, mock_list: Mock, volume_manager: VolumeManager) -> None:
        """Test listing volumes."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"volumes": ["volume1", "volume2"]}}
        mock_list.return_value = mock_response

        result = volume_manager.list_volumes(project_id="test_project", current_page=1, page_size=10)

        assert result.json()["data"]["volumes"] == ["volume1", "volume2"]
        mock_list.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_project",
            1,
            10,
        )


class TestVolumeFileManager:
    """Test VolumeFileManager class."""

    @pytest.fixture
    def volume_file_manager(self) -> VolumeFileManager:
        """Create a VolumeFileManager instance."""
        return VolumeFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            volume_name="test_volume",
            connect_type=ConnectType.AUTO,
        )

    @pytest.fixture
    def mock_volume_info(self) -> Dict[str, Any]:
        """Mock volume information."""
        return {
            "volumeName": "test_volume",
            "volumePrefix": "prefix/",
            "endpoint": "s3.amazonaws.com",
            "bucketName": "test-bucket",
            "region": "us-west-2",
            "cloud": "aws",
            "condition": {"maxContentLength": 1073741824},
            "credentials": {
                "tmpAK": "test_access_key",
                "tmpSK": "test_secret_key",
                "sessionToken": "test_token",
                "expireTime": "2099-12-31T23:59:59Z",
            },
        }

    def test_convert_dir_path(self, volume_file_manager: VolumeFileManager) -> None:
        """Test directory path conversion."""
        assert volume_file_manager._convert_dir_path("") == ""
        assert volume_file_manager._convert_dir_path("/") == ""
        assert volume_file_manager._convert_dir_path("data") == "data/"
        assert volume_file_manager._convert_dir_path("data/") == "data/"

    @patch("pymilvus.bulk_writer.volume_file_manager.apply_volume")
    @patch("pymilvus.bulk_writer.volume_file_manager.Minio")
    def test_refresh_volume_and_client(
        self,
        mock_minio: Mock,
        mock_apply: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        """Test refreshing volume info and client."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": mock_volume_info}
        mock_apply.return_value = mock_response

        volume_file_manager._refresh_volume_and_client("data/")

        assert volume_file_manager.volume_info == mock_volume_info
        mock_apply.assert_called_once()
        mock_minio.assert_called_once()

    def test_validate_size_success(
        self, volume_file_manager: VolumeFileManager, mock_volume_info: Dict[str, Any]
    ) -> None:
        """Test successful size validation."""
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager.total_bytes = 1000000  # 1MB

        # Should not raise any exception
        volume_file_manager._validate_size()

    def test_validate_size_failure(
        self, volume_file_manager: VolumeFileManager, mock_volume_info: Dict[str, Any]
    ) -> None:
        """Test size validation failure."""
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager.total_bytes = 2147483648  # 2GB

        with pytest.raises(ValueError, match="exceeds the maximum contentLength limit"):
            volume_file_manager._validate_size()

    @patch("pymilvus.bulk_writer.volume_file_manager.FileUtils.process_local_path")
    @patch.object(VolumeFileManager, "_refresh_volume_and_client")
    @patch.object(VolumeFileManager, "_validate_size")
    @patch.object(VolumeFileManager, "_put_object")
    def test_upload_file_to_volume(
        self,
        mock_put_object: Mock,
        mock_validate: Mock,
        mock_refresh: Mock,
        mock_process: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        """Test uploading file to volume."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            mock_process.return_value = ([str(test_file)], 12)
            volume_file_manager.volume_info = mock_volume_info

            result = volume_file_manager.upload_file_to_volume(str(test_file), "data/")

            assert result["volumeName"] == "test_volume"
            assert result["path"] == "data/"
            mock_refresh.assert_called_once_with("data/")
            mock_validate.assert_called_once()

    @patch.object(VolumeFileManager, "_upload_with_retry")
    @patch.object(VolumeFileManager, "_refresh_volume_and_client")
    def test_put_object_refresh_on_expiry(
        self,
        mock_refresh: Mock,
        mock_upload: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        """Test that credentials are refreshed when expired."""
        # Set expired credentials
        expired_info = mock_volume_info.copy()
        expired_info["credentials"]["expireTime"] = "2020-01-01T00:00:00Z"
        volume_file_manager.volume_info = expired_info

        volume_file_manager._put_object("test.txt", "remote/test.txt", "data/")

        mock_refresh.assert_called_once_with("data/")
        mock_upload.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_file_manager.Minio")
    def test_upload_with_retry_success(
        self, mock_minio: Mock, volume_file_manager: VolumeFileManager, mock_volume_info: Dict[str, Any]
    ) -> None:
        """Test successful upload with retry."""
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager._client = mock_minio.return_value

        volume_file_manager._upload_with_retry("test.txt", "remote/test.txt", "data/")

        volume_file_manager._client.fput_object.assert_called_once_with(
            bucket_name="test-bucket",
            object_name="remote/test.txt",
            file_path="test.txt",
        )

    @patch("pymilvus.bulk_writer.volume_file_manager.Minio")
    @patch.object(VolumeFileManager, "_refresh_volume_and_client")
    def test_upload_with_retry_failure(
        self,
        mock_refresh: Mock,
        mock_minio: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        """Test upload failure after max retries."""
        volume_file_manager.volume_info = mock_volume_info
        mock_client = mock_minio.return_value
        mock_client.fput_object.side_effect = Exception("Upload failed")
        volume_file_manager._client = mock_client

        with pytest.raises(RuntimeError, match="Upload failed after 2 attempts"):
            volume_file_manager._upload_with_retry("test.txt", "remote/test.txt", "data/", max_retries=2)

        assert mock_client.fput_object.call_count == 2
        assert mock_refresh.call_count == 2  # Refreshed on each retry


class TestVolumeBulkWriter:
    """Test VolumeBulkWriter class."""

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
    def volume_bulk_writer(self, simple_schema: CollectionSchema) -> VolumeBulkWriter:
        """Create a VolumeBulkWriter instance."""
        with patch("pymilvus.bulk_writer.volume_bulk_writer.VolumeFileManager"):
            return VolumeBulkWriter(
                schema=simple_schema,
                remote_path="test/data",
                cloud_endpoint="https://api.cloud.zilliz.com",
                api_key="test_api_key",
                volume_name="test_volume",
                chunk_size=1024,
                file_type=BulkFileType.PARQUET,
            )

    def test_init(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test VolumeBulkWriter initialization."""
        assert volume_bulk_writer._remote_path.endswith("/")
        assert volume_bulk_writer._volume_name == "test_volume"
        assert isinstance(volume_bulk_writer._volume_file_manager, MagicMock)

    def test_append_row(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test appending a row."""
        row = {
            "id": 1,
            "vector": [1.0] * 128,
            "text": "test text",
        }
        volume_bulk_writer.append_row(row)
        assert volume_bulk_writer.total_row_count == 1

    @patch.object(VolumeBulkWriter, "_upload")
    def test_commit(self, mock_upload: Mock, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test committing data."""
        # Add some data
        for i in range(10):
            volume_bulk_writer.append_row({
                "id": i,
                "vector": [float(i)] * 128,
                "text": f"text_{i}",
            })

        # Mock the upload to return file paths
        mock_upload.return_value = ["file1.parquet", "file2.parquet"]

        # Commit the data
        volume_bulk_writer.commit()

        # Upload should have been called during commit
        assert mock_upload.called

    def test_data_path_property(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test data_path property."""
        assert isinstance(volume_bulk_writer.data_path, str)
        assert "/" in volume_bulk_writer.data_path

    def test_batch_files_property(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test batch_files property."""
        assert volume_bulk_writer.batch_files == []
        volume_bulk_writer._remote_files = [["file1.parquet"], ["file2.parquet"]]
        assert volume_bulk_writer.batch_files == [["file1.parquet"], ["file2.parquet"]]

    def test_get_volume_upload_result(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test getting volume upload result."""
        result = volume_bulk_writer.get_volume_upload_result()
        assert result["volume_name"] == "test_volume"
        assert "path" in result

    @patch("pymilvus.bulk_writer.volume_bulk_writer.Path")
    def test_local_rm(self, mock_path: Mock, volume_bulk_writer: VolumeBulkWriter) -> None:
        """Test local file removal."""
        # Test successful removal
        mock_file = mock_path.return_value
        mock_file.parent.iterdir.return_value = []

        volume_bulk_writer._local_rm("test_file.parquet")

        mock_file.unlink.assert_called_once()

    @patch.object(VolumeBulkWriter, "_upload_object")
    @patch.object(VolumeBulkWriter, "_local_rm")
    @patch("pymilvus.bulk_writer.volume_bulk_writer.Path")
    def test_upload(
        self, mock_path_class: Mock, mock_rm: Mock, mock_upload_object: Mock, volume_bulk_writer: VolumeBulkWriter
    ) -> None:
        """Test uploading files."""
        # Mock Path behavior
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.relative_to.return_value = Path("test.parquet")

        file_list = ["test_file.parquet"]
        result = volume_bulk_writer._upload(file_list)

        assert len(result) == 1
        mock_upload_object.assert_called_once()
        # The actual call will be with the mock path object
        assert mock_rm.called

    @patch.object(VolumeFileManager, "upload_file_to_volume")
    def test_upload_object(
        self, mock_upload_to_volume: Mock, volume_bulk_writer: VolumeBulkWriter
    ) -> None:
        """Test uploading a single object."""
        volume_bulk_writer._upload_object("local_file.parquet", "remote_file.parquet")

        volume_bulk_writer._volume_file_manager.upload_file_to_volume.assert_called_once_with(
            "local_file.parquet", volume_bulk_writer._remote_path
        )

    def test_context_manager(self, simple_schema: CollectionSchema) -> None:
        """Test VolumeBulkWriter as context manager."""
        with patch("pymilvus.bulk_writer.volume_bulk_writer.VolumeFileManager"), VolumeBulkWriter(
                schema=simple_schema,
                remote_path="test/data",
                cloud_endpoint="https://api.cloud.zilliz.com",
                api_key="test_api_key",
                volume_name="test_volume",
            ) as writer:
                assert writer is not None
                writer.append_row({
                    "id": 1,
                    "vector": [1.0] * 128,
                    "text": "test",
                })

    @patch.object(VolumeBulkWriter, "_upload_object")
    @patch("pymilvus.bulk_writer.volume_bulk_writer.Path")
    def test_upload_error_handling(
        self, mock_path_class: Mock, mock_upload_object: Mock, volume_bulk_writer: VolumeBulkWriter
    ) -> None:
        """Test error handling during upload."""
        mock_upload_object.side_effect = Exception("Upload error")

        # Mock Path behavior
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.relative_to.return_value = Path("test.parquet")

        with pytest.raises(MilvusException, match="Failed to upload file"):
            volume_bulk_writer._upload(["test_file.parquet"])


class TestIntegration:
    """Integration tests for volume operations."""

    @pytest.fixture
    def mock_server_responses(self) -> Dict[str, Any]:
        """Mock server responses for integration testing."""
        return {
            "apply_volume": {
                "code": 0,
                "message": "success",
                "data": {
                    "volumeName": "test_volume",
                    "volumePrefix": "prefix/",
                    "endpoint": "s3.amazonaws.com",
                    "bucketName": "test-bucket",
                    "region": "us-west-2",
                    "cloud": "aws",
                    "condition": {"maxContentLength": 1073741824},
                    "credentials": {
                        "tmpAK": "test_access_key",
                        "tmpSK": "test_secret_key",
                        "sessionToken": "test_token",
                        "expireTime": "2099-12-31T23:59:59Z",
                    },
                },
            },
            "list_volumes": {
                "code": 0,
                "message": "success",
                "data": {"volumes": ["volume1", "volume2", "test_volume"]},
            },
        }

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_full_volume_workflow(
        self,
        mock_get: Mock,
        mock_post: Mock,
        mock_server_responses: Dict[str, Any],
    ) -> None:
        """Test complete volume workflow from creation to upload."""
        # Setup mock responses
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = mock_server_responses["apply_volume"]
        mock_post.return_value = mock_post_response

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = mock_server_responses["list_volumes"]
        mock_get.return_value = mock_get_response

        # Create volume manager
        volume_manager = VolumeManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
        )

        # List volumes
        result = volume_manager.list_volumes(project_id="test_project")
        assert "test_volume" in result.json()["data"]["volumes"]

        # Create volume file manager
        file_manager = VolumeFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            volume_name="test_volume",
            connect_type=ConnectType.AUTO,
        )

        # Verify volume info can be refreshed
        file_manager._refresh_volume_and_client("data/")
        assert file_manager.volume_info["volumeName"] == "test_volume"
