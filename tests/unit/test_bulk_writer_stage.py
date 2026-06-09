import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pymilvus.bulk_writer as bw
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
    describe_volume,
    list_volumes,
)
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema, FieldSchema

# ── Module-level shared fixtures ──────────────────────────────────────────────


@pytest.fixture
def simple_schema() -> CollectionSchema:
    """Simple 3-field schema shared across all test classes."""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
    ]
    return CollectionSchema(fields=fields)


@pytest.fixture
def mock_volume_info() -> Dict[str, Any]:
    """Mock volume information shared across test classes."""
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


# ── TestVolumeRestful ─────────────────────────────────────────────────────────


class TestVolumeRestful:
    """Test volume RESTful API functions."""

    @pytest.fixture
    def mock_response(self) -> Mock:
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"code": 0, "message": "success", "data": {}}
        return response

    @pytest.fixture
    def api_params(self) -> Dict[str, str]:
        return {
            "url": "https://api.cloud.zilliz.com",
            "api_key": "test_api_key",
        }

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_list_volumes_success(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        mock_get.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"volumes": ["volume1", "volume2"]},
        }
        response = list_volumes(
            **api_params, project_id="test_project", current_page=1, page_size=10
        )
        assert response.status_code == 200
        assert response.json()["data"]["volumes"] == ["volume1", "volume2"]
        mock_get.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_list_volumes_failure(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        mock_response.json.return_value = {"code": 1001, "message": "Invalid API key", "data": {}}
        mock_get.return_value = mock_response
        with pytest.raises(MilvusException, match="Invalid API key"):
            list_volumes(**api_params, project_id="test_project")

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    def test_create_volume_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
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
        mock_delete.return_value = mock_response
        response = delete_volume(**api_params, volume_name="test_volume")
        assert response.status_code == 200
        mock_delete.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    def test_create_external_volume_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        mock_post.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"volumeName": "ext-volume"},
        }
        response = create_volume(
            **api_params,
            project_id="test_project",
            region_id="aws-us-west-2",
            volume_name="ext-volume",
            volume_type="EXTERNAL",
            storage_integration_id="si-xxxx",
            path="data/",
        )
        assert response.status_code == 200
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["type"] == "EXTERNAL"
        assert body["storageIntegrationId"] == "si-xxxx"
        assert body["path"] == "data/"

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    def test_create_managed_volume_no_extra_params(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        """Verify that creating a MANAGED volume does not send type/storageIntegrationId/path."""
        mock_post.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {"volumeName": "my-volume"},
        }
        response = create_volume(
            **api_params,
            project_id="test_project",
            region_id="aws-us-west-2",
            volume_name="my-volume",
        )
        assert response.status_code == 200
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs["json"]
        assert "type" not in body
        assert "storageIntegrationId" not in body
        assert "path" not in body

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_describe_volume_success(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        mock_get.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {
                "volumeName": "ext-volume",
                "type": "EXTERNAL",
                "regionId": "aws-us-west-2",
                "storageIntegrationId": "si-xxxx",
                "path": "data/",
                "status": "RUNNING",
                "createTime": "2024-04-15T12:00:00Z",
            },
        }
        response = describe_volume(**api_params, volume_name="ext-volume")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["volumeName"] == "ext-volume"
        assert data["type"] == "EXTERNAL"
        assert data["status"] == "RUNNING"
        mock_get.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_describe_volume_failure(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        mock_response.json.return_value = {
            "code": 63242,
            "message": "The volume is not available in this region.",
            "data": {},
        }
        mock_get.return_value = mock_response
        with pytest.raises(MilvusException, match="The volume is not available"):
            describe_volume(**api_params, volume_name="nonexistent")

    @patch("pymilvus.bulk_writer.volume_restful.requests.post")
    def test_apply_volume_success(
        self, mock_post: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
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
        response = apply_volume(**api_params, volume_name="test_volume", path="data/")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["volumeName"] == "test_volume"
        assert data["endpoint"] == "s3.amazonaws.com"
        mock_post.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_list_volumes_with_type_filter(
        self, mock_get: Mock, mock_response: Mock, api_params: Dict[str, str]
    ) -> None:
        mock_get.return_value = mock_response
        mock_response.json.return_value = {
            "code": 0,
            "message": "success",
            "data": {
                "volumes": [{"volumeName": "ext-vol", "type": "EXTERNAL"}],
                "count": 1,
                "currentPage": 1,
                "pageSize": 10,
            },
        }
        response = list_volumes(**api_params, project_id="test_project", volume_type="EXTERNAL")
        assert response.status_code == 200
        # Verify type param was passed in the request
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["type"] == "EXTERNAL"

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_http_error_handling(self, mock_get: Mock, api_params: Dict[str, str]) -> None:
        mock_get.return_value.status_code = 404
        with pytest.raises(MilvusException, match="status code: 404"):
            list_volumes(**api_params, project_id="test_project")

    @patch("pymilvus.bulk_writer.volume_restful.requests.get")
    def test_network_error_handling(self, mock_get: Mock, api_params: Dict[str, str]) -> None:
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        with pytest.raises(MilvusException, match="Network error"):
            list_volumes(**api_params, project_id="test_project")


# ── TestVolumeManager ─────────────────────────────────────────────────────────


class TestVolumeManager:
    """Test VolumeManager class."""

    @pytest.fixture
    def volume_manager(self) -> VolumeManager:
        return VolumeManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
        )

    @patch("pymilvus.bulk_writer.volume_manager.create_volume")
    def test_create_volume(self, mock_create: Mock, volume_manager: VolumeManager) -> None:
        volume_manager.create_volume(
            project_id="test_project", region_id="us-west-2", volume_name="test_volume"
        )
        mock_create.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_project",
            "us-west-2",
            "test_volume",
            None,
            None,
            None,
        )

    @patch("pymilvus.bulk_writer.volume_manager.delete_volume")
    def test_delete_volume(self, mock_delete: Mock, volume_manager: VolumeManager) -> None:
        volume_manager.delete_volume(volume_name="test_volume")
        mock_delete.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_volume",
        )

    @patch("pymilvus.bulk_writer.volume_manager.list_volumes")
    def test_list_volumes(self, mock_list: Mock, volume_manager: VolumeManager) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"volumes": ["volume1", "volume2"]}}
        mock_list.return_value = mock_response
        result = volume_manager.list_volumes(
            project_id="test_project", current_page=1, page_size=10
        )
        assert result.json()["data"]["volumes"] == ["volume1", "volume2"]
        mock_list.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_project",
            1,
            10,
            None,
        )

    @patch("pymilvus.bulk_writer.volume_manager.describe_volume")
    def test_describe_volume(self, mock_describe: Mock, volume_manager: VolumeManager) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "volumeName": "ext-volume",
                "type": "EXTERNAL",
                "regionId": "aws-us-west-2",
                "storageIntegrationId": "si-xxxx",
                "path": "data/",
                "status": "RUNNING",
                "createTime": "2024-04-15T12:00:00Z",
            }
        }
        mock_describe.return_value = mock_response
        result = volume_manager.describe_volume(volume_name="ext-volume")
        assert result.json()["data"]["volumeName"] == "ext-volume"
        assert result.json()["data"]["type"] == "EXTERNAL"
        mock_describe.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "ext-volume",
        )

    @patch("pymilvus.bulk_writer.volume_manager.list_volumes")
    def test_list_volumes_with_type(self, mock_list: Mock, volume_manager: VolumeManager) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {"volumes": [{"volumeName": "ext-vol", "type": "EXTERNAL"}]}
        }
        mock_list.return_value = mock_response
        result = volume_manager.list_volumes(project_id="test_project", volume_type="EXTERNAL")
        assert result.json()["data"]["volumes"][0]["type"] == "EXTERNAL"
        mock_list.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_project",
            1,
            10,
            "EXTERNAL",
        )

    @patch("pymilvus.bulk_writer.volume_manager.create_volume")
    def test_create_external_volume(self, mock_create: Mock, volume_manager: VolumeManager) -> None:
        volume_manager.create_volume(
            project_id="test_project",
            region_id="aws-us-west-2",
            volume_name="ext-volume",
            volume_type="EXTERNAL",
            storage_integration_id="si-xxxx",
            path="data/",
        )
        mock_create.assert_called_once_with(
            volume_manager.cloud_endpoint,
            volume_manager.api_key,
            "test_project",
            "aws-us-west-2",
            "ext-volume",
            "EXTERNAL",
            "si-xxxx",
            "data/",
        )


# ── TestVolumeFileManager ─────────────────────────────────────────────────────


class TestVolumeFileManager:
    """Test VolumeFileManager class."""

    @pytest.fixture
    def volume_file_manager(self) -> VolumeFileManager:
        return VolumeFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            volume_name="test_volume",
            connect_type=ConnectType.AUTO,
        )

    def test_convert_dir_path(self, volume_file_manager: VolumeFileManager) -> None:
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
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager.total_bytes = 1000000  # 1MB
        volume_file_manager._validate_size()  # Should not raise

    def test_validate_size_failure(
        self, volume_file_manager: VolumeFileManager, mock_volume_info: Dict[str, Any]
    ) -> None:
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
        with tempfile.TemporaryDirectory() as temp_dir:
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
        expired_info = mock_volume_info.copy()
        expired_info["credentials"]["expireTime"] = "2020-01-01T00:00:00Z"
        volume_file_manager.volume_info = expired_info
        volume_file_manager._put_object("test.txt", "remote/test.txt", "data/")
        mock_refresh.assert_called_once_with("data/")
        mock_upload.assert_called_once()

    @patch("pymilvus.bulk_writer.volume_file_manager.Minio")
    def test_upload_with_retry_success(
        self,
        mock_minio: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
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
        volume_file_manager.volume_info = mock_volume_info
        mock_client = mock_minio.return_value
        mock_client.fput_object.side_effect = Exception("Upload failed")
        volume_file_manager._client = mock_client
        with pytest.raises(RuntimeError, match="Upload failed after 2 attempts"):
            volume_file_manager._upload_with_retry(
                "test.txt", "remote/test.txt", "data/", max_retries=2
            )
        assert mock_client.fput_object.call_count == 2
        assert mock_refresh.call_count == 2


# ── TestVolumeBulkWriter ──────────────────────────────────────────────────────


class TestVolumeBulkWriter:
    """Test VolumeBulkWriter class."""

    @pytest.fixture
    def volume_bulk_writer(self, simple_schema: CollectionSchema) -> VolumeBulkWriter:
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
        assert volume_bulk_writer._remote_path.endswith("/")
        assert volume_bulk_writer._volume_name == "test_volume"
        assert isinstance(volume_bulk_writer._volume_file_manager, MagicMock)

    def test_append_row(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        volume_bulk_writer.append_row({"id": 1, "vector": [1.0] * 128, "text": "test text"})
        assert volume_bulk_writer.total_row_count == 1

    @patch.object(VolumeBulkWriter, "_upload")
    def test_commit(self, mock_upload: Mock, volume_bulk_writer: VolumeBulkWriter) -> None:
        for i in range(10):
            volume_bulk_writer.append_row(
                {"id": i, "vector": [float(i)] * 128, "text": f"text_{i}"}
            )
        mock_upload.return_value = ["file1.parquet", "file2.parquet"]
        volume_bulk_writer.commit()
        assert mock_upload.called

    def test_data_path_property(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        assert isinstance(volume_bulk_writer.data_path, str)
        assert "/" in volume_bulk_writer.data_path

    def test_batch_files_property(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        assert volume_bulk_writer.batch_files == []
        volume_bulk_writer._remote_files = [["file1.parquet"], ["file2.parquet"]]
        assert volume_bulk_writer.batch_files == [["file1.parquet"], ["file2.parquet"]]

    def test_get_volume_upload_result(self, volume_bulk_writer: VolumeBulkWriter) -> None:
        result = volume_bulk_writer.get_volume_upload_result()
        assert result["volume_name"] == "test_volume"
        assert "path" in result

    @patch("pymilvus.bulk_writer.volume_bulk_writer.Path")
    def test_local_rm(self, mock_path: Mock, volume_bulk_writer: VolumeBulkWriter) -> None:
        mock_file = mock_path.return_value
        mock_file.parent.iterdir.return_value = []
        volume_bulk_writer._local_rm("test_file.parquet")
        mock_file.unlink.assert_called_once()

    @patch.object(VolumeBulkWriter, "_upload_object")
    @patch.object(VolumeBulkWriter, "_local_rm")
    @patch("pymilvus.bulk_writer.volume_bulk_writer.Path")
    def test_upload(
        self,
        mock_path_class: Mock,
        mock_rm: Mock,
        mock_upload_object: Mock,
        volume_bulk_writer: VolumeBulkWriter,
    ) -> None:
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.relative_to.return_value = Path("test.parquet")
        result = volume_bulk_writer._upload(["test_file.parquet"])
        assert len(result) == 1
        mock_upload_object.assert_called_once()
        assert mock_rm.called

    @patch.object(VolumeFileManager, "upload_file_to_volume")
    def test_upload_object(
        self, mock_upload_to_volume: Mock, volume_bulk_writer: VolumeBulkWriter
    ) -> None:
        volume_bulk_writer._upload_object("local_file.parquet", "remote_file.parquet")
        volume_bulk_writer._volume_file_manager.upload_file_to_volume.assert_called_once_with(
            "local_file.parquet", volume_bulk_writer._remote_path
        )

    def test_context_manager(self, simple_schema: CollectionSchema) -> None:
        with patch("pymilvus.bulk_writer.volume_bulk_writer.VolumeFileManager"), VolumeBulkWriter(
            schema=simple_schema,
            remote_path="test/data",
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            volume_name="test_volume",
        ) as writer:
            assert writer is not None
            writer.append_row({"id": 1, "vector": [1.0] * 128, "text": "test"})

    @patch.object(VolumeBulkWriter, "_upload_object")
    @patch("pymilvus.bulk_writer.volume_bulk_writer.Path")
    def test_upload_error_handling(
        self,
        mock_path_class: Mock,
        mock_upload_object: Mock,
        volume_bulk_writer: VolumeBulkWriter,
    ) -> None:
        mock_upload_object.side_effect = Exception("Upload error")
        mock_path = Mock()
        mock_path_class.return_value = mock_path
        mock_path.relative_to.return_value = Path("test.parquet")
        with pytest.raises(MilvusException, match="Failed to upload file"):
            volume_bulk_writer._upload(["test_file.parquet"])


# ── TestIntegration ───────────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests for volume operations."""

    @pytest.fixture
    def mock_server_responses(self) -> Dict[str, Any]:
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
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = mock_server_responses["apply_volume"]
        mock_post.return_value = mock_post_response

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = mock_server_responses["list_volumes"]
        mock_get.return_value = mock_get_response

        volume_manager = VolumeManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
        )
        result = volume_manager.list_volumes(project_id="test_project")
        assert "test_volume" in result.json()["data"]["volumes"]

        file_manager = VolumeFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            volume_name="test_volume",
            connect_type=ConnectType.AUTO,
        )
        file_manager._refresh_volume_and_client("data/")
        assert file_manager.volume_info["volumeName"] == "test_volume"


# ── TestVolumeExports ────────────────────────────────────────────────────────


class TestVolumeExports:
    """Test that volume classes are properly exported from bulk_writer package."""

    def test_volume_manager_exportable(self) -> None:
        assert hasattr(bw, "VolumeManager")
        assert bw.VolumeManager is VolumeManager

    def test_volume_bulk_writer_exportable(self) -> None:
        assert hasattr(bw, "VolumeBulkWriter")
        assert bw.VolumeBulkWriter is VolumeBulkWriter

    def test_volume_file_manager_exportable(self) -> None:
        assert hasattr(bw, "VolumeFileManager")
        assert bw.VolumeFileManager is VolumeFileManager
