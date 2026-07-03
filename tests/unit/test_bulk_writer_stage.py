import inspect
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import pymilvus.bulk_writer as bw
import pytest
import requests
from pymilvus.bulk_writer.constants import BulkFileType, ConnectType
from pymilvus.bulk_writer.volume_bulk_writer import VolumeBulkWriter
from pymilvus.bulk_writer.volume_file_manager import (
    UploadProgress,
    VolumeFileManager,
    _calculate_upload_part_size,
    _create_minio_http_client,
    _FileUploadProgress,
    _format_bytes,
    _format_duration,
    _format_part_size,
    _UploadProgressTracker,
    _VolumeUploadContext,
)
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
        assert volume_file_manager._convert_dir_path("/data//nested") == "data/nested/"
        with pytest.raises(ValueError, match="escape the volume root"):
            volume_file_manager._convert_dir_path("../data")

    def test_upload_options_are_per_upload_call(self) -> None:
        volume_file_manager = VolumeFileManager(
            cloud_endpoint="https://api.cloud.zilliz.com",
            api_key="test_api_key",
            volume_name="test_volume",
        )
        assert not hasattr(volume_file_manager, "upload_concurrency")
        assert not hasattr(volume_file_manager, "max_retries")
        assert not hasattr(volume_file_manager, "retry_interval")

    def test_upload_file_to_volume_default_upload_concurrency(self) -> None:
        signature = inspect.signature(VolumeFileManager.upload_file_to_volume)
        assert signature.parameters["upload_concurrency"].default == 5

    def test_upload_log_format_helpers(self) -> None:
        assert _format_bytes(12) == "12 B"
        assert _format_bytes(1536) == "1.50 KiB"
        assert _format_duration(-1) == "unknown"
        assert _format_duration(1) == "1s"
        assert _format_duration(61) == "1m 01s"
        assert _format_duration(3661) == "1h 01m 01s"
        assert _format_part_size(0) == "auto"
        assert _format_part_size(5 * 1024 * 1024) == "5242880 bytes (5.00 MiB)"

    def test_upload_progress_tracker_edge_cases(self) -> None:
        tracker = _UploadProgressTracker(total_bytes=0, total_files=1)
        assert tracker.finish_file("empty.txt", 0) == (0, 1, 100.0)
        assert tracker.estimated_remaining_time() == "0s"

        tracker = _UploadProgressTracker(total_bytes=10, total_files=1)
        tracker.update_file("test.txt", 10, 0)
        tracker.update_file("test.txt", 10, 10)
        tracker.update_file("test.txt", 10, 1)
        assert tracker.finish_file("test.txt", 10) == (10, 1, 100.0)

    def test_minio_http_client_has_timeouts(self) -> None:
        http_client = _create_minio_http_client()
        timeout = http_client.connection_pool_kw["timeout"]

        assert http_client.connection_pool_kw["maxsize"] == 100
        assert timeout.connect_timeout == 10.0
        assert timeout.read_timeout == 300.0

    def test_volume_upload_context_set_state(self, mock_volume_info: Dict[str, Any]) -> None:
        context = _VolumeUploadContext(mock_volume_info, "client-a", timedelta(seconds=1))
        updated_info = mock_volume_info.copy()
        updated_info["bucketName"] = "updated-bucket"

        context.set_state(updated_info, "client-b")

        assert context.get_state() == (updated_info, "client-b")

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

    def test_validate_file_number_failure(
        self, volume_file_manager: VolumeFileManager, mock_volume_info: Dict[str, Any]
    ) -> None:
        volume_info = mock_volume_info.copy()
        volume_info["condition"] = mock_volume_info["condition"].copy()
        volume_info["condition"]["maxFileNumber"] = 1

        with pytest.raises(ValueError, match="exceeds the maximum fileNumber limit"):
            volume_file_manager._validate_size(["a", "b"], 2, volume_info)

    @patch("pymilvus.bulk_writer.volume_file_manager.FileUtils.process_local_path")
    @patch.object(VolumeFileManager, "_refresh_volume_and_client")
    def test_upload_file_to_volume_logs_failure(
        self,
        mock_refresh: Mock,
        mock_process: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        volume_info = mock_volume_info.copy()
        volume_info["condition"] = mock_volume_info["condition"].copy()
        volume_info["condition"]["maxContentLength"] = 0
        mock_process.return_value = (["missing.txt"], 1)
        mock_refresh.return_value = _VolumeUploadContext(
            volume_info, Mock(), volume_file_manager.credential_refresh_margin
        )

        with pytest.raises(ValueError, match="maximum contentLength limit"):
            volume_file_manager.upload_file_to_volume("missing.txt", "data/")

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
            mock_refresh.return_value = _VolumeUploadContext(
                mock_volume_info, Mock(), volume_file_manager.credential_refresh_margin
            )
            result = volume_file_manager.upload_file_to_volume(str(test_file), "data/")
            assert result["volumeName"] == "test_volume"
            assert result["volume_name"] == "test_volume"
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
            part_size=5 * 1024 * 1024,
        )

    def test_upload_with_retry_reports_chunk_progress(
        self,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        tracker = _UploadProgressTracker(total_bytes=10, total_files=1)
        progress = _FileUploadProgress(tracker, "test.txt", 10)
        mock_client = Mock()

        def fake_fput_object(**kwargs: Any) -> None:
            kwargs["progress"].set_meta(object_name=kwargs["object_name"], total_length=10)
            kwargs["progress"].update(4)
            kwargs["progress"].update(6)

        mock_client.fput_object.side_effect = fake_fput_object
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager._client = mock_client

        volume_file_manager._upload_with_retry(
            "test.txt", "remote/test.txt", "data/", progress=progress
        )

        mock_client.fput_object.assert_called_once()
        assert mock_client.fput_object.call_args.kwargs["progress"] is progress
        assert tracker.finish_file("test.txt", 10) == (10, 1, 100.0)

    def test_upload_progress_callback_receives_chunk_file_and_final_progress(
        self,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        progress_events: List[UploadProgress] = []
        tracker = _UploadProgressTracker(
            total_bytes=10, total_files=1, progress_callback=progress_events.append
        )
        progress = _FileUploadProgress(tracker, "test.txt", 10)
        mock_client = Mock()

        def fake_fput_object(**kwargs: Any) -> None:
            kwargs["progress"].set_meta(object_name=kwargs["object_name"], total_length=10)
            kwargs["progress"].update(4)
            kwargs["progress"].update(6)

        mock_client.fput_object.side_effect = fake_fput_object
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager._client = mock_client

        volume_file_manager._upload_with_retry(
            "test.txt", "remote/test.txt", "data/", progress=progress, file_size=10
        )
        tracker.finish_file("test.txt", 10)
        tracker.finish_upload()

        assert [event.percent for event in progress_events] == [40.0, 100.0, 100.0, 100.0]
        assert progress_events[0].current_file == "test.txt"
        assert progress_events[0].current_file_uploaded_bytes == 4
        assert progress_events[-1].current_file == ""

    @patch.object(VolumeFileManager, "_refresh_volume_and_client")
    def test_upload_progress_idle_timeout_retries(
        self,
        mock_refresh: Mock,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        tracker = _UploadProgressTracker(total_bytes=10, total_files=1)
        progress = _FileUploadProgress(tracker, "test.txt", 10, idle_timeout_seconds=0.001)
        mock_client = Mock()

        def fake_fput_object(**kwargs: Any) -> None:
            time.sleep(0.01)
            kwargs["progress"].update(1)

        mock_client.fput_object.side_effect = fake_fput_object
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager._client = mock_client

        with pytest.raises(RuntimeError, match="Upload failed after 2 attempts"):
            volume_file_manager._upload_with_retry(
                "test.txt",
                "remote/test.txt",
                "data/",
                max_retries=2,
                retry_interval=0,
                progress=progress,
                file_size=10,
            )

        assert mock_client.fput_object.call_count == 2
        mock_refresh.assert_called_once_with("data/")

    def test_upload_progress_callback_failure_stops_upload_without_retry(
        self,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        tracker = _UploadProgressTracker(
            total_bytes=10,
            total_files=1,
            progress_callback=Mock(side_effect=RuntimeError("stop")),
        )
        progress = _FileUploadProgress(tracker, "test.txt", 10)
        mock_client = Mock()

        def fake_fput_object(**kwargs: Any) -> None:
            kwargs["progress"].set_meta(object_name=kwargs["object_name"], total_length=10)
            kwargs["progress"].update(10)

        mock_client.fput_object.side_effect = fake_fput_object
        volume_file_manager.volume_info = mock_volume_info
        volume_file_manager._client = mock_client

        with pytest.raises(RuntimeError, match="Upload progress callback failed"):
            volume_file_manager._upload_with_retry(
                "test.txt",
                "remote/test.txt",
                "data/",
                max_retries=3,
                progress=progress,
                file_size=10,
            )
        mock_client.fput_object.assert_called_once()

    def test_upload_part_size_auto_and_explicit(self) -> None:
        assert _calculate_upload_part_size(1) == 5 * 1024 * 1024
        assert _calculate_upload_part_size(20 * 1024 * 1024 * 1024) > 5 * 1024 * 1024
        assert _calculate_upload_part_size(1, requested_part_size=7 * 1024 * 1024) == (
            7 * 1024 * 1024
        )

    def test_concurrent_uploads_use_request_local_volume_state(
        self,
        volume_file_manager: VolumeFileManager,
        mock_volume_info: Dict[str, Any],
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_a = Path(temp_dir) / "a.txt"
            file_b = Path(temp_dir) / "b.txt"
            file_a.write_text("a")
            file_b.write_text("b")

            barrier = threading.Barrier(2, timeout=5)
            uploads: List[Tuple[str, str]] = []
            uploads_lock = threading.Lock()

            def make_client() -> Mock:
                client = Mock()

                def fake_fput_object(**kwargs: Any) -> None:
                    barrier.wait()
                    with uploads_lock:
                        uploads.append((kwargs["bucket_name"], kwargs["object_name"]))
                    file_size = Path(kwargs["file_path"]).stat().st_size
                    kwargs["progress"].set_meta(
                        object_name=kwargs["object_name"], total_length=file_size
                    )
                    kwargs["progress"].update(file_size)

                client.fput_object.side_effect = fake_fput_object
                return client

            def create_volume_state(path: str) -> Tuple[Dict[str, Any], Mock]:
                volume_info = mock_volume_info.copy()
                volume_info["condition"] = mock_volume_info["condition"].copy()
                volume_info["credentials"] = mock_volume_info["credentials"].copy()
                volume_info["bucketName"] = f"bucket-{path.rstrip('/')}"
                volume_info["volumePrefix"] = f"prefix-{path.rstrip('/')}/"
                return volume_info, make_client()

            with patch.object(
                volume_file_manager,
                "_create_volume_state",
                side_effect=create_volume_state,
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_a = executor.submit(
                        volume_file_manager.upload_file_to_volume,
                        str(file_a),
                        "a/",
                        1,
                    )
                    future_b = executor.submit(
                        volume_file_manager.upload_file_to_volume,
                        str(file_b),
                        "b/",
                        1,
                    )

                    assert future_a.result()["path"] == "a/"
                    assert future_b.result()["path"] == "b/"

            assert sorted(uploads) == [
                ("bucket-a", "prefix-a/a/a.txt"),
                ("bucket-b", "prefix-b/b/b.txt"),
            ]

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
                "test.txt", "remote/test.txt", "data/", max_retries=2, retry_interval=0
            )
        assert mock_client.fput_object.call_count == 2
        assert mock_refresh.call_count == 1


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
        assert volume_bulk_writer._remote_path.startswith("test/data/")
        assert "\\" not in volume_bulk_writer._remote_path
        assert volume_bulk_writer._volume_name == "test_volume"
        assert isinstance(volume_bulk_writer._volume_file_manager, MagicMock)

    def test_init_normalizes_windows_remote_path(
        self, simple_schema: CollectionSchema
    ) -> None:
        with patch("pymilvus.bulk_writer.volume_bulk_writer.VolumeFileManager"):
            writer = VolumeBulkWriter(
                schema=simple_schema,
                remote_path=r"test\data",
                cloud_endpoint="https://api.cloud.zilliz.com",
                api_key="test_api_key",
                volume_name="test_volume",
                chunk_size=1024,
                file_type=BulkFileType.PARQUET,
            )

        assert writer._remote_path.startswith("test/data/")
        assert writer._remote_path.endswith("/")
        assert "\\" not in writer._remote_path

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
        assert "\\" not in result["path"]

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
        _, kwargs = mock_upload_object.call_args
        assert kwargs["object_name"].endswith("/test.parquet")
        assert "\\" not in kwargs["object_name"]
        assert mock_rm.called

    @patch.object(VolumeFileManager, "upload_file_to_volume")
    def test_upload_object(
        self, mock_upload_to_volume: Mock, volume_bulk_writer: VolumeBulkWriter
    ) -> None:
        object_name = f"{volume_bulk_writer._remote_path}1/local_file.parquet"
        volume_bulk_writer._upload_object("local_file.parquet", object_name)
        volume_bulk_writer._volume_file_manager.upload_file_to_volume.assert_called_once_with(
            "local_file.parquet", f"{volume_bulk_writer._remote_path.rstrip('/')}/1"
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
