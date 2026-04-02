"""Tests for GrpcHandler external collection operations."""

from unittest.mock import MagicMock

from pymilvus.client.types import RefreshExternalCollectionJobInfo

from .conftest import make_response


def _make_job_info(job_id=1, collection_name="ext_coll", state=2, progress=100, reason=""):
    """Create a mock RefreshExternalCollectionJobInfo protobuf."""
    info = MagicMock()
    info.job_id = job_id
    info.collection_name = collection_name
    info.state = state
    info.progress = progress
    info.reason = reason
    info.external_source = "s3://bucket/path"
    info.start_time = 1000
    info.end_time = 2000
    return info


class TestGrpcHandlerExternalCollectionOps:
    """Tests for external collection refresh operations."""

    def test_refresh_external_collection(self, handler):
        handler._stub.RefreshExternalCollection.return_value = make_response(job_id=42)
        result = handler.refresh_external_collection("ext_coll")
        assert result == 42
        handler._stub.RefreshExternalCollection.assert_called_once()

    def test_refresh_external_collection_with_new_source(self, handler):
        handler._stub.RefreshExternalCollection.return_value = make_response(job_id=43)
        result = handler.refresh_external_collection(
            "ext_coll",
            external_source="s3://new-bucket/path",
            external_spec='{"format": "iceberg"}',
        )
        assert result == 43
        handler._stub.RefreshExternalCollection.assert_called_once()

    def test_get_refresh_progress_completed(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_info = _make_job_info(job_id=42, state=2, progress=100)
        handler._stub.GetRefreshExternalCollectionProgress.return_value = mock_resp

        result = handler.get_refresh_external_collection_progress(42)
        assert isinstance(result, RefreshExternalCollectionJobInfo)
        assert result.job_id == 42
        assert result.state == "RefreshCompleted"
        assert result.progress == 100
        assert result.external_source == "s3://bucket/path"

    def test_get_refresh_progress_in_progress(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_info = _make_job_info(job_id=42, state=1, progress=50)
        handler._stub.GetRefreshExternalCollectionProgress.return_value = mock_resp

        result = handler.get_refresh_external_collection_progress(42)
        assert result.state == "RefreshInProgress"
        assert result.progress == 50

    def test_get_refresh_progress_pending(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_info = _make_job_info(job_id=42, state=0, progress=0)
        handler._stub.GetRefreshExternalCollectionProgress.return_value = mock_resp

        result = handler.get_refresh_external_collection_progress(42)
        assert result.state == "RefreshPending"
        assert result.progress == 0

    def test_get_refresh_progress_failed(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_info = _make_job_info(
            job_id=42, state=3, progress=60, reason="data source unreachable"
        )
        handler._stub.GetRefreshExternalCollectionProgress.return_value = mock_resp

        result = handler.get_refresh_external_collection_progress(42)
        assert result.state == "RefreshFailed"
        assert result.progress == 60
        assert result.reason == "data source unreachable"

    def test_list_refresh_external_collection_jobs(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.jobs = [
            _make_job_info(job_id=1, state=2, progress=100),
            _make_job_info(job_id=2, state=1, progress=30),
        ]
        handler._stub.ListRefreshExternalCollectionJobs.return_value = mock_resp

        result = handler.list_refresh_external_collection_jobs("ext_coll")
        assert len(result) == 2
        assert result[0].job_id == 1
        assert result[0].state == "RefreshCompleted"
        assert result[1].job_id == 2
        assert result[1].state == "RefreshInProgress"

    def test_list_refresh_jobs_empty(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.jobs = []
        handler._stub.ListRefreshExternalCollectionJobs.return_value = mock_resp

        result = handler.list_refresh_external_collection_jobs()
        assert result == []
