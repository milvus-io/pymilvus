"""Tests for GrpcHandler snapshot operations."""

from unittest.mock import MagicMock

from .conftest import make_response, make_status


class TestGrpcHandlerSnapshotOps:
    """Tests for snapshot operations."""

    def test_create_snapshot(self, handler):
        handler._stub.CreateSnapshot.return_value = make_status()
        handler.create_snapshot("snap", "coll", "desc")
        handler._stub.CreateSnapshot.assert_called_once()

    def test_drop_snapshot(self, handler):
        handler._stub.DropSnapshot.return_value = make_status()
        handler.drop_snapshot("snap")
        handler._stub.DropSnapshot.assert_called_once()

    def test_list_snapshots(self, handler):
        mock_snap = MagicMock(name="snap1")
        handler._stub.ListSnapshots.return_value = make_response(snapshots=[mock_snap])
        result = handler.list_snapshots("coll")
        assert len(result) == 1

    def test_describe_snapshot(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.name = "snap"
        mock_resp.collection_name = "coll"
        mock_resp.description = "desc"
        mock_resp.partition_names = ["_default"]
        mock_resp.create_ts = 123
        mock_resp.s3_location = "s3://bucket"
        handler._stub.DescribeSnapshot.return_value = mock_resp
        result = handler.describe_snapshot("snap")
        assert result is not None

    def test_restore_snapshot(self, handler):
        handler._stub.RestoreSnapshot.return_value = make_response(job_id=123)
        result = handler.restore_snapshot("snap", "coll")
        assert result == 123

    def test_get_restore_snapshot_state(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.job_id = 123
        mock_resp.snapshot_name = "snap"
        mock_resp.db_name = "default"
        mock_resp.collection_name = "coll"
        mock_resp.state = 2
        mock_resp.progress = 100
        mock_resp.reason = ""
        mock_resp.start_time = 123
        mock_resp.time_cost = 10
        handler._stub.GetRestoreSnapshotState.return_value = mock_resp
        result = handler.get_restore_snapshot_state(123)
        assert result is not None

    def test_list_restore_snapshot_jobs(self, handler):
        mock_job = MagicMock()
        mock_job.job_id = 123
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.jobs = [mock_job]
        handler._stub.ListRestoreSnapshotJobs.return_value = mock_resp
        result = handler.list_restore_snapshot_jobs("coll")
        assert len(result) == 1

    def test_create_snapshot_with_partitions(self, handler):
        handler._stub.CreateSnapshot.return_value = make_status()
        handler.create_snapshot("snap", "coll", "desc", partition_names=["p1", "p2"])
        handler._stub.CreateSnapshot.assert_called_once()
