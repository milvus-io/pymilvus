"""Tests for GrpcHandler snapshot operations."""

from unittest.mock import MagicMock

from .conftest import make_response, make_status


class TestGrpcHandlerSnapshotOps:
    """Tests for snapshot operations."""

    def test_create_snapshot(self, handler):
        handler._stub.CreateSnapshot.return_value = make_status()
        handler.create_snapshot("snap", "coll", description="desc")
        handler._stub.CreateSnapshot.assert_called_once()
        req = handler._stub.CreateSnapshot.call_args[0][0]
        assert req.collection_name == "coll"
        assert req.name == "snap"
        assert req.compaction_protection_seconds == 0

    def test_create_snapshot_with_compaction_protection(self, handler):
        handler._stub.CreateSnapshot.return_value = make_status()
        handler.create_snapshot("snap", "coll", description="desc", compaction_protection_seconds=3600)
        req = handler._stub.CreateSnapshot.call_args[0][0]
        assert req.compaction_protection_seconds == 3600

    def test_drop_snapshot(self, handler):
        handler._stub.DropSnapshot.return_value = make_status()
        handler.drop_snapshot("snap", collection_name="coll")
        handler._stub.DropSnapshot.assert_called_once()
        req = handler._stub.DropSnapshot.call_args[0][0]
        assert req.name == "snap"
        assert req.collection_name == "coll"

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
        result = handler.describe_snapshot("snap", collection_name="coll")
        assert result is not None
        req = handler._stub.DescribeSnapshot.call_args[0][0]
        assert req.name == "snap"
        assert req.collection_name == "coll"

    def test_restore_snapshot(self, handler):
        handler._stub.RestoreSnapshot.return_value = make_response(job_id=123)
        result = handler.restore_snapshot(
            snapshot_name="snap",
            target_collection_name="new_coll",
            source_collection_name="src_coll",
        )
        assert result == 123
        req = handler._stub.RestoreSnapshot.call_args[0][0]
        assert req.name == "snap"
        assert req.target_collection_name == "new_coll"
        assert req.collection_name == "src_coll"

    def test_pin_snapshot_data(self, handler):
        handler._stub.PinSnapshotData.return_value = make_response(pin_id=42)
        pin_id = handler.pin_snapshot_data("snap", collection_name="coll", ttl_seconds=60)
        assert pin_id == 42
        req = handler._stub.PinSnapshotData.call_args[0][0]
        assert req.name == "snap"
        assert req.ttl_seconds == 60

    def test_unpin_snapshot_data(self, handler):
        handler._stub.UnpinSnapshotData.return_value = make_status()
        handler.unpin_snapshot_data(pin_id=42)
        req = handler._stub.UnpinSnapshotData.call_args[0][0]
        assert req.pin_id == 42

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
        handler.create_snapshot("snap", "coll", description="desc", partition_names=["p1", "p2"])
        handler._stub.CreateSnapshot.assert_called_once()
