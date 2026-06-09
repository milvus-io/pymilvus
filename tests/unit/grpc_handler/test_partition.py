"""Tests for GrpcHandler partition operations."""

from unittest.mock import MagicMock, patch

import pytest

from .conftest import PARTITION_VALIDATION_CASES, make_response, make_status


class TestGrpcHandlerPartitionOps:
    """Tests for partition operations."""

    @pytest.mark.parametrize("coll,part,error", PARTITION_VALIDATION_CASES)
    def test_create_partition_validation(self, handler, coll, part, error):
        if error:
            with pytest.raises(error):
                handler.create_partition(coll, part)
        else:
            handler._stub.CreatePartition.return_value = make_status()
            handler.create_partition(coll, part)
            handler._stub.CreatePartition.assert_called_once()

    @pytest.mark.parametrize("coll,part,error", PARTITION_VALIDATION_CASES)
    def test_drop_partition_validation(self, handler, coll, part, error):
        if error:
            with pytest.raises(error):
                handler.drop_partition(coll, part)
        else:
            handler._stub.DropPartition.return_value = make_status()
            handler.drop_partition(coll, part)
            handler._stub.DropPartition.assert_called_once()

    def test_has_partition(self, handler):
        handler._stub.HasPartition.return_value = make_response(value=True)
        assert handler.has_partition("coll", "part") is True

    def test_list_partitions(self, handler):
        handler._stub.ShowPartitions.return_value = make_response(
            partition_names=["_default", "p1"]
        )
        assert handler.list_partitions("coll") == ["_default", "p1"]

    def test_get_partition_stats(self, handler):
        handler._stub.GetPartitionStatistics.return_value = make_response(
            stats=[MagicMock(key="row_count", value="100")]
        )
        result = handler.get_partition_stats("coll", "part")
        assert len(result) == 1

    def test_load_partitions(self, handler):
        handler._stub.LoadPartitions.return_value = make_status()
        with patch.object(handler, "wait_for_loading_partitions"):
            handler.load_partitions("coll", ["p1", "p2"])
            handler._stub.LoadPartitions.assert_called_once()

    def test_release_partitions(self, handler):
        handler._stub.ReleasePartitions.return_value = make_status()
        handler.release_partitions("coll", ["p1"])
        handler._stub.ReleasePartitions.assert_called_once()
