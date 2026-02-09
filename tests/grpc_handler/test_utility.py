"""Tests for GrpcHandler utility operations."""

from itertools import count
from unittest.mock import MagicMock, patch

import pytest
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.client.cache import GlobalCache
from pymilvus.exceptions import AmbiguousIndexName, MilvusException

from .conftest import make_response


class TestGrpcHandlerUtilityOps:
    """Tests for utility operations."""

    def test_flush(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_seg = MagicMock(data=[1, 2])
        mock_resp.coll_segIDs = {"coll": mock_seg}
        mock_resp.coll_flush_ts = {"coll": 123}
        mock_resp.flush_coll_segIDs = {}
        mock_resp.coll_seal_times = {}
        mock_resp.flush_ts = 123
        handler._stub.Flush.future.return_value.result.return_value = mock_resp

        mock_flush_state = MagicMock()
        mock_flush_state.status.code = 0
        mock_flush_state.status.error_code = 0
        mock_flush_state.status.reason = ""
        mock_flush_state.flushed = True
        handler._stub.GetFlushState.return_value = mock_flush_state

        handler.flush(["coll"])
        handler._stub.Flush.future.assert_called_once()

    def test_compact(self, handler):
        handler._stub.ManualCompaction.return_value = make_response(compactionID=123)
        assert handler.compact("coll") == 123

    def test_get_compaction_state(self, handler):
        handler._stub.GetCompactionState.return_value = make_response(state=2)
        result = handler.get_compaction_state(123)
        assert result is not None

    def test_get_compaction_plans(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.state = 2
        mock_resp.mergeInfos = []
        handler._stub.GetCompactionStateWithPlans.return_value = mock_resp
        result = handler.get_compaction_plans(123)
        assert result is not None

    def test_get_server_version(self, handler):
        handler._stub.GetVersion.return_value = make_response(version="v2.4.0")
        assert handler.get_server_version() == "v2.4.0"

    def test_get_server_version_with_detail(self, handler):
        """Test get_server_version returns server info dict when detail=True"""
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status

        mock_server_info = MagicMock()
        mock_server_info.build_tags = "2.6.6"
        mock_server_info.build_time = "Fri Jan 23 03:05:45 UTC 2026"
        mock_server_info.git_commit = "cebbe1e4da"
        mock_server_info.go_version = "go version go1.24.11 linux/amd64"
        mock_server_info.deploy_mode = "STANDALONE"
        mock_response.server_info = mock_server_info

        handler._stub.Connect = MagicMock(return_value=mock_response)

        with patch("pymilvus.client.grpc_handler.check_status"):
            result = handler.get_server_version(detail=True)

        expected = {
            "version": "2.6.6",
            "build_time": "Fri Jan 23 03:05:45 UTC 2026",
            "git_commit": "cebbe1e4da",
            "go_version": "go version go1.24.11 linux/amd64",
            "deploy_mode": "STANDALONE",
        }
        assert result == expected
        handler._stub.Connect.assert_called_once()

    def test_get_server_version_with_detail_uses_cache(self, handler):
        """Test get_server_version caches server info and returns cached value"""
        mock_response = MagicMock()
        mock_status = MagicMock()
        mock_status.code = 0
        mock_status.error_code = 0
        mock_status.reason = ""
        mock_response.status = mock_status

        mock_server_info = MagicMock()
        mock_server_info.build_tags = "2.6.6"
        mock_server_info.build_time = "Fri Jan 23 03:05:45 UTC 2026"
        mock_server_info.git_commit = "cebbe1e4da"
        mock_server_info.go_version = "go version go1.24.11 linux/amd64"
        mock_server_info.deploy_mode = "STANDALONE"
        mock_response.server_info = mock_server_info

        handler._stub.Connect = MagicMock(return_value=mock_response)

        with patch("pymilvus.client.grpc_handler.check_status"):
            result1 = handler.get_server_version(detail=True)
            result2 = handler.get_server_version(detail=True)

        assert result1 == result2
        assert handler._stub.Connect.call_count == 1

    def test_get_load_state(self, handler):
        handler._stub.GetLoadState.return_value = make_response(state=3)
        result = handler.get_load_state("coll")
        assert result is not None

    def test_get_loading_progress(self, handler):
        handler._stub.GetLoadingProgress.return_value = make_response(progress=75)
        assert handler.get_loading_progress("coll") == 75

    def test_get_flush_state(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.flushed = True
        handler._stub.GetFlushState.return_value = mock_resp
        assert handler.get_flush_state([1, 2], "coll", 123) is True

    def test_alloc_timestamp(self, handler):
        handler._stub.AllocTimestamp.return_value = make_response(timestamp=123456)
        assert handler.alloc_timestamp() == 123456

    def test_get_replicas(self, handler):
        mock_replica = MagicMock()
        mock_replica.replicaID = 1
        mock_replica.shardReplicas = []
        mock_replica.node_ids = [1]
        mock_replica.resource_group_name = "__default"
        mock_replica.num_outbound_node = {}
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.replicas = [mock_replica]
        handler._stub.GetReplicas.return_value = mock_resp

        with patch.object(handler, "describe_collection", return_value={"collection_id": 1}):
            result = handler.get_replicas("coll")
            assert result is not None


class TestGrpcHandlerSegmentOps:
    """Tests for segment operations."""

    def test_get_query_segment_info(self, handler):
        mock_seg = MagicMock(segmentID=1, collectionID=100)
        handler._stub.GetQuerySegmentInfo.return_value = make_response(infos=[mock_seg])
        handler.get_query_segment_info("coll")
        handler._stub.GetQuerySegmentInfo.assert_called_once()

    def test_get_persistent_segment_infos(self, handler):
        mock_seg = MagicMock(segmentID=1, num_rows=1000)
        handler._stub.GetPersistentSegmentInfo.return_value = make_response(infos=[mock_seg])
        handler.get_persistent_segment_infos("coll")
        handler._stub.GetPersistentSegmentInfo.assert_called_once()


class TestGrpcHandlerImportExport:
    """Tests for bulk insert operations."""

    def test_get_bulk_insert_state(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.state = 2
        mock_resp.row_count = 1000
        mock_resp.id_list = [1, 2, 3]
        mock_resp.infos = []
        mock_resp.create_ts = 789
        handler._stub.GetImportState.return_value = mock_resp
        result = handler.get_bulk_insert_state(123)
        assert result is not None

    def test_list_bulk_insert_tasks(self, handler):
        mock_task = MagicMock()
        mock_task.id = 123
        mock_task.state = 2
        mock_task.row_count = 1000
        mock_task.id_list = []
        mock_task.infos = []
        mock_task.create_ts = 789
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.tasks = [mock_task]
        handler._stub.ListImportTasks.return_value = mock_resp
        result = handler.list_bulk_insert_tasks(10, "coll")
        assert result is not None

    def test_do_bulk_insert(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.tasks = [123]
        handler._stub.Import.return_value = mock_resp

        result = handler.do_bulk_insert("coll", "_default", ["file1.json"])
        assert result == 123


class TestGrpcHandlerFlushAll:
    """Tests for flush all operation."""

    def test_flush_all_sync(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.flush_all_ts = 123
        handler._stub.FlushAll.future.return_value.result.return_value = mock_resp

        with patch.object(handler, "_wait_for_flush_all"):
            handler.flush_all()
            handler._stub.FlushAll.future.assert_called_once()

    def test_flush_all_async(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.flush_all_ts = 123
        mock_future = MagicMock()
        mock_future.result.return_value = mock_resp
        handler._stub.FlushAll.future.return_value = mock_future

        with patch.object(handler, "_wait_for_flush_all"):
            result = handler.flush_all(_async=True)
            assert result is not None


class TestGrpcHandlerFlushAdditional:
    """Additional tests for flush operations."""

    def test_flush_async(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_seg = MagicMock(data=[1, 2])
        mock_resp.coll_segIDs = {"coll": mock_seg}
        mock_resp.coll_flush_ts = {"coll": 123}
        mock_resp.flush_coll_segIDs = {}
        mock_resp.coll_seal_times = {}
        mock_resp.flush_ts = 123
        mock_future = MagicMock()
        mock_future.result.return_value = mock_resp
        handler._stub.Flush.future.return_value = mock_future

        result = handler.flush(["coll"], _async=True)
        assert result is not None

    def test_flush_with_callback(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_seg = MagicMock(data=[1, 2])
        mock_resp.coll_segIDs = {"coll": mock_seg}
        mock_resp.coll_flush_ts = {"coll": 123}
        mock_resp.flush_coll_segIDs = {}
        mock_resp.coll_seal_times = {}
        mock_resp.flush_ts = 123
        mock_future = MagicMock()
        mock_future.result.return_value = mock_resp
        handler._stub.Flush.future.return_value = mock_future

        callback = MagicMock()
        result = handler.flush(["coll"], _async=True, _callback=callback)
        assert result is not None


class TestGrpcHandlerFlushInternal:
    """Tests for internal flush operations."""

    def test_wait_for_flushed(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.flushed = True
        handler._stub.GetFlushState.return_value = mock_resp

        handler._wait_for_flushed([1, 2], "coll", 123)
        handler._stub.GetFlushState.assert_called()


class TestGrpcHandlerAnalyzer:
    """Tests for analyzer operations."""

    def test_run_analyzer_single_text(self, handler):
        mock_result = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = [mock_result]
        handler._stub.RunAnalyzer.return_value = mock_resp

        handler.run_analyzer("hello world")
        handler._stub.RunAnalyzer.assert_called_once()

    def test_run_analyzer_multiple_texts(self, handler):
        mock_result1 = MagicMock()
        mock_result2 = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.results = [mock_result1, mock_result2]
        handler._stub.RunAnalyzer.return_value = mock_resp

        result = handler.run_analyzer(["hello", "world"])
        assert len(result) == 2


class TestGrpcHandlerCompactionAdditional:
    """Additional tests for compaction operations."""

    def test_compact_with_clustering(self, handler):
        handler._stub.ManualCompaction.return_value = make_response(compactionID=456)
        result = handler.compact("coll", is_clustering=True)
        assert result == 456

    def test_compact_with_target_size(self, handler):
        handler._stub.ManualCompaction.return_value = make_response(compactionID=789)
        result = handler.compact("coll", target_size=1024)
        assert result == 789

    def test_compact_is_l0(self, handler):
        handler._stub.ManualCompaction.return_value = make_response(compactionID=111)
        result = handler.compact("coll", is_l0=True)
        assert result == 111


class TestGrpcHandlerMisc:
    """Tests for miscellaneous operations."""

    def test_load_collection_progress(self, handler):
        handler._stub.GetLoadingProgress.return_value = make_response(progress=100)
        result = handler.load_collection_progress("coll")
        assert "loading_progress" in result

    def test_load_partitions_progress(self, handler):
        handler._stub.GetLoadingProgress.return_value = make_response(progress=50)
        result = handler.load_partitions_progress("coll", ["p1"])
        assert "loading_progress" in result

    def test_get_info_with_schema(self, handler):
        schema = {"fields": [{"name": "id"}], "enable_dynamic_field": True}
        fields, dynamic = handler._get_info("coll", schema=schema)
        assert fields == schema["fields"]
        assert dynamic is True

    def test_get_info_without_schema(self, handler):
        with patch.object(handler, "describe_collection") as mock_desc:
            mock_desc.return_value = {"fields": [{"name": "id"}], "enable_dynamic_field": False}
            fields, dynamic = handler._get_info("coll")
            assert fields == [{"name": "id"}]
            assert dynamic is False


class TestGrpcHandlerWaitOps:
    """Tests for wait operations."""

    def test_wait_for_loading_collection(self, handler):
        mock_progress_resp = MagicMock()
        mock_progress_resp.status.code = 0
        mock_progress_resp.status.error_code = 0
        mock_progress_resp.status.reason = ""
        mock_progress_resp.progress = 100
        handler._stub.GetLoadingProgress.return_value = mock_progress_resp

        with patch("time.sleep"):
            handler.wait_for_loading_collection("coll")
            handler._stub.GetLoadingProgress.assert_called()

    def test_wait_for_loading_partitions(self, handler):
        mock_progress_resp = MagicMock()
        mock_progress_resp.status.code = 0
        mock_progress_resp.status.error_code = 0
        mock_progress_resp.status.reason = ""
        mock_progress_resp.progress = 100
        handler._stub.GetLoadingProgress.return_value = mock_progress_resp

        with patch("time.sleep"):
            handler.wait_for_loading_partitions("coll", ["p1"])
            handler._stub.GetLoadingProgress.assert_called()

    def test_wait_for_loading_collection_timeout(self, handler):
        mock_progress_resp = MagicMock()
        mock_progress_resp.status.code = 0
        mock_progress_resp.status.error_code = 0
        mock_progress_resp.status.reason = ""
        mock_progress_resp.progress = 50  # Not 100, never completes
        handler._stub.GetLoadingProgress.return_value = mock_progress_resp

        time_values = [0, 0.5, 1.5]
        with patch("time.sleep"):
            with patch("time.time", side_effect=lambda: time_values.pop(0) if time_values else 2.0):
                with pytest.raises(MilvusException):
                    handler.wait_for_loading_collection("coll", timeout=1)

    def test_wait_for_loading_partitions_timeout(self, handler):
        mock_progress_resp = MagicMock()
        mock_progress_resp.status.code = 0
        mock_progress_resp.status.error_code = 0
        mock_progress_resp.status.reason = ""
        mock_progress_resp.progress = 50  # Not 100, never completes
        handler._stub.GetLoadingProgress.return_value = mock_progress_resp

        time_values = [0, 0.5, 1.5]
        with patch("time.sleep"):
            with patch("time.time", side_effect=lambda: time_values.pop(0) if time_values else 2.0):
                with pytest.raises(MilvusException):
                    handler.wait_for_loading_partitions("coll", ["p1"], timeout=1)

    def test_wait_for_creating_index_timeout(self, handler):
        mock_idx = MagicMock()
        mock_idx.state = 1  # InProgress
        mock_idx.index_state_fail_reason = ""
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx]
        handler._stub.DescribeIndex.return_value = mock_resp
        handler._stub.AllocTimestamp.return_value = make_response(timestamp=123)

        time_counter = count(start=0, step=0.6)  # Increment by 0.6 each call
        with patch("time.sleep"):
            with patch("time.time", side_effect=lambda: next(time_counter)):
                with pytest.raises(MilvusException):
                    handler.wait_for_creating_index("coll", "idx", timeout=1)


class TestGrpcHandlerSchemaCache:
    """Additional tests for schema cache operations."""

    def test_invalidate_db_schemas(self, handler):
        GlobalCache._reset_for_testing()
        GlobalCache.schema.set(handler.server_address, "db1", "coll1", {"fields": []})
        GlobalCache.schema.set(handler.server_address, "db1", "coll2", {"fields": []})

        handler._invalidate_db_schemas("db1")

        assert GlobalCache.schema.get(handler.server_address, "db1", "coll1") is None
        GlobalCache._reset_for_testing()


class TestGrpcHandlerSearchException:
    """Tests for search exception handling."""

    def test_search_exception_async(self, handler):
        handler._stub.Search.future.side_effect = Exception("network error")

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            result = handler.search(
                "coll", "vec", {"metric_type": "L2"}, 10, data=[[0.1, 0.2, 0.3, 0.4]], _async=True
            )
            # Should return a SearchFuture with exception
            assert result is not None

    def test_search_exception_sync_raises(self, handler):
        handler._stub.Search.side_effect = MilvusException(message="network error")

        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            with pytest.raises(MilvusException):
                handler.search(
                    "coll", "vec", {"metric_type": "L2"}, 10, data=[[0.1, 0.2, 0.3, 0.4]]
                )


class TestGrpcHandlerHybridSearchException:
    """Tests for hybrid search exception handling."""

    def test_hybrid_search_exception_async(self, handler):
        handler._stub.HybridSearch.future.side_effect = Exception("network error")

        req = AnnSearchRequest(
            data=[[0.1, 0.2, 0.3, 0.4]], anns_field="vec", param={"metric_type": "L2"}, limit=10
        )
        with patch(
            "pymilvus.client.grpc_handler.ts_utils.construct_guarantee_ts", return_value=True
        ):
            result = handler.hybrid_search("coll", [req], RRFRanker(), 10, _async=True)
            # Should return a SearchFuture with exception
            assert result is not None


class TestGrpcHandlerIndexBuildProgressEdge:
    """Edge cases for get index build progress."""

    def test_get_index_build_progress_ambiguous(self, handler):
        mock_idx1 = MagicMock()
        mock_idx2 = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = [mock_idx1, mock_idx2]  # Multiple indexes
        handler._stub.DescribeIndex.return_value = mock_resp

        with pytest.raises(AmbiguousIndexName):
            handler.get_index_build_progress("coll", "idx")


class TestGrpcHandlerDescribeIndexEmpty:
    """Tests for describe index edge cases."""

    def test_describe_index_not_found(self, handler):
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.index_descriptions = []  # No indexes found
        handler._stub.DescribeIndex.return_value = mock_resp

        # With empty index_descriptions, it raises AmbiguousIndexName
        with pytest.raises(AmbiguousIndexName):
            handler.describe_index("coll", "nonexistent")


class TestGrpcHandlerFileResource:
    """Tests for file resource operations."""

    def test_add_file_resource(self, handler):
        mock_resp = MagicMock()
        mock_resp.code = 0
        mock_resp.error_code = 0
        mock_resp.reason = ""
        handler._stub.AddFileResource.return_value = mock_resp

        handler.add_file_resource("test_resource", "/path/to/file")
        handler._stub.AddFileResource.assert_called_once()

    def test_remove_file_resource(self, handler):
        mock_resp = MagicMock()
        mock_resp.code = 0
        mock_resp.error_code = 0
        mock_resp.reason = ""
        handler._stub.RemoveFileResource.return_value = mock_resp

        handler.remove_file_resource("test_resource")
        handler._stub.RemoveFileResource.assert_called_once()

    def test_list_file_resources(self, handler):
        mock_info = MagicMock()
        mock_info.name = "res1"
        mock_info.path = "/path/to/res1"
        mock_resp = MagicMock()
        mock_resp.status.code = 0
        mock_resp.status.error_code = 0
        mock_resp.status.reason = ""
        mock_resp.resources = [mock_info]
        handler._stub.ListFileResources.return_value = mock_resp

        result = handler.list_file_resources()
        assert len(result) == 1
        handler._stub.ListFileResources.assert_called_once()
