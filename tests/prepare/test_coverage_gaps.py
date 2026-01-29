"""Additional tests to increase coverage for Prepare methods."""

import numpy as np
import pytest
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError


class TestCreateCollectionNumPartitions:
    """Tests for num_partitions validation in create_collection_request."""

    def test_num_partitions_none(self, basic_schema):
        """Test create_collection_request with None num_partitions (default)."""
        req = Prepare.create_collection_request("test_coll", basic_schema)
        assert req is not None

    def test_num_partitions_valid_int(self, basic_schema):
        """Test create_collection_request with valid int num_partitions."""
        req = Prepare.create_collection_request("test_coll", basic_schema, num_partitions=8)
        assert req.num_partitions == 8


class TestShowCollectionsRequest:
    """Tests for show_collections_request."""

    def test_show_collections_default(self):
        """Test show_collections_request with no params."""
        req = Prepare.show_collections_request()
        assert req is not None

    def test_show_collections_with_names(self):
        """Test show_collections_request with collection names."""
        req = Prepare.show_collections_request(collection_names=["coll1", "coll2"])
        assert list(req.collection_names) == ["coll1", "coll2"]

    def test_show_collections_invalid_names_type(self):
        """Test show_collections_request with invalid collection names type."""
        with pytest.raises(ParamError):
            Prepare.show_collections_request(collection_names="not_a_list")


class TestShowPartitionsRequest:
    """Tests for show_partitions_request."""

    def test_show_partitions_basic(self):
        """Test basic show_partitions_request."""
        req = Prepare.show_partitions_request("test_coll")
        assert req.collection_name == "test_coll"

    def test_show_partitions_with_names(self):
        """Test show_partitions_request with partition names."""
        req = Prepare.show_partitions_request("test_coll", partition_names=["part1", "part2"])
        assert list(req.partition_names) == ["part1", "part2"]

    def test_show_partitions_invalid_names_type(self):
        """Test show_partitions_request with invalid partition names type."""
        with pytest.raises(ParamError):
            Prepare.show_partitions_request("test_coll", partition_names="not_a_list")

    def test_show_partitions_type_in_memory_false(self):
        """Test show_partitions_request with type_in_memory=False."""
        req = Prepare.show_partitions_request("test_coll", type_in_memory=False)
        assert req is not None


class TestGetLoadingProgress:
    """Tests for get_loading_progress."""

    def test_loading_progress_basic(self):
        """Test basic loading progress request."""
        req = Prepare.get_loading_progress("test_coll")
        assert req.collection_name == "test_coll"

    def test_loading_progress_with_partitions(self):
        """Test loading progress with partition names."""
        req = Prepare.get_loading_progress("test_coll", partition_names=["part1"])
        assert list(req.partition_names) == ["part1"]


class TestGetLoadState:
    """Tests for get_load_state."""

    def test_load_state_basic(self):
        """Test basic load state request."""
        req = Prepare.get_load_state("test_coll")
        assert req.collection_name == "test_coll"

    def test_load_state_with_partitions(self):
        """Test load state with partition names."""
        req = Prepare.get_load_state("test_coll", partition_names=["part1", "part2"])
        assert list(req.partition_names) == ["part1", "part2"]


class TestRenameCollectionsRequest:
    """Tests for rename_collections_request."""

    def test_rename_collection(self):
        """Test rename collection request."""
        req = Prepare.rename_collections_request("old_name", "new_name", "new_db")
        assert req.oldName == "old_name"
        assert req.newName == "new_name"
        assert req.newDBName == "new_db"


class TestDescribeCollectionRequest:
    """Tests for describe_collection_request."""

    def test_describe_collection_basic(self):
        """Test basic describe collection request."""
        req = Prepare.describe_collection_request("test_coll")
        assert req.collection_name == "test_coll"


class TestPreparePlaceholderStrEdgeCases:
    """Additional tests for _prepare_placeholder_str edge cases."""

    def test_numpy_float64(self):
        """Test with numpy float64 arrays."""
        data = [np.array([1.0, 2.0], dtype=np.float64)]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None


class TestExpressionTemplateEdgeCases:
    """Additional tests for prepare_expression_template edge cases."""

    def test_uniform_type_array(self):
        """Test with uniform type array."""
        result = Prepare.prepare_expression_template({"ids": [1, 2, 3]})
        assert "ids" in result


class TestSearchParamsEdgeCases:
    """Additional tests for search_requests_with_expr parameter handling."""

    @pytest.fixture
    def basic_params(self):
        return {"metric_type": "L2", "params": {"nprobe": 10}}

    def test_search_with_page_retain_order_from_param(self, basic_params):
        """Test page_retain_order from param dict."""
        basic_params["page_retain_order"] = True
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_params,
            limit=10,
        )
        assert req is not None

    def test_search_with_offset_from_param(self, basic_params):
        """Test offset from param dict."""
        basic_params["offset"] = 10
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_params,
            limit=10,
        )
        assert req is not None


class TestAliasRequests:
    """Tests for alias requests."""

    def test_create_alias(self):
        """Test create alias request."""
        req = Prepare.create_alias_request("test_coll", "test_alias")
        assert req.collection_name == "test_coll"
        assert req.alias == "test_alias"

    def test_drop_alias(self):
        """Test drop alias request."""
        req = Prepare.drop_alias_request("test_alias")
        assert req.alias == "test_alias"

    def test_alter_alias(self):
        """Test alter alias request."""
        req = Prepare.alter_alias_request("test_coll", "test_alias")
        assert req.collection_name == "test_coll"
        assert req.alias == "test_alias"

    def test_describe_alias(self):
        """Test describe alias request."""
        req = Prepare.describe_alias_request("test_alias")
        assert req.alias == "test_alias"

    def test_list_aliases(self):
        """Test list aliases request."""
        req = Prepare.list_aliases_request("test_coll")
        assert req.collection_name == "test_coll"


class TestIndexRequests:
    """Additional tests for index requests."""

    def test_describe_index_basic(self):
        """Test describe index request."""
        req = Prepare.describe_index_request("test_coll", "test_idx")
        assert req.collection_name == "test_coll"
        assert req.index_name == "test_idx"

    def test_describe_index_with_timestamp(self):
        """Test describe index with timestamp."""
        req = Prepare.describe_index_request("test_coll", "test_idx", timestamp=12345)
        assert req.timestamp == 12345

    def test_get_index_build_progress(self):
        """Test get index build progress."""
        req = Prepare.get_index_build_progress("test_coll", "test_idx")
        assert req.collection_name == "test_coll"
        assert req.index_name == "test_idx"

    def test_get_index_state(self):
        """Test get index state."""
        req = Prepare.get_index_state_request("test_coll", "test_idx")
        assert req.collection_name == "test_coll"
        assert req.index_name == "test_idx"

    def test_drop_index(self):
        """Test drop index request."""
        req = Prepare.drop_index_request("test_coll", "vector", "test_idx")
        assert req.collection_name == "test_coll"
        assert req.field_name == "vector"
        assert req.index_name == "test_idx"


class TestFlushRequests:
    """Tests for flush-related requests."""

    def test_flush_param(self):
        """Test flush param request."""
        req = Prepare.flush_param(["coll1", "coll2"])
        assert list(req.collection_names) == ["coll1", "coll2"]

    def test_get_flush_state(self):
        """Test get flush state request."""
        req = Prepare.get_flush_state_request([1, 2, 3], "test_coll", 12345)
        assert list(req.segmentIDs) == [1, 2, 3]
        assert req.collection_name == "test_coll"
        assert req.flush_ts == 12345

    def test_flush_all(self):
        """Test flush all request."""
        req = Prepare.flush_all_request("test_db")
        assert req.db_name == "test_db"

    def test_get_flush_all_state(self):
        """Test get flush all state request."""
        req = Prepare.get_flush_all_state_request(12345, "test_db")
        assert req.flush_all_ts == 12345
        assert req.db_name == "test_db"


class TestSegmentRequests:
    """Tests for segment-related requests."""

    def test_get_persistent_segment_info(self):
        """Test get persistent segment info request."""
        req = Prepare.get_persistent_segment_info_request("test_coll")
        assert req.collectionName == "test_coll"

    def test_get_query_segment_info(self):
        """Test get query segment info request."""
        req = Prepare.get_query_segment_info_request("test_coll")
        assert req.collectionName == "test_coll"


class TestPartitionRequests:
    """Tests for partition requests."""

    def test_create_partition(self):
        """Test create partition request."""
        req = Prepare.create_partition_request("test_coll", "test_part")
        assert req.collection_name == "test_coll"
        assert req.partition_name == "test_part"

    def test_drop_partition(self):
        """Test drop partition request."""
        req = Prepare.drop_partition_request("test_coll", "test_part")
        assert req.collection_name == "test_coll"
        assert req.partition_name == "test_part"

    def test_has_partition(self):
        """Test has partition request."""
        req = Prepare.has_partition_request("test_coll", "test_part")
        assert req.collection_name == "test_coll"
        assert req.partition_name == "test_part"


class TestGetServerVersion:
    """Tests for get_server_version."""

    def test_get_server_version(self):
        """Test get server version request."""
        req = Prepare.get_server_version()
        assert req is not None


class TestReleaseRequests:
    """Tests for release requests."""

    def test_release_collection(self):
        """Test release collection request."""
        req = Prepare.release_collection("test_db", "test_coll")
        assert req.db_name == "test_db"
        assert req.collection_name == "test_coll"

    def test_release_partitions(self):
        """Test release partitions request."""
        req = Prepare.release_partitions("test_db", "test_coll", ["part1", "part2"])
        assert req.db_name == "test_db"
        assert req.collection_name == "test_coll"
        assert list(req.partition_names) == ["part1", "part2"]


class TestGetPartitionStatsRequest:
    """Tests for get_partition_stats_request."""

    def test_get_partition_stats(self):
        """Test get partition stats request."""
        req = Prepare.get_partition_stats_request("test_coll", "test_part")
        assert req.collection_name == "test_coll"
        assert req.partition_name == "test_part"
