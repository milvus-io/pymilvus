# Copyright (C) 2019-2024 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

"""Comprehensive tests for pymilvus/client/types.py"""

from unittest.mock import MagicMock

import pytest
from pymilvus.client.types import (
    ALWAYS_KEEP_ZERO_KEYS,
    BulkInsertState,
    CompactionPlans,
    CompactionState,
    DataType,
    ExtraList,
    FunctionType,
    HighlightType,
    IndexState,
    IndexType,
    LoadState,
    MetricType,
    OmitZeroDict,
    PlaceholderType,
    Plan,
    RangeType,
    ReplicaInfo,
    State,
    Status,
    cmp_consistency_level,
    get_consistency_level,
    get_cost_from_status,
    get_extra_info,
)
from pymilvus.exceptions import AutoIDException, InvalidConsistencyLevel
from pymilvus.grpc_gen import common_pb2
from pymilvus.grpc_gen import milvus_pb2 as milvus_types


class TestStatus:
    """Tests for Status class."""

    def test_status_default_success(self):
        """Test that Status defaults to SUCCESS with 'Success' message."""
        status = Status()
        assert status.code == Status.SUCCESS
        assert status.code == 0
        assert status.message == "Success"

    def test_status_custom_error(self):
        """Test Status with custom error code and message."""
        status = Status(code=Status.UNEXPECTED_ERROR, message="Something went wrong")
        assert status.code == Status.UNEXPECTED_ERROR
        assert status.code == 1
        assert status.message == "Something went wrong"

    def test_status_repr(self):
        """Test Status repr method."""
        status = Status(code=Status.PERMISSION_DENIED, message="Access denied")
        repr_str = repr(status)
        assert "Status" in repr_str
        assert "code=3" in repr_str
        assert "message=Access denied" in repr_str

    def test_status_equality_with_int(self):
        """Test Status equality comparison with int."""
        status = Status(code=Status.SUCCESS)
        assert status == 0
        assert status == Status.SUCCESS
        assert status != 1
        assert status != Status.UNEXPECTED_ERROR

    def test_status_equality_with_status(self):
        """Test Status equality comparison with another Status."""
        status1 = Status(code=Status.SUCCESS)
        status2 = Status(code=Status.SUCCESS, message="Different message")
        status3 = Status(code=Status.UNEXPECTED_ERROR)

        assert status1 == status2  # Same code, different message
        assert status1 != status3  # Different codes
        assert status1 != "not a status"  # Different type

    def test_status_ok_method(self):
        """Test Status.OK() method."""
        success_status = Status(code=Status.SUCCESS)
        error_status = Status(code=Status.UNEXPECTED_ERROR)

        assert success_status.OK() is True
        assert error_status.OK() is False

    def test_status_error_codes(self):
        """Test various Status error codes exist and have expected values."""
        assert Status.SUCCESS == 0
        assert Status.UNEXPECTED_ERROR == 1
        assert Status.CONNECT_FAILED == 2
        assert Status.PERMISSION_DENIED == 3
        assert Status.COLLECTION_NOT_EXISTS == 4
        assert Status.ILLEGAL_ARGUMENT == 5
        assert Status.ILLEGAL_RANGE == 6
        assert Status.ILLEGAL_DIMENSION == 7
        assert Status.ILLEGAL_INDEX_TYPE == 8
        assert Status.ILLEGAL_COLLECTION_NAME == 9
        assert Status.ILLEGAL_TOPK == 10
        assert Status.ILLEGAL_ROWRECORD == 11
        assert Status.ILLEGAL_VECTOR_ID == 12
        assert Status.ILLEGAL_SEARCH_RESULT == 13
        assert Status.FILE_NOT_FOUND == 14
        assert Status.META_FAILED == 15
        assert Status.CACHE_FAILED == 16
        assert Status.CANNOT_CREATE_FOLDER == 17
        assert Status.CANNOT_CREATE_FILE == 18
        assert Status.CANNOT_DELETE_FOLDER == 19
        assert Status.CANNOT_DELETE_FILE == 20
        assert Status.BUILD_INDEX_ERROR == 21
        assert Status.ILLEGAL_NLIST == 22
        assert Status.ILLEGAL_METRIC_TYPE == 23
        assert Status.OUT_OF_MEMORY == 24
        assert Status.INDEX_NOT_EXIST == 25
        assert Status.EMPTY_COLLECTION == 26


class TestOmitZeroDict:
    """Tests for OmitZeroDict class."""

    def test_str_omits_zero_values(self):
        """Test that __str__ omits zero values."""
        d = OmitZeroDict({"a": 1, "b": 0, "c": 3, "d": 0})
        str_repr = str(d)
        assert "'a': 1" in str_repr
        assert "'c': 3" in str_repr
        assert "'b': 0" not in str_repr
        assert "'d': 0" not in str_repr

    def test_str_keeps_special_zero_keys(self):
        """Test that __str__ keeps special zero keys that are always kept."""
        d = OmitZeroDict(
            {
                "cache_hit_ratio": 0,
                "scanned_remote_bytes": 0,
                "scanned_total_bytes": 0,
                "regular_zero": 0,
                "non_zero": 1,
            }
        )
        str_repr = str(d)
        assert "'cache_hit_ratio': 0" in str_repr
        assert "'scanned_remote_bytes': 0" in str_repr
        assert "'scanned_total_bytes': 0" in str_repr
        assert "'regular_zero'" not in str_repr
        assert "'non_zero': 1" in str_repr

    def test_repr_shows_all(self):
        """Test that __repr__ shows all values including zeros."""
        d = OmitZeroDict({"a": 1, "b": 0, "c": 3})
        repr_str = repr(d)
        assert "'a': 1" in repr_str
        assert "'b': 0" in repr_str
        assert "'c': 3" in repr_str

    def test_omit_zero_len(self):
        """Test omit_zero_len method."""
        d = OmitZeroDict(
            {
                "a": 1,
                "b": 0,
                "c": 3,
                "cache_hit_ratio": 0,  # should be counted
            }
        )
        assert d.omit_zero_len() == 3  # a, c, and cache_hit_ratio

    def test_always_keep_zero_keys_constant(self):
        """Test that ALWAYS_KEEP_ZERO_KEYS contains expected keys."""
        assert "scanned_remote_bytes" in ALWAYS_KEEP_ZERO_KEYS
        assert "scanned_total_bytes" in ALWAYS_KEEP_ZERO_KEYS
        assert "cache_hit_ratio" in ALWAYS_KEEP_ZERO_KEYS


class TestState:
    """Tests for State enum."""

    def test_state_new_executing(self):
        """Test State.new returns Executing for value 1."""
        state = State.new(1)
        assert state == State.Executing

    def test_state_new_completed(self):
        """Test State.new returns Completed for value 2."""
        state = State.new(2)
        assert state == State.Completed

    def test_state_new_undefined(self):
        """Test State.new returns UndefiedState for invalid values."""
        assert State.new(0) == State.UndefiedState
        assert State.new(3) == State.UndefiedState
        assert State.new(-1) == State.UndefiedState
        assert State.new(100) == State.UndefiedState

    def test_state_repr(self):
        """Test State repr method."""
        assert "<State: Executing>" in repr(State.Executing)
        assert "<State: Completed>" in repr(State.Completed)
        assert "<State: UndefiedState>" in repr(State.UndefiedState)

    def test_state_str(self):
        """Test State str method."""
        assert str(State.Executing) == "Executing"
        assert str(State.Completed) == "Completed"
        assert str(State.UndefiedState) == "UndefiedState"

    def test_state_values(self):
        """Test State enum values."""
        assert State.UndefiedState == 0
        assert State.Executing == 1
        assert State.Completed == 2


class TestCompactionState:
    """Tests for CompactionState class."""

    def test_compaction_state_init(self):
        """Test CompactionState initialization."""
        cs = CompactionState(
            compaction_id=123,
            state=State.Executing,
            in_executing=5,
            in_timeout=2,
            completed=10,
        )
        assert cs.compaction_id == 123
        assert cs.state == State.Executing
        assert cs.in_executing == 5
        assert cs.in_timeout == 2
        assert cs.completed == 10

    def test_compaction_state_name(self):
        """Test CompactionState.state_name property."""
        cs = CompactionState(
            compaction_id=123,
            state=State.Completed,
            in_executing=0,
            in_timeout=0,
            completed=10,
        )
        assert cs.state_name == "Completed"

    def test_compaction_state_repr(self):
        """Test CompactionState repr method."""
        cs = CompactionState(
            compaction_id=456,
            state=State.Executing,
            in_executing=3,
            in_timeout=1,
            completed=5,
        )
        repr_str = repr(cs)
        assert "CompactionState" in repr_str
        assert "456" in repr_str
        assert "Executing" in repr_str
        assert "3" in repr_str
        assert "1" in repr_str
        assert "5" in repr_str


class TestPlan:
    """Tests for Plan class."""

    def test_plan_init(self):
        """Test Plan initialization."""
        plan = Plan(sources=[1, 2, 3], target=100)
        assert plan.sources == [1, 2, 3]
        assert plan.target == 100

    def test_plan_repr(self):
        """Test Plan repr method."""
        plan = Plan(sources=[10, 20], target=200)
        repr_str = repr(plan)
        assert "Plan" in repr_str
        assert "[10, 20]" in repr_str
        assert "200" in repr_str


class TestCompactionPlans:
    """Tests for CompactionPlans class."""

    def test_compaction_plans_init(self):
        """Test CompactionPlans initialization."""
        cp = CompactionPlans(compaction_id=789, state=1)  # 1 = Executing
        assert cp.compaction_id == 789
        assert cp.state == State.Executing
        assert cp.plans == []

    def test_compaction_plans_init_completed(self):
        """Test CompactionPlans initialization with completed state."""
        cp = CompactionPlans(compaction_id=101, state=2)  # 2 = Completed
        assert cp.state == State.Completed

    def test_compaction_plans_init_undefined(self):
        """Test CompactionPlans initialization with undefined state."""
        cp = CompactionPlans(compaction_id=102, state=99)  # invalid = UndefiedState
        assert cp.state == State.UndefiedState

    def test_compaction_plans_repr(self):
        """Test CompactionPlans repr method."""
        cp = CompactionPlans(compaction_id=555, state=2)
        cp.plans = [Plan([1, 2], 10), Plan([3, 4], 20)]
        repr_str = repr(cp)
        assert "Compaction Plans" in repr_str
        assert "555" in repr_str
        assert "Completed" in repr_str


class TestCmpConsistencyLevel:
    """Tests for cmp_consistency_level function."""

    def test_cmp_same_int(self):
        """Test comparing same int consistency levels."""
        assert cmp_consistency_level(0, 0) is True
        assert cmp_consistency_level(1, 1) is True
        assert cmp_consistency_level(2, 2) is True
        assert cmp_consistency_level(3, 3) is True

    def test_cmp_different_int(self):
        """Test comparing different int consistency levels."""
        assert cmp_consistency_level(0, 1) is False
        assert cmp_consistency_level(1, 2) is False
        assert cmp_consistency_level(2, 3) is False

    def test_cmp_str_str(self):
        """Test comparing string consistency levels."""
        assert cmp_consistency_level("Strong", "Strong") is True
        assert cmp_consistency_level("Session", "Session") is True
        assert cmp_consistency_level("Bounded", "Bounded") is True
        assert cmp_consistency_level("Eventually", "Eventually") is True
        assert cmp_consistency_level("Strong", "Session") is False
        assert cmp_consistency_level("Bounded", "Eventually") is False

    def test_cmp_str_int(self):
        """Test comparing string with int consistency levels."""
        # Strong = 0, Session = 1, Bounded = 2, Eventually = 3, Customized = 4
        assert cmp_consistency_level("Strong", 0) is True
        assert cmp_consistency_level(0, "Strong") is True
        assert cmp_consistency_level("Session", 1) is True
        assert cmp_consistency_level("Bounded", 2) is True
        assert cmp_consistency_level("Eventually", 3) is True
        assert cmp_consistency_level("Customized", 4) is True
        assert cmp_consistency_level("Strong", 1) is False

    def test_cmp_invalid_str(self):
        """Test comparing with invalid string returns False."""
        assert cmp_consistency_level("InvalidLevel", "Strong") is False
        assert cmp_consistency_level("Strong", "InvalidLevel") is False
        assert cmp_consistency_level("NotReal", "AlsoNotReal") is False

    def test_cmp_invalid_int(self):
        """Test comparing with invalid int returns False."""
        assert cmp_consistency_level(999, 0) is False
        assert cmp_consistency_level(0, 999) is False
        assert cmp_consistency_level(100, 100) is False


class TestLoadState:
    """Tests for LoadState enum."""

    def test_load_state_values(self):
        """Test LoadState enum values."""
        assert LoadState.NotExist == 0
        assert LoadState.NotLoad == 1
        assert LoadState.Loading == 2
        assert LoadState.Loaded == 3

    def test_load_state_repr(self):
        """Test LoadState repr method."""
        assert "<LoadState: NotExist>" in repr(LoadState.NotExist)
        assert "<LoadState: NotLoad>" in repr(LoadState.NotLoad)
        assert "<LoadState: Loading>" in repr(LoadState.Loading)
        assert "<LoadState: Loaded>" in repr(LoadState.Loaded)

    def test_load_state_str(self):
        """Test LoadState str method."""
        assert str(LoadState.NotExist) == "NotExist"
        assert str(LoadState.NotLoad) == "NotLoad"
        assert str(LoadState.Loading) == "Loading"
        assert str(LoadState.Loaded) == "Loaded"


class TestIndexState:
    """Tests for IndexState enum."""

    def test_index_state_values(self):
        """Test IndexState enum values."""
        assert IndexState.IndexStateNone == 0
        assert IndexState.Unissued == 1
        assert IndexState.InProgress == 2
        assert IndexState.Finished == 3
        assert IndexState.Failed == 4
        assert IndexState.Deleted == 5


class TestIndexType:
    """Tests for IndexType enum."""

    def test_index_type_values(self):
        """Test IndexType enum values."""
        assert IndexType.INVALID == 0
        assert IndexType.FLAT == 1
        assert IndexType.IVFLAT == 2
        assert IndexType.IVF_SQ8 == 3
        assert IndexType.RNSG == 4
        assert IndexType.IVF_SQ8H == 5
        assert IndexType.IVF_PQ == 6
        assert IndexType.HNSW == 11
        assert IndexType.ANNOY == 12

    def test_index_type_aliases(self):
        """Test IndexType alternative names."""
        assert IndexType.IVF_FLAT == IndexType.IVFLAT
        assert IndexType.IVF_SQ8_H == IndexType.IVF_SQ8H

    def test_index_type_repr(self):
        """Test IndexType repr method."""
        assert "<IndexType: FLAT>" in repr(IndexType.FLAT)
        assert "<IndexType: HNSW>" in repr(IndexType.HNSW)

    def test_index_type_str(self):
        """Test IndexType str method."""
        assert str(IndexType.FLAT) == "FLAT"
        assert str(IndexType.HNSW) == "HNSW"
        assert str(IndexType.IVF_PQ) == "IVF_PQ"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.INVALID == 0
        assert MetricType.L2 == 1
        assert MetricType.IP == 2
        assert MetricType.HAMMING == 3
        assert MetricType.JACCARD == 4
        assert MetricType.TANIMOTO == 5
        assert MetricType.SUBSTRUCTURE == 6
        assert MetricType.SUPERSTRUCTURE == 7

    def test_metric_type_repr(self):
        """Test MetricType repr method."""
        assert "<MetricType: L2>" in repr(MetricType.L2)
        assert "<MetricType: IP>" in repr(MetricType.IP)

    def test_metric_type_str(self):
        """Test MetricType str method."""
        assert str(MetricType.L2) == "L2"
        assert str(MetricType.IP) == "IP"
        assert str(MetricType.HAMMING) == "HAMMING"


class TestDataType:
    """Tests for DataType enum."""

    def test_data_type_values(self):
        """Test DataType enum values."""
        assert DataType.NONE == 0
        assert DataType.BOOL == 1
        assert DataType.INT8 == 2
        assert DataType.INT16 == 3
        assert DataType.INT32 == 4
        assert DataType.INT64 == 5
        assert DataType.FLOAT == 10
        assert DataType.DOUBLE == 11

    def test_data_type_str(self):
        """Test DataType str is its numeric value."""
        assert str(DataType.BOOL) == "1"
        assert str(DataType.INT64) == "5"
        assert str(DataType.FLOAT) == "10"


class TestFunctionType:
    """Tests for FunctionType enum."""

    def test_function_type_values(self):
        """Test FunctionType enum values."""
        assert FunctionType.UNKNOWN == 0
        assert FunctionType.BM25 == 1
        assert FunctionType.TEXTEMBEDDING == 2
        assert FunctionType.RERANK == 3
        assert FunctionType.MINHASH == 4


class TestHighlightType:
    """Tests for HighlightType enum."""

    def test_highlight_type_values(self):
        """Test HighlightType enum values."""
        assert HighlightType.LEXICAL == 0
        assert HighlightType.SEMANTIC == 1


class TestRangeType:
    """Tests for RangeType enum."""

    def test_range_type_values(self):
        """Test RangeType enum values."""
        assert RangeType.LT == 0
        assert RangeType.LTE == 1
        assert RangeType.EQ == 2
        assert RangeType.GT == 3
        assert RangeType.GTE == 4
        assert RangeType.NE == 5


class TestPlaceholderType:
    """Tests for PlaceholderType enum."""

    def test_placeholder_type_values(self):
        """Test PlaceholderType enum values."""
        assert PlaceholderType.NoneType == 0
        assert PlaceholderType.BinaryVector == 100
        assert PlaceholderType.FloatVector == 101
        assert PlaceholderType.FLOAT16_VECTOR == 102
        assert PlaceholderType.BFLOAT16_VECTOR == 103
        assert PlaceholderType.SparseFloatVector == 104
        assert PlaceholderType.Int8Vector == 105
        assert PlaceholderType.VARCHAR == 21

    def test_placeholder_type_emb_list_values(self):
        """Test PlaceholderType embedding list values."""
        assert PlaceholderType.EmbListBinaryVector == 300
        assert PlaceholderType.EmbListFloatVector == 301
        assert PlaceholderType.EmbListFloat16Vector == 302
        assert PlaceholderType.EmbListBFloat16Vector == 303
        assert PlaceholderType.EmbListSparseFloatVector == 304
        assert PlaceholderType.EmbListInt8Vector == 305


class TestReplicaInfo:
    """Tests for ReplicaInfo class."""

    def test_replica_info_init(self):
        """Test ReplicaInfo initialization."""
        replica = ReplicaInfo(
            replica_id=1,
            shards=["shard1", "shard2"],
            nodes=[1, 2, 3],
            resource_group="default",
            num_outbound_node={"rg1": 1},
        )
        assert replica.id == 1
        assert replica.shards == ["shard1", "shard2"]
        assert replica.group_nodes == (1, 2, 3)
        assert replica.resource_group == "default"
        assert replica.num_outbound_node == {"rg1": 1}

    def test_replica_info_repr(self):
        """Test ReplicaInfo repr method."""
        replica = ReplicaInfo(
            replica_id=42,
            shards=["s1"],
            nodes=[10, 20],
            resource_group="rg_test",
            num_outbound_node={},
        )
        repr_str = repr(replica)
        assert "ReplicaInfo" in repr_str
        assert "42" in repr_str
        assert "(10, 20)" in repr_str
        assert "rg_test" in repr_str


class TestBulkInsertState:
    """Tests for BulkInsertState class."""

    def test_bulk_insert_state_constants(self):
        """Test BulkInsertState class constants."""
        assert BulkInsertState.ImportPending == 0
        assert BulkInsertState.ImportFailed == 1
        assert BulkInsertState.ImportStarted == 2
        assert BulkInsertState.ImportPersisted == 5
        assert BulkInsertState.ImportCompleted == 6
        assert BulkInsertState.ImportFailedAndCleaned == 7
        assert BulkInsertState.ImportUnknownState == 100

    def test_bulk_insert_state_info_keys(self):
        """Test BulkInsertState info key constants."""
        assert BulkInsertState.FAILED_REASON == "failed_reason"
        assert BulkInsertState.IMPORT_FILES == "files"
        assert BulkInsertState.IMPORT_COLLECTION == "collection"
        assert BulkInsertState.IMPORT_PARTITION == "partition"
        assert BulkInsertState.IMPORT_PROGRESS == "progress_percent"

    def test_bulk_insert_state_2_name(self):
        """Test state_2_name mapping."""
        assert BulkInsertState.state_2_name[BulkInsertState.ImportPending] == "Pending"
        assert BulkInsertState.state_2_name[BulkInsertState.ImportFailed] == "Failed"
        assert BulkInsertState.state_2_name[BulkInsertState.ImportStarted] == "Started"
        assert BulkInsertState.state_2_name[BulkInsertState.ImportPersisted] == "Persisted"
        assert BulkInsertState.state_2_name[BulkInsertState.ImportCompleted] == "Completed"
        assert (
            BulkInsertState.state_2_name[BulkInsertState.ImportFailedAndCleaned]
            == "Failed and cleaned"
        )
        assert BulkInsertState.state_2_name[BulkInsertState.ImportUnknownState] == "Unknown"

    def test_bulk_insert_state_init(self):
        """Test BulkInsertState initialization."""
        # Create mock infos
        mock_infos = []
        for key, value in [
            ("files", "test.json"),
            ("collection", "test_collection"),
            ("partition", "test_partition"),
            ("failed_reason", ""),
            ("progress_percent", "50"),
        ]:
            kv = MagicMock()
            kv.key = key
            kv.value = value
            mock_infos.append(kv)

        state = BulkInsertState(
            task_id=12345,
            state=common_pb2.ImportPending,
            row_count=1000,
            id_ranges=[1, 100, 200, 250],
            infos=mock_infos,
            create_ts=1661398759,
        )

        assert state.task_id == 12345
        assert state.row_count == 1000
        assert state.id_ranges == [1, 100, 200, 250]
        assert state.files == "test.json"
        assert state.collection_name == "test_collection"
        assert state.partition_name == "test_partition"
        assert state.failed_reason == ""
        assert state.progress == 50
        assert state.create_timestamp == 1661398759

    def test_bulk_insert_state_ids(self):
        """Test BulkInsertState.ids property with valid ranges."""
        mock_infos = []
        state = BulkInsertState(
            task_id=1,
            state=common_pb2.ImportCompleted,
            row_count=10,
            id_ranges=[1, 5, 10, 13],  # ranges: [1,5), [10,13)
            infos=mock_infos,
            create_ts=0,
        )

        ids = state.ids
        # [1, 5) = [1, 2, 3, 4], [10, 13) = [10, 11, 12]
        assert ids == [1, 2, 3, 4, 10, 11, 12]

    def test_bulk_insert_state_ids_invalid_ranges(self):
        """Test BulkInsertState.ids raises exception for odd length ranges."""
        mock_infos = []
        state = BulkInsertState(
            task_id=1,
            state=common_pb2.ImportCompleted,
            row_count=10,
            id_ranges=[1, 5, 10],  # odd length - invalid
            infos=mock_infos,
            create_ts=0,
        )

        with pytest.raises(AutoIDException):
            _ = state.ids

    def test_bulk_insert_state_repr(self):
        """Test BulkInsertState repr method."""
        mock_infos = []
        state = BulkInsertState(
            task_id=999,
            state=common_pb2.ImportPending,
            row_count=500,
            id_ranges=[],
            infos=mock_infos,
            create_ts=1661398759,
        )

        repr_str = repr(state)
        assert "Bulk insert state" in repr_str
        assert "999" in repr_str
        assert "500" in repr_str

    def test_bulk_insert_state_create_time_str(self):
        """Test BulkInsertState.create_time_str property."""
        mock_infos = []
        # Use a known timestamp
        ts = 1704067200  # 2024-01-01 00:00:00 UTC
        state = BulkInsertState(
            task_id=1,
            state=common_pb2.ImportCompleted,
            row_count=10,
            id_ranges=[],
            infos=mock_infos,
            create_ts=ts,
        )

        time_str = state.create_time_str
        # Just verify it's a formatted string
        assert len(time_str) == 19  # "YYYY-MM-DD HH:MM:SS"
        assert "-" in time_str
        assert ":" in time_str


class TestGetConsistencyLevel:
    """Tests for get_consistency_level function."""

    def test_get_consistency_level_valid_int(self):
        """Test get_consistency_level with valid int values."""
        for v in common_pb2.ConsistencyLevel.values():
            assert get_consistency_level(v) == v

    def test_get_consistency_level_valid_str(self):
        """Test get_consistency_level with valid string values."""
        for k in common_pb2.ConsistencyLevel.keys():
            expected = common_pb2.ConsistencyLevel.Value(k)
            assert get_consistency_level(k) == expected

    def test_get_consistency_level_invalid_int(self):
        """Test get_consistency_level with invalid int raises exception."""
        with pytest.raises(InvalidConsistencyLevel):
            get_consistency_level(999)

    def test_get_consistency_level_invalid_str(self):
        """Test get_consistency_level with invalid string raises exception."""
        with pytest.raises(InvalidConsistencyLevel):
            get_consistency_level("NotAValidLevel")

    def test_get_consistency_level_invalid_type(self):
        """Test get_consistency_level with invalid type raises exception."""
        with pytest.raises(InvalidConsistencyLevel):
            get_consistency_level(1.5)


class TestExtraList:
    """Tests for ExtraList class."""

    def test_extra_list_init(self):
        """Test ExtraList initialization."""
        el = ExtraList([1, 2, 3], extra={"cost": 10})
        assert list(el) == [1, 2, 3]
        assert el.extra["cost"] == 10

    def test_extra_list_init_no_extra(self):
        """Test ExtraList initialization without extra."""
        el = ExtraList([1, 2, 3])
        assert list(el) == [1, 2, 3]
        assert isinstance(el.extra, OmitZeroDict)

    def test_extra_list_str_with_extra(self):
        """Test ExtraList str with extra info."""
        el = ExtraList([1, 2, 3], extra={"cost": 10})
        str_repr = str(el)
        assert "data:" in str_repr
        assert "extra_info:" in str_repr
        assert "10" in str_repr

    def test_extra_list_str_without_extra(self):
        """Test ExtraList str without extra info."""
        el = ExtraList([1, 2, 3], extra={})
        str_repr = str(el)
        assert "data:" in str_repr
        # No extra_info when empty
        assert "extra_info" not in str_repr

    def test_extra_list_str_long_list(self):
        """Test ExtraList str truncates long lists."""
        el = ExtraList(list(range(20)), extra={"cost": 5})
        str_repr = str(el)
        assert "..." in str_repr

    def test_extra_list_repr(self):
        """Test ExtraList repr is same as str."""
        el = ExtraList([1, 2, 3], extra={"cost": 10})
        assert str(el) == repr(el)


class TestGetCostFromStatus:
    """Tests for get_cost_from_status function."""

    def test_get_cost_from_status_with_value(self):
        """Test get_cost_from_status with report_value."""
        status = MagicMock()
        status.extra_info = {"report_value": "42"}
        assert get_cost_from_status(status) == 42

    def test_get_cost_from_status_no_value(self):
        """Test get_cost_from_status without report_value."""
        status = MagicMock()
        status.extra_info = {}
        assert get_cost_from_status(status) == 0

    def test_get_cost_from_status_none(self):
        """Test get_cost_from_status with None."""
        assert get_cost_from_status(None) == 0

    def test_get_cost_from_status_no_extra_info(self):
        """Test get_cost_from_status without extra_info."""
        status = MagicMock()
        status.extra_info = None
        assert get_cost_from_status(status) == 0


class TestGetExtraInfo:
    """Tests for get_extra_info function."""

    def test_get_extra_info_full(self):
        """Test get_extra_info with all values."""
        status = MagicMock()
        status.extra_info = {
            "report_value": "100",
            "scanned_remote_bytes": "1024",
            "scanned_total_bytes": "2048",
            "cache_hit_ratio": "0.75",
        }
        extra = get_extra_info(status)
        assert extra["cost"] == 100
        assert extra["scanned_remote_bytes"] == 1024
        assert extra["scanned_total_bytes"] == 2048
        assert extra["cache_hit_ratio"] == 0.75

    def test_get_extra_info_partial(self):
        """Test get_extra_info with partial values."""
        status = MagicMock()
        status.extra_info = {"report_value": "50"}
        extra = get_extra_info(status)
        assert extra["cost"] == 50
        assert "scanned_remote_bytes" not in extra
        assert "scanned_total_bytes" not in extra
        assert "cache_hit_ratio" not in extra

    def test_get_extra_info_none(self):
        """Test get_extra_info with None."""
        extra = get_extra_info(None)
        assert extra["cost"] == 0


# Import additional classes for testing
from pymilvus.client.types import (
    AnalyzeResult,
    AnalyzeToken,
    DatabaseInfo,
    GrantInfo,
    GrantItem,
    Group,
    HybridExtraList,
    NodeInfo,
    PrivilegeGroupInfo,
    PrivilegeGroupItem,
    Replica,
    ResourceGroupInfo,
    RoleInfo,
    RoleItem,
    Shard,
    UserInfo,
    UserItem,
)


class TestShard:
    """Tests for Shard class."""

    def test_shard_init(self):
        """Test Shard initialization."""
        shard = Shard(channel_name="test-channel", shard_nodes=[1, 2, 3], shard_leader=1)
        assert shard.channel_name == "test-channel"
        assert shard.shard_nodes == {1, 2, 3}
        assert shard.shard_leader == 1

    def test_shard_repr(self):
        """Test Shard repr method."""
        shard = Shard(channel_name="my-channel", shard_nodes=[10, 20], shard_leader=10)
        repr_str = repr(shard)
        assert "Shard" in repr_str
        assert "my-channel" in repr_str
        assert "10" in repr_str


class TestGroup:
    """Tests for Group class."""

    def test_group_init(self):
        """Test Group initialization."""
        shard = Shard("channel-1", [1, 2], 1)
        group = Group(
            group_id=1,
            shards=[shard],
            group_nodes=[1, 2, 3],
            resource_group="default",
            num_outbound_node={"rg1": 2},
        )
        assert group.id == 1
        assert group.shards == [shard]
        assert group.group_nodes == (1, 2, 3)
        assert group.resource_group == "default"
        assert group.num_outbound_node == {"rg1": 2}

    def test_group_repr(self):
        """Test Group repr method."""
        shard = Shard("ch1", [1], 1)
        group = Group(
            group_id=42,
            shards=[shard],
            group_nodes=[10, 20],
            resource_group="rg_test",
            num_outbound_node={},
        )
        repr_str = repr(group)
        assert "Group" in repr_str
        assert "42" in repr_str
        assert "rg_test" in repr_str


class TestReplica:
    """Tests for Replica class."""

    def test_replica_init(self):
        """Test Replica initialization."""
        shard = Shard("ch1", [1], 1)
        group = Group(1, [shard], [1, 2], "default", {})
        replica = Replica(groups=[group])
        assert replica.groups == [group]

    def test_replica_repr(self):
        """Test Replica repr method."""
        shard = Shard("ch1", [1], 1)
        group = Group(1, [shard], [1, 2], "default", {})
        replica = Replica(groups=[group, group])
        repr_str = repr(replica)
        assert "Replica groups" in repr_str


class TestGrantItem:
    """Tests for GrantItem class."""

    def test_grant_item_init(self):
        """Test GrantItem initialization."""
        # Create mock entity
        entity = MagicMock()
        entity.object.name = "Collection"
        entity.object_name = "test_collection"
        entity.db_name = "test_db"
        entity.role.name = "admin"
        entity.grantor.user.name = "root"
        entity.grantor.privilege.name = "Insert"

        item = GrantItem(entity)
        assert item.object == "Collection"
        assert item.object_name == "test_collection"
        assert item.db_name == "test_db"
        assert item.role_name == "admin"
        assert item.grantor_name == "root"
        assert item.privilege == "Insert"

    def test_grant_item_repr(self):
        """Test GrantItem repr method."""
        entity = MagicMock()
        entity.object.name = "Global"
        entity.object_name = "*"
        entity.db_name = ""
        entity.role.name = "public"
        entity.grantor.user.name = "root"
        entity.grantor.privilege.name = "CreateCollection"

        item = GrantItem(entity)
        repr_str = repr(item)
        assert "GrantItem" in repr_str
        assert "Global" in repr_str
        assert "public" in repr_str
        assert "CreateCollection" in repr_str

    def test_grant_item_iter(self):
        """Test GrantItem __iter__ method."""
        entity = MagicMock()
        entity.object.name = "Collection"
        entity.object_name = "coll1"
        entity.db_name = "db1"
        entity.role.name = "role1"
        entity.grantor.user.name = "user1"
        entity.grantor.privilege.name = "Insert"

        item = GrantItem(entity)
        d = dict(item)
        assert d["object_type"] == "Collection"
        assert d["object_name"] == "coll1"
        assert d["db_name"] == "db1"
        assert d["role_name"] == "role1"
        assert d["privilege"] == "Insert"
        assert d["grantor_name"] == "user1"

    def test_grant_item_iter_no_db_name(self):
        """Test GrantItem __iter__ without db_name."""
        entity = MagicMock()
        entity.object.name = "Collection"
        entity.object_name = "coll1"
        entity.db_name = ""
        entity.role.name = "role1"
        entity.grantor.user.name = ""
        entity.grantor.privilege.name = "Insert"

        item = GrantItem(entity)
        d = dict(item)
        assert "db_name" not in d
        assert "grantor_name" not in d


class TestGrantInfo:
    """Tests for GrantInfo class."""

    def test_grant_info_init(self):
        """Test GrantInfo initialization with grant entities."""

        # Create mock GrantEntity without spec to allow nested attributes
        entity = MagicMock()
        # Make isinstance check work
        entity.__class__ = milvus_types.GrantEntity
        entity.object.name = "Collection"
        entity.object_name = "test"
        entity.db_name = "db"
        entity.role.name = "admin"
        entity.grantor.user.name = "root"
        entity.grantor.privilege.name = "Insert"

        info = GrantInfo([entity])
        assert len(info.groups) == 1
        assert info.groups[0].object == "Collection"

    def test_grant_info_repr(self):
        """Test GrantInfo repr method."""

        entity = MagicMock()
        entity.__class__ = milvus_types.GrantEntity
        entity.object.name = "Global"
        entity.object_name = "*"
        entity.db_name = ""
        entity.role.name = "public"
        entity.grantor.user.name = "root"
        entity.grantor.privilege.name = "DescribeCollection"

        info = GrantInfo([entity])
        repr_str = repr(info)
        assert "GrantInfo groups" in repr_str

    def test_grant_info_empty(self):
        """Test GrantInfo with empty list."""
        info = GrantInfo([])
        assert info.groups == []


class TestPrivilegeGroupItem:
    """Tests for PrivilegeGroupItem class."""

    def test_privilege_group_item_init(self):
        """Test PrivilegeGroupItem initialization."""

        priv1 = MagicMock(spec=milvus_types.PrivilegeEntity)
        priv1.name = "Insert"
        priv2 = MagicMock(spec=milvus_types.PrivilegeEntity)
        priv2.name = "Delete"

        item = PrivilegeGroupItem("group1", [priv1, priv2])
        assert item.privilege_group == "group1"
        assert item.privileges == ("Insert", "Delete")

    def test_privilege_group_item_repr(self):
        """Test PrivilegeGroupItem repr method."""

        priv = MagicMock(spec=milvus_types.PrivilegeEntity)
        priv.name = "Query"

        item = PrivilegeGroupItem("readers", [priv])
        repr_str = repr(item)
        assert "PrivilegeGroupItem" in repr_str
        assert "readers" in repr_str
        assert "Query" in repr_str


class TestPrivilegeGroupInfo:
    """Tests for PrivilegeGroupInfo class."""

    def test_privilege_group_info_init(self):
        """Test PrivilegeGroupInfo initialization."""

        priv = MagicMock(spec=milvus_types.PrivilegeEntity)
        priv.name = "Insert"

        result = MagicMock(spec=milvus_types.PrivilegeGroupInfo)
        result.group_name = "writers"
        result.privileges = [priv]

        info = PrivilegeGroupInfo([result])
        assert len(info.groups) == 1
        assert info.groups[0].privilege_group == "writers"

    def test_privilege_group_info_repr(self):
        """Test PrivilegeGroupInfo repr method."""

        result = MagicMock(spec=milvus_types.PrivilegeGroupInfo)
        result.group_name = "admins"
        result.privileges = []

        info = PrivilegeGroupInfo([result])
        repr_str = repr(info)
        assert "PrivilegeGroupInfo groups" in repr_str


class TestUserItem:
    """Tests for UserItem class."""

    def test_user_item_init(self):
        """Test UserItem initialization."""

        role1 = MagicMock(spec=milvus_types.RoleEntity)
        role1.name = "admin"
        role2 = MagicMock(spec=milvus_types.RoleEntity)
        role2.name = "public"

        item = UserItem("testuser", [role1, role2])
        assert item.username == "testuser"
        assert item.roles == ("admin", "public")

    def test_user_item_repr(self):
        """Test UserItem repr method."""

        role = MagicMock(spec=milvus_types.RoleEntity)
        role.name = "reader"

        item = UserItem("myuser", [role])
        repr_str = repr(item)
        assert "UserItem" in repr_str
        assert "myuser" in repr_str
        assert "reader" in repr_str


class TestUserInfo:
    """Tests for UserInfo class."""

    def test_user_info_init(self):
        """Test UserInfo initialization."""

        role = MagicMock()
        role.__class__ = milvus_types.RoleEntity
        role.name = "admin"

        result = MagicMock()
        result.__class__ = milvus_types.UserResult
        result.user.name = "root"
        result.roles = [role]

        info = UserInfo([result])
        assert len(info.groups) == 1
        assert info.groups[0].username == "root"

    def test_user_info_repr(self):
        """Test UserInfo repr method."""

        result = MagicMock()
        result.__class__ = milvus_types.UserResult
        result.user.name = "user1"
        result.roles = []

        info = UserInfo([result])
        repr_str = repr(info)
        assert "UserInfo groups" in repr_str


class TestRoleItem:
    """Tests for RoleItem class."""

    def test_role_item_init(self):
        """Test RoleItem initialization."""

        user1 = MagicMock(spec=milvus_types.UserEntity)
        user1.name = "user1"
        user2 = MagicMock(spec=milvus_types.UserEntity)
        user2.name = "user2"

        item = RoleItem("admin", [user1, user2])
        assert item.role_name == "admin"
        assert item.users == ("user1", "user2")

    def test_role_item_repr(self):
        """Test RoleItem repr method."""

        user = MagicMock(spec=milvus_types.UserEntity)
        user.name = "root"

        item = RoleItem("superadmin", [user])
        repr_str = repr(item)
        assert "RoleItem" in repr_str
        assert "superadmin" in repr_str
        assert "root" in repr_str


class TestRoleInfo:
    """Tests for RoleInfo class."""

    def test_role_info_init(self):
        """Test RoleInfo initialization."""

        user = MagicMock()
        user.__class__ = milvus_types.UserEntity
        user.name = "root"

        result = MagicMock()
        result.__class__ = milvus_types.RoleResult
        result.role.name = "admin"
        result.users = [user]

        info = RoleInfo([result])
        assert len(info.groups) == 1
        assert info.groups[0].role_name == "admin"

    def test_role_info_repr(self):
        """Test RoleInfo repr method."""

        result = MagicMock()
        result.__class__ = milvus_types.RoleResult
        result.role.name = "public"
        result.users = []

        info = RoleInfo([result])
        repr_str = repr(info)
        assert "RoleInfo groups" in repr_str


class TestNodeInfo:
    """Tests for NodeInfo class."""

    def test_node_info_init(self):
        """Test NodeInfo initialization."""
        info = MagicMock()
        info.node_id = 1
        info.address = "127.0.0.1:9091"
        info.hostname = "localhost"

        node = NodeInfo(info)
        assert node.node_id == 1
        assert node.address == "127.0.0.1:9091"
        assert node.hostname == "localhost"

    def test_node_info_repr(self):
        """Test NodeInfo repr method."""
        info = MagicMock()
        info.node_id = 42
        info.address = "10.0.0.1:8080"
        info.hostname = "node-42"

        node = NodeInfo(info)
        repr_str = repr(node)
        assert "NodeInfo" in repr_str
        assert "42" in repr_str
        assert "10.0.0.1:8080" in repr_str
        assert "node-42" in repr_str


class TestResourceGroupInfo:
    """Tests for ResourceGroupInfo class."""

    def test_resource_group_info_init(self):
        """Test ResourceGroupInfo initialization."""
        node_info = MagicMock()
        node_info.node_id = 1
        node_info.address = "127.0.0.1"
        node_info.hostname = "host1"

        rg = MagicMock()
        rg.name = "default_rg"
        rg.capacity = 10
        rg.num_available_node = 5
        rg.num_loaded_replica = 3
        rg.num_outgoing_node = 1
        rg.num_incoming_node = 2
        rg.config = {"key": "value"}
        rg.nodes = [node_info]

        info = ResourceGroupInfo(rg)
        assert info.name == "default_rg"
        assert info.capacity == 10
        assert info.num_available_node == 5
        assert info.num_loaded_replica == 3
        assert info.num_outgoing_node == 1
        assert info.num_incoming_node == 2
        assert info.config == {"key": "value"}
        assert len(info.nodes) == 1

    def test_resource_group_info_repr(self):
        """Test ResourceGroupInfo repr method."""
        rg = MagicMock()
        rg.name = "test_rg"
        rg.capacity = 5
        rg.num_available_node = 3
        rg.num_loaded_replica = 2
        rg.num_outgoing_node = 0
        rg.num_incoming_node = 1
        rg.config = {}
        rg.nodes = []

        info = ResourceGroupInfo(rg)
        repr_str = repr(info)
        assert "ResourceGroupInfo" in repr_str
        assert "test_rg" in repr_str


class TestDatabaseInfo:
    """Tests for DatabaseInfo class."""

    def test_database_info_init(self):
        """Test DatabaseInfo initialization."""
        prop1 = MagicMock()
        prop1.key = "max_collections"
        prop1.value = "100"

        info = MagicMock()
        info.db_name = "test_db"
        info.properties = [prop1]

        db_info = DatabaseInfo(info)
        assert db_info.name == "test_db"
        assert db_info.properties == {"max_collections": "100"}

    def test_database_info_str(self):
        """Test DatabaseInfo str method."""
        info = MagicMock()
        info.db_name = "my_db"
        info.properties = []

        db_info = DatabaseInfo(info)
        str_repr = str(db_info)
        assert "DatabaseInfo" in str_repr
        assert "my_db" in str_repr

    def test_database_info_to_dict(self):
        """Test DatabaseInfo to_dict method."""
        prop = MagicMock()
        prop.key = "setting1"
        prop.value = "val1"

        info = MagicMock()
        info.db_name = "db1"
        info.properties = [prop]

        db_info = DatabaseInfo(info)
        d = db_info.to_dict()
        assert d["name"] == "db1"
        assert d["setting1"] == "val1"


class TestAnalyzeToken:
    """Tests for AnalyzeToken class."""

    def test_analyze_token_basic(self):
        """Test AnalyzeToken basic initialization."""
        token = MagicMock()
        token.token = "hello"

        at = AnalyzeToken(token, with_hash=False, with_detail=False)
        assert at.token == "hello"
        assert at["token"] == "hello"

    def test_analyze_token_with_detail(self):
        """Test AnalyzeToken with detail information."""
        token = MagicMock()
        token.token = "world"
        token.start_offset = 0
        token.end_offset = 5
        token.position = 1
        token.position_length = 1

        at = AnalyzeToken(token, with_hash=False, with_detail=True)
        assert at.token == "world"
        assert at.start_offset == 0
        assert at.end_offset == 5
        assert at.position == 1
        assert at.position_length == 1

    def test_analyze_token_with_hash(self):
        """Test AnalyzeToken with hash."""
        token = MagicMock()
        token.token = "test"
        token.hash = 12345

        at = AnalyzeToken(token, with_hash=True, with_detail=False)
        assert at.token == "test"
        assert at.hash == 12345

    def test_analyze_token_str(self):
        """Test AnalyzeToken str method."""
        token = MagicMock()
        token.token = "sample"

        at = AnalyzeToken(token)
        str_repr = str(at)
        assert "sample" in str_repr

    def test_analyze_token_repr(self):
        """Test AnalyzeToken repr is same as str."""
        token = MagicMock()
        token.token = "example"

        at = AnalyzeToken(token)
        assert str(at) == repr(at)


class TestAnalyzeResult:
    """Tests for AnalyzeResult class."""

    def test_analyze_result_basic(self):
        """Test AnalyzeResult basic initialization."""
        token1 = MagicMock()
        token1.token = "hello"
        token2 = MagicMock()
        token2.token = "world"

        info = MagicMock()
        info.tokens = [token1, token2]

        result = AnalyzeResult(info, with_hash=False, with_detail=False)
        assert result.tokens == ["hello", "world"]

    def test_analyze_result_with_detail(self):
        """Test AnalyzeResult with detail."""
        token = MagicMock()
        token.token = "test"
        token.start_offset = 0
        token.end_offset = 4
        token.position = 0
        token.position_length = 1

        info = MagicMock()
        info.tokens = [token]

        result = AnalyzeResult(info, with_hash=False, with_detail=True)
        assert len(result.tokens) == 1
        assert isinstance(result.tokens[0], AnalyzeToken)
        assert result.tokens[0].token == "test"

    def test_analyze_result_str(self):
        """Test AnalyzeResult str method."""
        token = MagicMock()
        token.token = "sample"

        info = MagicMock()
        info.tokens = [token]

        result = AnalyzeResult(info)
        str_repr = str(result)
        assert "sample" in str_repr

    def test_analyze_result_repr(self):
        """Test AnalyzeResult repr is same as str."""
        token = MagicMock()
        token.token = "example"

        info = MagicMock()
        info.tokens = [token]

        result = AnalyzeResult(info)
        assert str(result) == repr(result)


class TestHybridExtraList:
    """Tests for HybridExtraList class."""

    def test_hybrid_extra_list_init(self):
        """Test HybridExtraList initialization."""
        hel = HybridExtraList(lazy_field_data=[], extra={"cost": 10})
        assert hel.extra["cost"] == 10
        assert hel._lazy_field_data == []

    def test_hybrid_extra_list_basic_access(self):
        """Test HybridExtraList basic element access."""
        hel = HybridExtraList(lazy_field_data=[], extra={"cost": 5})
        hel.append({"id": 1, "value": "a"})
        hel.append({"id": 2, "value": "b"})
        # Initialize materialized bitmap after appending
        hel._materialized_bitmap = [False] * len(hel)

        assert hel[0]["id"] == 1
        assert hel[1]["value"] == "b"

    def test_hybrid_extra_list_str(self):
        """Test HybridExtraList str method."""
        hel = HybridExtraList(lazy_field_data=[], extra={"cost": 10})
        hel.append({"id": 1})
        hel._materialized_bitmap = [False] * len(hel)

        str_repr = str(hel)
        assert "data:" in str_repr
        assert "extra_info:" in str_repr

    def test_hybrid_extra_list_materialize(self):
        """Test HybridExtraList materialize method."""
        hel = HybridExtraList(lazy_field_data=[], extra={})
        hel.append({"id": 1})
        hel.append({"id": 2})
        hel._materialized_bitmap = [False] * len(hel)

        result = hel.materialize()
        assert result is hel
        assert all(hel._materialized_bitmap)

    def test_hybrid_extra_list_slice(self):
        """Test HybridExtraList slice access."""
        hel = HybridExtraList(lazy_field_data=[], extra={})
        hel.append({"id": 1})
        hel.append({"id": 2})
        hel.append({"id": 3})
        hel._materialized_bitmap = [False] * len(hel)

        sliced = hel[0:2]
        assert len(sliced) == 2
        assert sliced[0]["id"] == 1
        assert sliced[1]["id"] == 2

    def test_hybrid_extra_list_iter(self):
        """Test HybridExtraList iteration."""
        hel = HybridExtraList(lazy_field_data=[], extra={})
        hel.append({"id": 1})
        hel.append({"id": 2})
        hel._materialized_bitmap = [False] * len(hel)

        ids = [item["id"] for item in hel]
        assert ids == [1, 2]

    def test_hybrid_extra_list_negative_index(self):
        """Test HybridExtraList negative index access."""
        hel = HybridExtraList(lazy_field_data=[], extra={})
        hel.append({"id": 1})
        hel.append({"id": 2})
        hel.append({"id": 3})
        hel._materialized_bitmap = [False] * len(hel)

        assert hel[-1]["id"] == 3
        assert hel[-2]["id"] == 2
