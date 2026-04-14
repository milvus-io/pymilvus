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
    AnalyzeResult,
    AnalyzeToken,
    BulkInsertState,
    CompactionPlans,
    CompactionState,
    DatabaseInfo,
    DataType,
    ExtraList,
    FunctionType,
    GrantInfo,
    GrantItem,
    Group,
    HighlightType,
    HybridExtraList,
    IndexState,
    IndexType,
    LoadState,
    MetricType,
    NodeInfo,
    OmitZeroDict,
    PlaceholderType,
    Plan,
    PrivilegeGroupInfo,
    PrivilegeGroupItem,
    RangeType,
    Replica,
    ReplicaInfo,
    ResourceGroupInfo,
    RoleInfo,
    RoleItem,
    Shard,
    State,
    Status,
    UserInfo,
    UserItem,
    cmp_consistency_level,
    get_consistency_level,
    get_cost_from_status,
    get_extra_info,
)
from pymilvus.exceptions import AutoIDException, InvalidConsistencyLevel
from pymilvus.grpc_gen import common_pb2
from pymilvus.grpc_gen import milvus_pb2 as milvus_types


# Helpers
def _make_mock_entity(**kwargs):
    """Build a MagicMock that mimics a GrantEntity-like protobuf object."""
    entity = MagicMock()
    entity.object.name = kwargs.get("object_name_attr", "Collection")
    entity.object_name = kwargs.get("object_name", "test_collection")
    entity.db_name = kwargs.get("db_name", "test_db")
    entity.role.name = kwargs.get("role_name", "admin")
    entity.grantor.user.name = kwargs.get("grantor_name", "root")
    entity.grantor.privilege.name = kwargs.get("privilege", "Insert")
    return entity


def _make_user_result(username, role_names):
    """Build a mock UserResult protobuf object."""
    role_mocks = [MagicMock(__class__=milvus_types.RoleEntity, name=n) for n in role_names]
    result = MagicMock(__class__=milvus_types.UserResult)
    result.user.name = username
    result.roles = role_mocks
    return result


def _make_role_result(role_name, user_names):
    """Build a mock RoleResult protobuf object."""
    user_mocks = [MagicMock(__class__=milvus_types.UserEntity, name=n) for n in user_names]
    result = MagicMock(__class__=milvus_types.RoleResult)
    result.role.name = role_name
    result.users = user_mocks
    return result


def _make_grant_entity(**kwargs):
    """Build a mock GrantEntity protobuf object."""
    entity = _make_mock_entity(**kwargs)
    entity.__class__ = milvus_types.GrantEntity
    return entity


def _make_status(extra_info):
    """Build a MagicMock status with extra_info."""
    s = MagicMock()
    s.extra_info = extra_info
    return s


# TestStatus
class TestStatus:

    def test_status_default_success(self):
        status = Status()
        assert status.code == Status.SUCCESS and status.code == 0 and status.message == "Success"

    def test_status_custom_error(self):
        status = Status(code=Status.UNEXPECTED_ERROR, message="Something went wrong")
        assert status.code == Status.UNEXPECTED_ERROR and status.code == 1
        assert status.message == "Something went wrong"

    def test_status_repr(self):
        r = repr(Status(code=Status.PERMISSION_DENIED, message="Access denied"))
        assert "Status" in r and "code=3" in r and "message=Access denied" in r

    def test_status_equality_with_int(self):
        status = Status(code=Status.SUCCESS)
        assert status == 0 and status == Status.SUCCESS
        assert status not in (1, Status.UNEXPECTED_ERROR)

    def test_status_equality_with_status(self):
        status1 = Status(code=Status.SUCCESS)
        status2 = Status(code=Status.SUCCESS, message="Different message")
        status3 = Status(code=Status.UNEXPECTED_ERROR)
        assert status1 == status2
        assert status1 != status3
        assert status1 != "not a status"

    def test_status_ok_method(self):
        assert Status(code=Status.SUCCESS).OK() is True
        assert Status(code=Status.UNEXPECTED_ERROR).OK() is False

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("SUCCESS", 0),
            ("UNEXPECTED_ERROR", 1),
            ("CONNECT_FAILED", 2),
            ("PERMISSION_DENIED", 3),
            ("COLLECTION_NOT_EXISTS", 4),
            ("ILLEGAL_ARGUMENT", 5),
            ("ILLEGAL_RANGE", 6),
            ("ILLEGAL_DIMENSION", 7),
            ("ILLEGAL_INDEX_TYPE", 8),
            ("ILLEGAL_COLLECTION_NAME", 9),
            ("ILLEGAL_TOPK", 10),
            ("ILLEGAL_ROWRECORD", 11),
            ("ILLEGAL_VECTOR_ID", 12),
            ("ILLEGAL_SEARCH_RESULT", 13),
            ("FILE_NOT_FOUND", 14),
            ("META_FAILED", 15),
            ("CACHE_FAILED", 16),
            ("CANNOT_CREATE_FOLDER", 17),
            ("CANNOT_CREATE_FILE", 18),
            ("CANNOT_DELETE_FOLDER", 19),
            ("CANNOT_DELETE_FILE", 20),
            ("BUILD_INDEX_ERROR", 21),
            ("ILLEGAL_NLIST", 22),
            ("ILLEGAL_METRIC_TYPE", 23),
            ("OUT_OF_MEMORY", 24),
            ("INDEX_NOT_EXIST", 25),
            ("EMPTY_COLLECTION", 26),
        ],
    )
    def test_status_error_codes(self, attr, value):
        assert getattr(Status, attr) == value


# TestOmitZeroDict
class TestOmitZeroDict:

    def test_str_omits_zero_values(self):
        s = str(OmitZeroDict({"a": 1, "b": 0, "c": 3, "d": 0}))
        assert "'a': 1" in s and "'c': 3" in s
        assert "'b': 0" not in s and "'d': 0" not in s

    def test_str_keeps_special_zero_keys(self):
        s = str(
            OmitZeroDict(
                {
                    "cache_hit_ratio": 0,
                    "scanned_remote_bytes": 0,
                    "scanned_total_bytes": 0,
                    "regular_zero": 0,
                    "non_zero": 1,
                }
            )
        )
        assert all(
            f"'{k}': 0" in s
            for k in ("cache_hit_ratio", "scanned_remote_bytes", "scanned_total_bytes")
        )
        assert "'regular_zero'" not in s and "'non_zero': 1" in s

    def test_repr_shows_all(self):
        r = repr(OmitZeroDict({"a": 1, "b": 0, "c": 3}))
        assert "'a': 1" in r and "'b': 0" in r and "'c': 3" in r

    def test_omit_zero_len(self):
        # a=1, c=3, cache_hit_ratio=0 (kept), b=0 (omitted) → 3
        assert OmitZeroDict({"a": 1, "b": 0, "c": 3, "cache_hit_ratio": 0}).omit_zero_len() == 3

    def test_always_keep_zero_keys_constant(self):
        for key in ("scanned_remote_bytes", "scanned_total_bytes", "cache_hit_ratio"):
            assert key in ALWAYS_KEEP_ZERO_KEYS


# TestState
class TestState:

    def test_state_new(self):
        assert State.new(1) == State.Executing
        assert State.new(2) == State.Completed
        for v in (0, 3, -1, 100):
            assert State.new(v) == State.UndefiedState

    @pytest.mark.parametrize(
        "member,value",
        [
            ("UndefiedState", 0),
            ("Executing", 1),
            ("Completed", 2),
        ],
    )
    def test_state_values_repr_str(self, member, value):
        obj = getattr(State, member)
        assert obj == value
        assert f"<State: {member}>" in repr(obj)
        assert str(obj) == member


# TestCompactionState
class TestCompactionState:

    def test_compaction_state_init(self):
        cs = CompactionState(
            compaction_id=123, state=State.Executing, in_executing=5, in_timeout=2, completed=10
        )
        assert cs.compaction_id == 123 and cs.state == State.Executing
        assert cs.in_executing == 5 and cs.in_timeout == 2 and cs.completed == 10

    def test_compaction_state_name(self):
        cs = CompactionState(
            compaction_id=123, state=State.Completed, in_executing=0, in_timeout=0, completed=10
        )
        assert cs.state_name == "Completed"

    def test_compaction_state_repr(self):
        r = repr(
            CompactionState(
                compaction_id=456, state=State.Executing, in_executing=3, in_timeout=1, completed=5
            )
        )
        assert all(s in r for s in ["CompactionState", "456", "Executing", "3", "1", "5"])


# TestPlan
class TestPlan:

    def test_plan_init(self):
        plan = Plan(sources=[1, 2, 3], target=100)
        assert plan.sources == [1, 2, 3]
        assert plan.target == 100

    def test_plan_repr(self):
        r = repr(Plan(sources=[10, 20], target=200))
        assert "Plan" in r and "[10, 20]" in r and "200" in r


# TestCompactionPlans
class TestCompactionPlans:

    def test_compaction_plans_init(self):
        cp = CompactionPlans(compaction_id=789, state=1)
        assert cp.compaction_id == 789
        assert cp.state == State.Executing
        assert cp.plans == []
        assert CompactionPlans(compaction_id=101, state=2).state == State.Completed
        assert CompactionPlans(compaction_id=102, state=99).state == State.UndefiedState

    def test_compaction_plans_repr(self):
        cp = CompactionPlans(compaction_id=555, state=2)
        cp.plans = [Plan([1, 2], 10), Plan([3, 4], 20)]
        r = repr(cp)
        assert "Compaction Plans" in r and "555" in r and "Completed" in r


# TestCmpConsistencyLevel  (collapsed with parametrize)
class TestCmpConsistencyLevel:

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            # Same int
            (0, 0, True),
            (1, 1, True),
            (2, 2, True),
            (3, 3, True),
            # Different int
            (0, 1, False),
            (1, 2, False),
            (2, 3, False),
            # str vs str match
            ("Strong", "Strong", True),
            ("Session", "Session", True),
            ("Bounded", "Bounded", True),
            ("Eventually", "Eventually", True),
            # str vs str mismatch
            ("Strong", "Session", False),
            ("Bounded", "Eventually", False),
            # str vs int match
            ("Strong", 0, True),
            (0, "Strong", True),
            ("Session", 1, True),
            ("Bounded", 2, True),
            ("Eventually", 3, True),
            ("Customized", 4, True),
            # str vs int mismatch
            ("Strong", 1, False),
            # invalid strings
            ("InvalidLevel", "Strong", False),
            ("Strong", "InvalidLevel", False),
            ("NotReal", "AlsoNotReal", False),
            # invalid ints
            (999, 0, False),
            (0, 999, False),
            (100, 100, False),
        ],
    )
    def test_cmp_consistency_level(self, a, b, expected):
        assert cmp_consistency_level(a, b) is expected


# Repr/str parametrized tests for simple enums
@pytest.mark.parametrize(
    "enum_cls,member,value",
    [
        (LoadState, "NotExist", 0),
        (LoadState, "NotLoad", 1),
        (LoadState, "Loading", 2),
        (LoadState, "Loaded", 3),
        (IndexState, "IndexStateNone", 0),
        (IndexState, "Unissued", 1),
        (IndexState, "InProgress", 2),
        (IndexState, "Finished", 3),
        (IndexState, "Failed", 4),
        (IndexState, "Deleted", 5),
    ],
)
def test_enum_values(enum_cls, member, value):
    assert getattr(enum_cls, member) == value


@pytest.mark.parametrize(
    "enum_cls,member",
    [
        (LoadState, "NotExist"),
        (LoadState, "NotLoad"),
        (LoadState, "Loading"),
        (LoadState, "Loaded"),
        (IndexType, "FLAT"),
        (IndexType, "HNSW"),
        (MetricType, "L2"),
        (MetricType, "IP"),
    ],
)
def test_enum_repr_and_str(enum_cls, member):
    obj = getattr(enum_cls, member)
    assert f"<{enum_cls.__name__}: {member}>" in repr(obj)
    assert str(obj) == member


# TestIndexType (aliases + remaining values)
class TestIndexType:

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("INVALID", 0),
            ("FLAT", 1),
            ("IVFLAT", 2),
            ("IVF_SQ8", 3),
            ("RNSG", 4),
            ("IVF_SQ8H", 5),
            ("IVF_PQ", 6),
            ("HNSW", 11),
            ("ANNOY", 12),
        ],
    )
    def test_index_type_values(self, attr, value):
        assert getattr(IndexType, attr) == value

    def test_index_type_aliases(self):
        assert IndexType.IVF_FLAT == IndexType.IVFLAT
        assert IndexType.IVF_SQ8_H == IndexType.IVF_SQ8H

    def test_index_type_str_ivf_pq(self):
        assert str(IndexType.IVF_PQ) == "IVF_PQ"


# TestMetricType
class TestMetricType:

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("INVALID", 0),
            ("L2", 1),
            ("IP", 2),
            ("HAMMING", 3),
            ("JACCARD", 4),
            ("TANIMOTO", 5),
            ("SUBSTRUCTURE", 6),
            ("SUPERSTRUCTURE", 7),
        ],
    )
    def test_metric_type_values(self, attr, value):
        assert getattr(MetricType, attr) == value

    def test_metric_type_str_hamming(self):
        assert str(MetricType.HAMMING) == "HAMMING"


# TestDataType
class TestDataType:

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("NONE", 0),
            ("BOOL", 1),
            ("INT8", 2),
            ("INT16", 3),
            ("INT32", 4),
            ("INT64", 5),
            ("FLOAT", 10),
            ("DOUBLE", 11),
        ],
    )
    def test_data_type_values(self, attr, value):
        assert getattr(DataType, attr) == value

    def test_data_type_str(self):
        assert str(DataType.BOOL) == "1"
        assert str(DataType.INT64) == "5"
        assert str(DataType.FLOAT) == "10"


# TestFunctionType / HighlightType / RangeType / PlaceholderType
class TestFunctionType:
    @pytest.mark.parametrize(
        "attr,value",
        [
            ("UNKNOWN", 0),
            ("BM25", 1),
            ("TEXTEMBEDDING", 2),
            ("RERANK", 3),
            ("MINHASH", 4),
            ("MOL_FINGERPRINT", 5),
        ],
    )
    def test_function_type_values(self, attr, value):
        assert getattr(FunctionType, attr) == value


class TestHighlightType:
    @pytest.mark.parametrize("attr,value", [("LEXICAL", 0), ("SEMANTIC", 1)])
    def test_highlight_type_values(self, attr, value):
        assert getattr(HighlightType, attr) == value


class TestRangeType:
    @pytest.mark.parametrize(
        "attr,value",
        [
            ("LT", 0),
            ("LTE", 1),
            ("EQ", 2),
            ("GT", 3),
            ("GTE", 4),
            ("NE", 5),
        ],
    )
    def test_range_type_values(self, attr, value):
        assert getattr(RangeType, attr) == value


class TestPlaceholderType:
    @pytest.mark.parametrize(
        "attr,value",
        [
            ("NoneType", 0),
            ("BinaryVector", 100),
            ("FloatVector", 101),
            ("FLOAT16_VECTOR", 102),
            ("BFLOAT16_VECTOR", 103),
            ("SparseFloatVector", 104),
            ("Int8Vector", 105),
            ("VARCHAR", 21),
            ("EmbListBinaryVector", 300),
            ("EmbListFloatVector", 301),
            ("EmbListFloat16Vector", 302),
            ("EmbListBFloat16Vector", 303),
            ("EmbListSparseFloatVector", 304),
            ("EmbListInt8Vector", 305),
        ],
    )
    def test_placeholder_type_values(self, attr, value):
        assert getattr(PlaceholderType, attr) == value


# TestReplicaInfo
class TestReplicaInfo:
    def test_replica_info_init(self):
        replica = ReplicaInfo(
            replica_id=1,
            shards=["shard1", "shard2"],
            nodes=[1, 2, 3],
            resource_group="default",
            num_outbound_node={"rg1": 1},
        )
        assert replica.id == 1 and replica.shards == ["shard1", "shard2"]
        assert replica.group_nodes == (1, 2, 3) and replica.resource_group == "default"
        assert replica.num_outbound_node == {"rg1": 1}

    def test_replica_info_repr(self):
        r = repr(
            ReplicaInfo(
                replica_id=42,
                shards=["s1"],
                nodes=[10, 20],
                resource_group="rg_test",
                num_outbound_node={},
            )
        )
        assert all(s in r for s in ["ReplicaInfo", "42", "(10, 20)", "rg_test"])


# TestBulkInsertState
class TestBulkInsertState:
    @pytest.mark.parametrize(
        "attr,value",
        [
            ("ImportPending", 0),
            ("ImportFailed", 1),
            ("ImportStarted", 2),
            ("ImportPersisted", 5),
            ("ImportCompleted", 6),
            ("ImportFailedAndCleaned", 7),
            ("ImportUnknownState", 100),
        ],
    )
    def test_bulk_insert_state_constants(self, attr, value):
        assert getattr(BulkInsertState, attr) == value

    @pytest.mark.parametrize(
        "attr,key",
        [
            ("FAILED_REASON", "failed_reason"),
            ("IMPORT_FILES", "files"),
            ("IMPORT_COLLECTION", "collection"),
            ("IMPORT_PARTITION", "partition"),
            ("IMPORT_PROGRESS", "progress_percent"),
        ],
    )
    def test_bulk_insert_state_info_keys(self, attr, key):
        assert getattr(BulkInsertState, attr) == key

    def test_bulk_insert_state_2_name(self):
        expected = {
            BulkInsertState.ImportPending: "Pending",
            BulkInsertState.ImportFailed: "Failed",
            BulkInsertState.ImportStarted: "Started",
            BulkInsertState.ImportPersisted: "Persisted",
            BulkInsertState.ImportCompleted: "Completed",
            BulkInsertState.ImportFailedAndCleaned: "Failed and cleaned",
            BulkInsertState.ImportUnknownState: "Unknown",
        }
        assert BulkInsertState.state_2_name == expected

    def _make_state(self, task_id, state, row_count, id_ranges, infos, create_ts):
        return BulkInsertState(
            task_id=task_id,
            state=state,
            row_count=row_count,
            id_ranges=id_ranges,
            infos=infos,
            create_ts=create_ts,
        )

    def _make_infos(self, pairs):
        return [MagicMock(key=k, value=v) for k, v in pairs]

    def test_bulk_insert_state_init(self):
        infos = self._make_infos(
            [
                ("files", "test.json"),
                ("collection", "test_collection"),
                ("partition", "test_partition"),
                ("failed_reason", ""),
                ("progress_percent", "50"),
            ]
        )
        state = self._make_state(
            12345, common_pb2.ImportPending, 1000, [1, 100, 200, 250], infos, 1661398759
        )
        assert state.task_id == 12345 and state.row_count == 1000
        assert state.id_ranges == [1, 100, 200, 250]
        assert state.files == "test.json" and state.collection_name == "test_collection"
        assert state.partition_name == "test_partition" and state.failed_reason == ""
        assert state.progress == 50 and state.create_timestamp == 1661398759

    def test_bulk_insert_state_ids(self):
        state = self._make_state(1, common_pb2.ImportCompleted, 10, [1, 5, 10, 13], [], 0)
        assert state.ids == [1, 2, 3, 4, 10, 11, 12]

    def test_bulk_insert_state_ids_invalid_ranges(self):
        state = self._make_state(1, common_pb2.ImportCompleted, 10, [1, 5, 10], [], 0)
        with pytest.raises(AutoIDException):
            _ = state.ids

    def test_bulk_insert_state_repr(self):
        r = repr(self._make_state(999, common_pb2.ImportPending, 500, [], [], 1661398759))
        assert "Bulk insert state" in r and "999" in r and "500" in r

    def test_bulk_insert_state_create_time_str(self):
        t = self._make_state(1, common_pb2.ImportCompleted, 10, [], [], 1704067200).create_time_str
        assert len(t) == 19 and "-" in t and ":" in t  # "YYYY-MM-DD HH:MM:SS"


# TestGetConsistencyLevel
class TestGetConsistencyLevel:
    def test_get_consistency_level_valid_int(self):
        for v in common_pb2.ConsistencyLevel.values():
            assert get_consistency_level(v) == v

    def test_get_consistency_level_valid_str(self):
        for k in common_pb2.ConsistencyLevel.keys():
            expected = common_pb2.ConsistencyLevel.Value(k)
            assert get_consistency_level(k) == expected

    @pytest.mark.parametrize("val", [999, "NotAValidLevel", 1.5])
    def test_get_consistency_level_invalid(self, val):
        with pytest.raises(InvalidConsistencyLevel):
            get_consistency_level(val)


# TestExtraList
class TestExtraList:
    def test_extra_list_init(self):
        el = ExtraList([1, 2, 3], extra={"cost": 10})
        assert list(el) == [1, 2, 3]
        assert el.extra["cost"] == 10

    def test_extra_list_init_no_extra(self):
        el = ExtraList([1, 2, 3])
        assert list(el) == [1, 2, 3]
        assert isinstance(el.extra, OmitZeroDict)

    def test_extra_list_str_with_extra(self):
        s = str(ExtraList([1, 2, 3], extra={"cost": 10}))
        assert "data:" in s and "extra_info:" in s and "10" in s

    def test_extra_list_str_without_extra(self):
        s = str(ExtraList([1, 2, 3], extra={}))
        assert "data:" in s and "extra_info" not in s

    def test_extra_list_str_long_list(self):
        el = ExtraList(list(range(20)), extra={"cost": 5})
        assert "..." in str(el)

    def test_extra_list_repr(self):
        el = ExtraList([1, 2, 3], extra={"cost": 10})
        assert str(el) == repr(el)


# TestGetCostFromStatus
class TestGetCostFromStatus:
    def test_get_cost_from_status_with_value(self):
        assert get_cost_from_status(_make_status({"report_value": "42"})) == 42

    def test_get_cost_from_status_no_value(self):
        assert get_cost_from_status(_make_status({})) == 0

    def test_get_cost_from_status_none(self):
        assert get_cost_from_status(None) == 0

    def test_get_cost_from_status_no_extra_info(self):
        assert get_cost_from_status(_make_status(None)) == 0


# TestGetExtraInfo
class TestGetExtraInfo:
    def test_get_extra_info_full(self):
        extra = get_extra_info(
            _make_status(
                {
                    "report_value": "100",
                    "scanned_remote_bytes": "1024",
                    "scanned_total_bytes": "2048",
                    "cache_hit_ratio": "0.75",
                }
            )
        )
        assert extra == {
            "cost": 100,
            "scanned_remote_bytes": 1024,
            "scanned_total_bytes": 2048,
            "cache_hit_ratio": 0.75,
        }

    def test_get_extra_info_partial(self):
        extra = get_extra_info(_make_status({"report_value": "50"}))
        assert extra["cost"] == 50
        assert all(
            k not in extra
            for k in ("scanned_remote_bytes", "scanned_total_bytes", "cache_hit_ratio")
        )

    def test_get_extra_info_none(self):
        assert get_extra_info(None)["cost"] == 0


class TestDataClasses:

    def _make_shard(self):
        return Shard(channel_name="ch1", shard_nodes=[1, 2], shard_leader=1)

    def _make_group(self, group_id=1, resource_group="default"):
        return Group(
            group_id=group_id,
            shards=[self._make_shard()],
            group_nodes=[1, 2],
            resource_group=resource_group,
            num_outbound_node={},
        )

    # Shard
    def test_shard_init(self):
        shard = Shard(channel_name="test-channel", shard_nodes=[1, 2, 3], shard_leader=1)
        assert shard.channel_name == "test-channel" and shard.shard_nodes == {1, 2, 3}
        assert shard.shard_leader == 1

    def test_shard_repr(self):
        r = repr(Shard(channel_name="my-channel", shard_nodes=[10, 20], shard_leader=10))
        assert "Shard" in r and "my-channel" in r and "10" in r

    # Group
    def test_group_init(self):
        shard = self._make_shard()
        group = Group(
            group_id=1,
            shards=[shard],
            group_nodes=[1, 2, 3],
            resource_group="default",
            num_outbound_node={"rg1": 2},
        )
        assert group.id == 1 and group.shards == [shard]
        assert group.group_nodes == (1, 2, 3) and group.resource_group == "default"
        assert group.num_outbound_node == {"rg1": 2}

    def test_group_repr(self):
        r = repr(self._make_group(group_id=42, resource_group="rg_test"))
        assert "Group" in r and "42" in r and "rg_test" in r

    # Replica
    def test_replica_init(self):
        group = self._make_group()
        replica = Replica(groups=[group])
        assert replica.groups == [group]

    def test_replica_repr(self):
        group = self._make_group()
        repr_str = repr(Replica(groups=[group, group]))
        assert "Replica groups" in repr_str

    # NodeInfo
    def _make_node_mock(self, node_id, address, hostname):
        return MagicMock(node_id=node_id, address=address, hostname=hostname)

    def test_node_info_init(self):
        node = NodeInfo(self._make_node_mock(1, "127.0.0.1:9091", "localhost"))
        assert (
            node.node_id == 1 and node.address == "127.0.0.1:9091" and node.hostname == "localhost"
        )

    def test_node_info_repr(self):
        r = repr(NodeInfo(self._make_node_mock(42, "10.0.0.1:8080", "node-42")))
        assert all(s in r for s in ["NodeInfo", "42", "10.0.0.1:8080", "node-42"])

    # ResourceGroupInfo
    def _make_rg_mock(
        self,
        name,
        capacity=10,
        available=5,
        loaded=3,
        outgoing=1,
        incoming=2,
        config=None,
        nodes=None,
    ):
        rg = MagicMock()
        rg.name = name
        rg.capacity = capacity
        rg.num_available_node = available
        rg.num_loaded_replica = loaded
        rg.num_outgoing_node = outgoing
        rg.num_incoming_node = incoming
        rg.config = config if config is not None else {}
        rg.nodes = nodes if nodes is not None else []
        return rg

    def test_resource_group_info_init(self):
        rg = self._make_rg_mock(
            "default_rg",
            config={"key": "value"},
            nodes=[self._make_node_mock(1, "127.0.0.1", "host1")],
        )
        info = ResourceGroupInfo(rg)
        assert info.name == "default_rg" and info.capacity == 10
        assert info.num_available_node == 5 and info.num_loaded_replica == 3
        assert info.num_outgoing_node == 1 and info.num_incoming_node == 2
        assert info.config == {"key": "value"} and len(info.nodes) == 1

    def test_resource_group_info_repr(self):
        r = repr(ResourceGroupInfo(self._make_rg_mock("test_rg")))
        assert "ResourceGroupInfo" in r and "test_rg" in r

    # DatabaseInfo
    def _make_db_mock(self, db_name, props=()):
        info = MagicMock()
        info.db_name = db_name
        info.properties = [MagicMock(key=k, value=v) for k, v in props]
        return info

    def test_database_info_init(self):
        db_info = DatabaseInfo(self._make_db_mock("test_db", [("max_collections", "100")]))
        assert db_info.name == "test_db" and db_info.properties == {"max_collections": "100"}

    def test_database_info_str(self):
        s = str(DatabaseInfo(self._make_db_mock("my_db")))
        assert "DatabaseInfo" in s and "my_db" in s

    def test_database_info_to_dict(self):
        d = DatabaseInfo(self._make_db_mock("db1", [("setting1", "val1")])).to_dict()
        assert d["name"] == "db1" and d["setting1"] == "val1"


# TestGrantItem / TestGrantInfo
class TestGrantItem:
    def test_grant_item_init(self):
        item = GrantItem(_make_mock_entity())
        assert item.object == "Collection" and item.object_name == "test_collection"
        assert item.db_name == "test_db" and item.role_name == "admin"
        assert item.grantor_name == "root" and item.privilege == "Insert"

    def test_grant_item_repr(self):
        r = repr(
            GrantItem(
                _make_mock_entity(
                    object_name_attr="Global",
                    object_name="*",
                    db_name="",
                    role_name="public",
                    grantor_name="root",
                    privilege="CreateCollection",
                )
            )
        )
        assert all(s in r for s in ["GrantItem", "Global", "public", "CreateCollection"])

    def test_grant_item_iter(self):
        d = dict(
            GrantItem(
                _make_mock_entity(
                    object_name_attr="Collection",
                    object_name="coll1",
                    db_name="db1",
                    role_name="role1",
                    grantor_name="user1",
                    privilege="Insert",
                )
            )
        )
        assert d == {
            "object_type": "Collection",
            "object_name": "coll1",
            "db_name": "db1",
            "role_name": "role1",
            "privilege": "Insert",
            "grantor_name": "user1",
        }

    def test_grant_item_iter_no_db_name(self):
        d = dict(
            GrantItem(
                _make_mock_entity(
                    object_name_attr="Collection",
                    object_name="coll1",
                    db_name="",
                    role_name="role1",
                    grantor_name="",
                    privilege="Insert",
                )
            )
        )
        assert "db_name" not in d and "grantor_name" not in d


class TestGrantInfo:
    def test_grant_info_init(self):
        info = GrantInfo([_make_grant_entity()])
        assert len(info.groups) == 1 and info.groups[0].object == "Collection"

    def test_grant_info_repr(self):
        assert "GrantInfo groups" in repr(
            GrantInfo(
                [
                    _make_grant_entity(
                        object_name_attr="Global",
                        object_name="*",
                        db_name="",
                        role_name="public",
                        grantor_name="root",
                        privilege="DescribeCollection",
                    )
                ]
            )
        )

    def test_grant_info_empty(self):
        assert GrantInfo([]).groups == []


# TestPrivilegeGroupItem / TestPrivilegeGroupInfo
class TestPrivilegeGroupItem:
    def _privs(self, *names):
        privs = [MagicMock(spec=milvus_types.PrivilegeEntity) for _ in names]
        for p, n in zip(privs, names):
            p.name = n
        return privs

    def test_privilege_group_item_init(self):
        item = PrivilegeGroupItem("group1", self._privs("Insert", "Delete"))
        assert item.privilege_group == "group1" and item.privileges == ("Insert", "Delete")

    def test_privilege_group_item_repr(self):
        r = repr(PrivilegeGroupItem("readers", self._privs("Query")))
        assert "PrivilegeGroupItem" in r and "readers" in r and "Query" in r


class TestPrivilegeGroupInfo:
    def _make_pg_result(self, group_name, priv_names=()):
        result = MagicMock(spec=milvus_types.PrivilegeGroupInfo)
        result.group_name = group_name
        privs = [MagicMock(spec=milvus_types.PrivilegeEntity) for _ in priv_names]
        for p, n in zip(privs, priv_names):
            p.name = n
        result.privileges = privs
        return result

    def test_privilege_group_info_init(self):
        info = PrivilegeGroupInfo([self._make_pg_result("writers", ["Insert"])])
        assert len(info.groups) == 1 and info.groups[0].privilege_group == "writers"

    def test_privilege_group_info_repr(self):
        assert "PrivilegeGroupInfo groups" in repr(
            PrivilegeGroupInfo([self._make_pg_result("admins")])
        )


# TestUserItem / TestUserInfo / TestRoleItem / TestRoleInfo
class TestUserItem:
    def _roles(self, *names):
        roles = [MagicMock(spec=milvus_types.RoleEntity) for _ in names]
        for r, n in zip(roles, names):
            r.name = n
        return roles

    def test_user_item_init(self):
        item = UserItem("testuser", self._roles("admin", "public"))
        assert item.username == "testuser" and item.roles == ("admin", "public")

    def test_user_item_repr(self):
        r = repr(UserItem("myuser", self._roles("reader")))
        assert "UserItem" in r and "myuser" in r and "reader" in r


class TestUserInfo:
    def test_user_info_init(self):
        info = UserInfo([_make_user_result("root", ["admin"])])
        assert len(info.groups) == 1 and info.groups[0].username == "root"

    def test_user_info_repr(self):
        assert "UserInfo groups" in repr(UserInfo([_make_user_result("user1", [])]))


class TestRoleItem:
    def _users(self, *names):
        users = [MagicMock(spec=milvus_types.UserEntity) for _ in names]
        for u, n in zip(users, names):
            u.name = n
        return users

    def test_role_item_init(self):
        item = RoleItem("admin", self._users("user1", "user2"))
        assert item.role_name == "admin" and item.users == ("user1", "user2")

    def test_role_item_repr(self):
        r = repr(RoleItem("superadmin", self._users("root")))
        assert "RoleItem" in r and "superadmin" in r and "root" in r


class TestRoleInfo:
    def test_role_info_init(self):
        info = RoleInfo([_make_role_result("admin", ["root"])])
        assert len(info.groups) == 1 and info.groups[0].role_name == "admin"

    def test_role_info_repr(self):
        assert "RoleInfo groups" in repr(RoleInfo([_make_role_result("public", [])]))


# TestAnalyzeToken / TestAnalyzeResult
class TestAnalyzeToken:
    def test_analyze_token_basic(self):
        at = AnalyzeToken(MagicMock(token="hello"), with_hash=False, with_detail=False)
        assert at.token == "hello" and at["token"] == "hello"

    def test_analyze_token_with_detail(self):
        token = MagicMock(
            token="world", start_offset=0, end_offset=5, position=1, position_length=1
        )
        at = AnalyzeToken(token, with_hash=False, with_detail=True)
        assert at.token == "world" and at.start_offset == 0 and at.end_offset == 5
        assert at.position == 1 and at.position_length == 1

    def test_analyze_token_with_hash(self):
        at = AnalyzeToken(MagicMock(token="test", hash=12345), with_hash=True, with_detail=False)
        assert at.token == "test" and at.hash == 12345

    def test_analyze_token_str(self):
        assert "sample" in str(AnalyzeToken(MagicMock(token="sample")))

    def test_analyze_token_repr(self):
        at = AnalyzeToken(MagicMock(token="example"))
        assert str(at) == repr(at)


class TestAnalyzeResult:
    def _make_info(self, token_texts):
        info = MagicMock()
        info.tokens = [MagicMock(token=text) for text in token_texts]
        return info

    def test_analyze_result_basic(self):
        result = AnalyzeResult(
            self._make_info(["hello", "world"]), with_hash=False, with_detail=False
        )
        assert result.tokens == ["hello", "world"]

    def test_analyze_result_with_detail(self):
        info = MagicMock()
        info.tokens = [
            MagicMock(token="test", start_offset=0, end_offset=4, position=0, position_length=1)
        ]
        result = AnalyzeResult(info, with_hash=False, with_detail=True)
        assert len(result.tokens) == 1 and isinstance(result.tokens[0], AnalyzeToken)
        assert result.tokens[0].token == "test"

    def test_analyze_result_str(self):
        assert "sample" in str(AnalyzeResult(self._make_info(["sample"])))

    def test_analyze_result_repr(self):
        r = AnalyzeResult(self._make_info(["example"]))
        assert str(r) == repr(r)


# TestHybridExtraList
class TestHybridExtraList:
    def _make_hel(self, items=(), extra=None):
        hel = HybridExtraList(lazy_field_data=[], extra=extra or {})
        for item in items:
            hel.append(item)
        hel._materialized_bitmap = [False] * len(hel)
        return hel

    def test_hybrid_extra_list_init(self):
        hel = HybridExtraList(lazy_field_data=[], extra={"cost": 10})
        assert hel.extra["cost"] == 10 and hel._lazy_field_data == []

    def test_hybrid_extra_list_basic_access(self):
        hel = self._make_hel([{"id": 1, "value": "a"}, {"id": 2, "value": "b"}])
        assert hel[0]["id"] == 1 and hel[1]["value"] == "b"

    def test_hybrid_extra_list_str(self):
        s = str(self._make_hel([{"id": 1}], extra={"cost": 10}))
        assert "data:" in s and "extra_info:" in s

    def test_hybrid_extra_list_materialize(self):
        hel = self._make_hel([{"id": 1}, {"id": 2}])
        result = hel.materialize()
        assert result is hel and all(hel._materialized_bitmap)

    def test_hybrid_extra_list_slice(self):
        sliced = self._make_hel([{"id": 1}, {"id": 2}, {"id": 3}])[0:2]
        assert len(sliced) == 2 and sliced[0]["id"] == 1 and sliced[1]["id"] == 2

    def test_hybrid_extra_list_iter(self):
        hel = self._make_hel([{"id": 1}, {"id": 2}])
        assert [item["id"] for item in hel] == [1, 2]

    def test_hybrid_extra_list_negative_index(self):
        hel = self._make_hel([{"id": 1}, {"id": 2}, {"id": 3}])
        assert hel[-1]["id"] == 3 and hel[-2]["id"] == 2
