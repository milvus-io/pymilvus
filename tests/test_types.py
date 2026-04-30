# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy as np
import pytest
from pymilvus import DataType
from pymilvus.client.constants import DEFAULT_RESOURCE_GROUP
from pymilvus.client.types import (
    ConsistencyLevel,
    Group,
    LoadedSegmentInfo,
    Replica,
    SegmentInfo,
    Shard,
    get_consistency_level,
)
from pymilvus.exceptions import InvalidConsistencyLevel
from pymilvus.orm.types import (
    infer_dtype_bydata,
)


class TestTypes:
    @pytest.mark.parametrize(
        "data,expect",
        [
            ([1], DataType.FLOAT_VECTOR),
            ([True], DataType.ARRAY),
            ([1.0, 2.0], DataType.FLOAT_VECTOR),
            (["abc"], DataType.ARRAY),
            (bytes("abc", encoding="ascii"), DataType.BINARY_VECTOR),
            (1, DataType.INT64),
            (True, DataType.BOOL),
            ("abc", DataType.VARCHAR),
            (np.int8(1), DataType.INT8),
            (np.int16(1), DataType.INT16),
        ],
    )
    def test_infer_dtype_bydata(self, data, expect):
        got = infer_dtype_bydata(data)
        assert got == expect

    def test_str_of_data_type(self):
        for v in DataType:
            assert isinstance(v, DataType)
            assert str(v) == str(v.value)
            assert str(v) != v.name

    def test_mol_data_type(self):
        assert DataType.MOL.name == "MOL"
        assert isinstance(DataType.MOL.value, int)


class TestConsistencyLevel:
    def test_consistency_level_int(self):
        for v in ConsistencyLevel.values():
            assert v == get_consistency_level(v)

    def test_consistency_level_str(self):
        for k in ConsistencyLevel.keys():
            assert ConsistencyLevel.Value(k) == get_consistency_level(k)

    @pytest.mark.parametrize("invalid", [6, 100, "not supported", "中文", 1.0])
    def test_consistency_level_invalid(self, invalid):
        with pytest.raises(InvalidConsistencyLevel):
            get_consistency_level(invalid)


class TestReplica:
    def test_shard(self):
        s = Shard("channel-1", (1, 2, 3), 1)
        assert s.channel_name == "channel-1"
        assert s.shard_nodes == {1, 2, 3}
        assert s.shard_leader == 1

        g = Group(2, [s], [1, 2, 3], DEFAULT_RESOURCE_GROUP, {})
        assert g.id == 2
        assert g.shards == [s]
        assert g.group_nodes == (1, 2, 3)

        replica = Replica([g, g])
        assert replica.groups == [g, g]

    def test_shard_dup_nodeIDs(self):
        s = Shard("channel-1", (1, 1, 1), 1)
        assert s.channel_name == "channel-1"
        assert s.shard_nodes == {
            1,
        }
        assert s.shard_leader == 1


class TestSegmentInfoRepr:
    def test_segment_info_repr_shows_state_and_level_names(self):
        info = SegmentInfo(
            segment_id=123,
            collection_id=456,
            collection_name="test_col",
            num_rows=1000,
            is_sorted=True,
            state=4,  # Flushed
            level=2,  # L1
            storage_version=2,
        )
        r = repr(info)
        assert "state='Flushed'" in r
        assert "level='L1'" in r
        assert "segment_id=123" in r
        assert "collection_name='test_col'" in r

    def test_loaded_segment_info_repr_shows_all_fields(self):
        info = LoadedSegmentInfo(
            segment_id=789,
            collection_id=456,
            collection_name="test_col",
            num_rows=5000,
            is_sorted=False,
            state=3,  # Sealed
            level=1,  # L0
            storage_version=1,
            partition_id=100,
            index_name="idx_vec",
            index_id=200,
            node_ids=[1, 2, 3],
            mem_size=4096,
        )
        r = repr(info)
        assert "LoadedSegmentInfo(" in r
        assert "state='Sealed'" in r
        assert "level='L0'" in r
        assert "partition_id=100" in r
        assert "index_name='idx_vec'" in r
        assert "index_id=200" in r
        assert "node_ids=[1, 2, 3]" in r
        assert "mem_size=4096" in r

    def test_segment_info_state_name_property(self):
        info = SegmentInfo(
            segment_id=1,
            collection_id=2,
            collection_name="c",
            num_rows=0,
            is_sorted=False,
            state=2,  # Growing
            level=0,  # Legacy
            storage_version=1,
        )
        assert info.state_name == "Growing"
        assert info.level_name == "Legacy"

    @pytest.mark.parametrize(
        "state_val,expected",
        [
            (0, "SegmentStateNone"),
            (1, "NotExist"),
            (2, "Growing"),
            (3, "Sealed"),
            (4, "Flushed"),
            (5, "Flushing"),
            (6, "Dropped"),
            (7, "Importing"),
        ],
    )
    def test_segment_info_all_states(self, state_val, expected):
        info = SegmentInfo(
            segment_id=1,
            collection_id=2,
            collection_name="c",
            num_rows=0,
            is_sorted=False,
            state=state_val,
            level=0,
            storage_version=1,
        )
        assert info.state_name == expected

    @pytest.mark.parametrize(
        "level_val,expected",
        [
            (0, "Legacy"),
            (1, "L0"),
            (2, "L1"),
            (3, "L2"),
        ],
    )
    def test_segment_info_all_levels(self, level_val, expected):
        info = SegmentInfo(
            segment_id=1,
            collection_id=2,
            collection_name="c",
            num_rows=0,
            is_sorted=False,
            state=0,
            level=level_val,
            storage_version=1,
        )
        assert info.level_name == expected
