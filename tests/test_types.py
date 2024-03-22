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

from pymilvus import DataType
from pymilvus.client.constants import DEFAULT_RESOURCE_GROUP
from pymilvus.exceptions import InvalidConsistencyLevel
from pymilvus.client.types import (
    get_consistency_level,
    Shard,
    Group,
    Replica,
    ConsistencyLevel,
)
from pymilvus.orm.types import (
    infer_dtype_bydata,
)

from pymilvus.grpc_gen import common_pb2

import pytest
import pandas as pd
import numpy as np
#  from ml_dtypes import bfloat16


@pytest.mark.skip("please fix me")
class TestTypes:
    @pytest.mark.parametrize("input_expect", [
        ([1], DataType.FLOAT_VECTOR),
        ([True], DataType.UNKNOWN),
        ([1.0, 2.0], DataType.FLOAT_VECTOR),
        (["abc"], DataType.UNKNOWN),
        (bytes("abc", encoding='ascii'), DataType.BINARY_VECTOR),
        (1, DataType.INT64),
        (True, DataType.BOOL),
        ("abc", DataType.VARCHAR),
        (np.int8(1), DataType.INT8),
        (np.int16(1), DataType.INT16),
        ([np.int8(1)], DataType.FLOAT_VECTOR),
        ([np.float16(1.0)], DataType.FLOAT16_VECTOR),
        #  ([np.array([1, 1], dtype=bfloat16)], DataType.BFLOAT16_VECTOR),
    ])
    def test_infer_dtype_bydata(self, input_expect):
        data, expect = input_expect
        got = infer_dtype_bydata(data)
        assert got == expect


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
        print(s)

        g = Group(2, [s], [1, 2, 3], DEFAULT_RESOURCE_GROUP, {})
        assert g.id == 2
        assert g.shards == [s]
        assert g.group_nodes == (1, 2, 3)

        print(g)

        replica = Replica([g, g])
        assert replica.groups == [g, g]
        print(replica)

    def test_shard_dup_nodeIDs(self):
        s = Shard("channel-1", (1, 1, 1), 1)
        assert s.channel_name == "channel-1"
        assert s.shard_nodes == {1,}
        assert s.shard_leader == 1
        print(s)
