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

from pymilvus_orm.types import *
import pandas as pd
import numpy as np


class TestTypes:
    def test_map_numpy_dtype_to_datatype(self):
        data1 = {
            'double': [2.0],
            'float32': [np.float32(1.0)],
            'double2': [np.float64(1.0)],
            'int8': [np.int8(1)],
            'int16': [2],
            'int32': [4],
            'int64': [8],
            'bool': [True],
            'float_vec': [np.array([1.1, 1.2])],
        }

        df = pd.DataFrame(data1)

        wants1 = [
            DataType.DOUBLE,
            DataType.DOUBLE,
            DataType.DOUBLE,
            DataType.INT64,
            DataType.INT64,
            DataType.INT64,
            DataType.INT64,
            DataType.BOOL,
            DataType.UNKNOWN,
        ]

        ret1 = [map_numpy_dtype_to_datatype(x) for x in df.dtypes]
        assert ret1 == wants1

        df2 = pd.DataFrame(data=[1, 2, 3], columns=['a'],
                           dtype=np.int8)
        assert DataType.INT8 == map_numpy_dtype_to_datatype(df2.dtypes[0])

        df2 = pd.DataFrame(data=[1, 2, 3], columns=['a'],
                           dtype=np.int16)
        assert DataType.INT16 == map_numpy_dtype_to_datatype(df2.dtypes[0])

        df2 = pd.DataFrame(data=[1, 2, 3], columns=['a'],
                           dtype=np.int32)
        assert DataType.INT32 == map_numpy_dtype_to_datatype(df2.dtypes[0])

        df2 = pd.DataFrame(data=[1, 2, 3], columns=['a'],
                           dtype=np.int64)
        assert DataType.INT64 == map_numpy_dtype_to_datatype(df2.dtypes[0])

    def test_infer_dtype_bydata(self):
        data1 = [
            [1],
            [True],
            [1.0, 2.0],
            ["abc"],
            bytes("abc", encoding='ascii'),
            1,
            True,
            "abc",
            np.int8(1),
            np.int16(1),
            [np.int8(1)]
        ]

        wants = [
            DataType.FLOAT_VECTOR,
            DataType.UNKNOWN,
            DataType.FLOAT_VECTOR,
            DataType.UNKNOWN,
            DataType.BINARY_VECTOR,
            DataType.INT64,
            DataType.BOOL,
            DataType.STRING,
            DataType.INT8,
            DataType.INT16,
            DataType.FLOAT_VECTOR,
        ]

        actual = []
        for d in data1:
            actual.append(infer_dtype_bydata(d))

        assert actual == wants
