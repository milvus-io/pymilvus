# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

from enum import IntEnum

import numpy as np

from pymilvus.client.types import (
    DataType,
)

MB = 1024 * 1024
GB = 1024 * MB

DYNAMIC_FIELD_NAME = "$meta"
DEFAULT_BUCKET_NAME = "a-bucket"

TYPE_SIZE = {
    DataType.BOOL.name: 1,
    DataType.INT8.name: 8,
    DataType.INT16.name: 8,
    DataType.INT32.name: 8,
    DataType.INT64.name: 8,
    DataType.FLOAT.name: 8,
    DataType.DOUBLE.name: 8,
}

TYPE_VALIDATOR = {
    DataType.BOOL.name: lambda x: isinstance(x, bool),
    DataType.INT8.name: lambda x: isinstance(x, int) and -128 <= x <= 127,
    DataType.INT16.name: lambda x: isinstance(x, int) and -32768 <= x <= 32767,
    DataType.INT32.name: lambda x: isinstance(x, int) and -2147483648 <= x <= 2147483647,
    DataType.INT64.name: lambda x: isinstance(x, int),
    DataType.FLOAT.name: lambda x: isinstance(x, float),
    DataType.DOUBLE.name: lambda x: isinstance(x, float),
    DataType.VARCHAR.name: lambda x, max_len: isinstance(x, str) and len(x) <= max_len,
    DataType.JSON.name: lambda x: isinstance(x, dict) and len(x) <= 65535,
    DataType.FLOAT_VECTOR.name: lambda x, dim: isinstance(x, list) and len(x) == dim,
    DataType.BINARY_VECTOR.name: lambda x, dim: isinstance(x, list) and len(x) * 8 == dim,
}

NUMPY_TYPE_CREATOR = {
    DataType.BOOL.name: np.dtype("bool"),
    DataType.INT8.name: np.dtype("int8"),
    DataType.INT16.name: np.dtype("int16"),
    DataType.INT32.name: np.dtype("int32"),
    DataType.INT64.name: np.dtype("int64"),
    DataType.FLOAT.name: np.dtype("float32"),
    DataType.DOUBLE.name: np.dtype("float64"),
    DataType.VARCHAR.name: None,
    DataType.JSON.name: None,
    DataType.FLOAT_VECTOR.name: np.dtype("float32"),
    DataType.BINARY_VECTOR.name: np.dtype("uint8"),
}


class BulkFileType(IntEnum):
    NPY = 1
    JSON_RB = 2
