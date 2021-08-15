# Copyright (C) 2019-2021 Zilliz. All rights reserved.
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
import logging
from pandas.api.types import infer_dtype, is_list_like, is_scalar, is_float, is_array_like
import numpy as np

LOGGER = logging.getLogger(__name__)


class DataType(IntEnum):
    NONE = 0
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5

    FLOAT = 10
    DOUBLE = 11

    STRING = 20

    BINARY_VECTOR = 100
    FLOAT_VECTOR = 101

    UNKNOWN = 999


dtype_str_map = {
    "string": DataType.STRING,
    "floating": DataType.FLOAT,
    "integer": DataType.INT64,
    "mixed-integer": DataType.INT64,
    "mixed-integer-float": DataType.FLOAT,
    "boolean": DataType.BOOL,
    "mixed": DataType.UNKNOWN,
    "bytes": DataType.UNKNOWN,
}

numpy_dtype_str_map = {
    "bool_": DataType.BOOL,
    "bool": DataType.BOOL,
    "int": DataType.INT64,
    "int_": DataType.INT64,
    "intc": DataType.INT64,
    "intp": DataType.INT64,
    "int8": DataType.INT8,
    "int16": DataType.INT16,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "uint8": DataType.INT8,
    "uint16": DataType.INT16,
    "uint32": DataType.INT32,
    "uint64": DataType.INT64,
    "float": DataType.FLOAT,
    "float_": DataType.FLOAT,
    "float16": DataType.FLOAT,
    "float32": DataType.FLOAT,
    "float64": DataType.DOUBLE,
}


def is_integer_datatype(data_type):
    return data_type in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64)


def is_float_datatype(data_type):
    return data_type in (DataType.FLOAT,)


def is_numeric_datatype(data_type):
    return is_float_datatype(data_type) or is_integer_datatype(data_type)


# pylint: disable=too-many-return-statements
def infer_dtype_by_scaladata(data):
    if isinstance(data, float):
        return DataType.DOUBLE
    if isinstance(data, bool):
        return DataType.BOOL
    if isinstance(data, int):
        return DataType.INT64
    if isinstance(data, str):
        return DataType.STRING
    if isinstance(data, np.float64):
        return DataType.DOUBLE
    if isinstance(data, np.float32):
        return DataType.FLOAT
    if isinstance(data, np.int64):
        return DataType.INT64
    if isinstance(data, np.int32):
        return DataType.INT32
    if isinstance(data, np.int16):
        return DataType.INT16
    if isinstance(data, np.int8):
        return DataType.INT8
    if isinstance(data, np.bool8):
        return DataType.BOOL
    if isinstance(data, np.bool_):
        return DataType.BOOL
    if isinstance(data, bytes):
        return DataType.BINARY_VECTOR
    if is_float(data):
        return DataType.DOUBLE

    return DataType.UNKNOWN


def infer_dtype_bydata(data):
    d_type = DataType.UNKNOWN
    if is_scalar(data):
        d_type = infer_dtype_by_scaladata(data)
        return d_type

    if is_list_like(data) or is_array_like(data):
        failed = False
        try:
            type_str = infer_dtype(data)
        except TypeError:
            failed = True
        if not failed:
            d_type = dtype_str_map.get(type_str, DataType.UNKNOWN)
            if is_numeric_datatype(d_type):
                d_type = DataType.FLOAT_VECTOR
            else:
                d_type = DataType.UNKNOWN

            return d_type

    if d_type == DataType.UNKNOWN:
        try:
            elem = data[0]
        except:
            elem = None

        if elem is not None and is_scalar(elem):
            d_type = infer_dtype_by_scaladata(elem)

    if d_type == DataType.UNKNOWN:
        _dtype = getattr(data, "dtype", None)

        if _dtype is not None:
            d_type = map_numpy_dtype_to_datatype(_dtype)

    return d_type


def map_numpy_dtype_to_datatype(d_type):
    d_type_str = str(d_type)
    return numpy_dtype_str_map.get(d_type_str, DataType.UNKNOWN)
