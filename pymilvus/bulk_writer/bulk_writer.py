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

import json
import logging
from threading import Lock

import numpy as np

from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema, FieldSchema

from .buffer import (
    Buffer,
)
from .constants import (
    TYPE_SIZE,
    TYPE_VALIDATOR,
    BulkFileType,
)

logger = logging.getLogger("bulk_writer")
logger.setLevel(logging.DEBUG)


class BulkWriter:
    def __init__(
        self,
        schema: CollectionSchema,
        chunk_size: int,
        file_type: BulkFileType,
        **kwargs,
    ):
        self._schema = schema
        self._buffer_size = 0
        self._buffer_row_count = 0
        self._total_row_count = 0
        self._file_type = file_type
        self._buffer_lock = Lock()

        # the old parameter segment_size is changed to chunk_size, compatible with the legacy code
        self._chunk_size = chunk_size
        segment_size = kwargs.get("segment_size", 0)
        if segment_size > 0:
            self._chunk_size = segment_size

        if len(self._schema.fields) == 0:
            self._throw("collection schema fields list is empty")

        if self._schema.primary_field is None:
            self._throw("primary field is null")

        self._buffer = None
        self._new_buffer()

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def buffer_row_count(self):
        return self._buffer_row_count

    @property
    def total_row_count(self):
        return self._total_row_count

    @property
    def chunk_size(self):
        return self._chunk_size

    def _new_buffer(self):
        old_buffer = self._buffer
        with self._buffer_lock:
            self._buffer = Buffer(self._schema, self._file_type)
        return old_buffer

    def append_row(self, row: dict, **kwargs):
        self._verify_row(row)
        with self._buffer_lock:
            self._buffer.append_row(row)

    def commit(self, **kwargs):
        with self._buffer_lock:
            self._buffer_size = 0
            self._buffer_row_count = 0

    @property
    def data_path(self):
        return ""

    def _try_convert_json(self, field_name: str, obj: object):
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception as e:
                self._throw(
                    f"Illegal JSON value for field '{field_name}', type mismatch or illegal format, error: {e}"
                )
        return obj

    def _throw(self, msg: str):
        logger.error(msg)
        raise MilvusException(message=msg)

    def _verify_vector(self, x: object, field: FieldSchema):
        dtype = DataType(field.dtype)
        validator = TYPE_VALIDATOR[dtype.name]
        dim = field.params["dim"]
        if not validator(x, dim):
            self._throw(
                f"Illegal vector data for vector field: '{field.name}',"
                f" dim is not {dim} or type mismatch"
            )

        return len(x) * 4 if dtype == DataType.FLOAT_VECTOR else len(x) / 8

    def _verify_json(self, x: object, field: FieldSchema):
        size = 0
        validator = TYPE_VALIDATOR[DataType.JSON.name]
        if isinstance(x, str):
            size = len(x)
            x = self._try_convert_json(field.name, x)
        elif validator(x):
            size = len(json.dumps(x))
        else:
            self._throw(f"Illegal JSON value for field '{field.name}', type mismatch")

        return x, size

    def _verify_varchar(self, x: object, field: FieldSchema):
        max_len = field.params["max_length"]
        validator = TYPE_VALIDATOR[DataType.VARCHAR.name]
        if not validator(x, max_len):
            self._throw(
                f"Illegal varchar value for field '{field.name}',"
                f" length exceeds {max_len} or type mismatch"
            )

        return len(x)

    def _verify_array(self, x: object, field: FieldSchema):
        max_capacity = field.params["max_capacity"]
        element_type = field.element_type
        validator = TYPE_VALIDATOR[DataType.ARRAY.name]
        if not validator(x, max_capacity):
            self._throw(
                f"Illegal array value for field '{field.name}', length exceeds capacity or type mismatch"
            )

        row_size = 0
        if element_type.name in TYPE_SIZE:
            row_size = TYPE_SIZE[element_type.name] * len(x)
        elif element_type == DataType.VARCHAR:
            for ele in x:
                row_size = row_size + self._verify_varchar(ele, field)
        else:
            self._throw(f"Unsupported element type for array field '{field.name}'")

        return row_size

    def _verify_row(self, row: dict):
        if not isinstance(row, dict):
            self._throw("The input row must be a dict object")

        row_size = 0
        for field in self._schema.fields:
            if field.is_primary and field.auto_id:
                if field.name in row:
                    self._throw(
                        f"The primary key field '{field.name}' is auto-id, no need to provide"
                    )
                else:
                    continue

            if field.name not in row:
                self._throw(f"The field '{field.name}' is missed in the row")

            dtype = DataType(field.dtype)
            if dtype in {DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR}:
                if isinstance(row[field.name], np.ndarray):
                    row[field.name] = row[field.name].tolist()

                row_size = row_size + self._verify_vector(row[field.name], field)
            elif dtype == DataType.VARCHAR:
                row_size = row_size + self._verify_varchar(row[field.name], field)
            elif dtype == DataType.JSON:
                row[field.name], size = self._verify_json(row[field.name], field)
                row_size = row_size + size
            elif dtype == DataType.ARRAY:
                if isinstance(row[field.name], np.ndarray):
                    row[field.name] = row[field.name].tolist()

                row_size = row_size + self._verify_array(row[field.name], field)
            else:
                if isinstance(row[field.name], np.generic):
                    row[field.name] = row[field.name].item()

                validator = TYPE_VALIDATOR[dtype.name]
                if not validator(row[field.name]):
                    self._throw(
                        f"Illegal scalar value for field '{field.name}', value overflow or type mismatch"
                    )

                row_size = row_size + TYPE_SIZE[dtype.name]

        with self._buffer_lock:
            self._buffer_size = self._buffer_size + row_size
            self._buffer_row_count = self._buffer_row_count + 1
            self._total_row_count = self._total_row_count + 1
