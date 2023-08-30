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

import logging
from threading import Lock

from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import CollectionSchema

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
        segment_size: int,
        file_type: BulkFileType = BulkFileType.NPY,
    ):
        self._schema = schema
        self._buffer_size = 0
        self._buffer_row_count = 0
        self._segment_size = segment_size
        self._file_type = file_type
        self._buffer_lock = Lock()

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
    def segment_size(self):
        return self._segment_size

    def _new_buffer(self):
        old_buffer = self._buffer
        with self._buffer_lock:
            self._buffer = Buffer(self._schema, self._file_type)
        return old_buffer

    def append(self, row: dict, **kwargs):
        self._verify_row(row)
        with self._buffer_lock:
            self._buffer.append_row(row)

    def commit(self, **kwargs):
        self._buffer_size = 0
        self._buffer_row_count = 0

    @property
    def data_path(self):
        return ""

    def _throw(self, msg: str):
        logger.error(msg)
        raise MilvusException(message=msg)

    def _verify_row(self, row: dict):
        if not isinstance(row, dict):
            self._throw("The input row must be a dict object")

        row_size = 0
        for field in self._schema.fields:
            if field.name not in row:
                self._throw(f"The field '{field.name}' is missed in the row")

            if field.is_parimary and field.auto_id:
                self._throw(f"The primary key field '{field.name}' is auto-id, no need to provide")

            dtype = DataType(field.dtype)
            validator = TYPE_VALIDATOR[dtype.name]
            if dtype in {DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR}:
                dim = field.params["dim"]
                if not validator(row[field.name], dim):
                    self._throw(f"Illegal vector data for vector field '{dtype.name}'")

                vec_size = (
                    len(row[field.name]) * 4
                    if dtype == DataType.FLOAT_VECTOR
                    else len(row[field.name]) / 8
                )
                row_size = row_size + vec_size
            elif dtype == DataType.VARCHAR:
                max_len = field.params["max_length"]
                if not validator(row[field.name], max_len):
                    self._throw(f"Illegal varchar value for field '{dtype.name}'")

                row_size = row_size + len(row[field.name])
            else:
                if not validator(row[field.name]):
                    self._throw(f"Illegal scalar value for field '{dtype.name}'")

                row_size = row_size + TYPE_SIZE[dtype.name]

        self._buffer_size = self._buffer_size + row_size
        self._buffer_row_count = self._buffer_row_count + 1
