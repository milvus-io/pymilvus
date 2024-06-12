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
from pathlib import Path

import numpy as np
import pandas as pd

from pymilvus.client.types import (
    DataType,
)
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import (
    CollectionSchema,
    FieldSchema,
)

from .constants import (
    DYNAMIC_FIELD_NAME,
    MB,
    NUMPY_TYPE_CREATOR,
    BulkFileType,
)

logger = logging.getLogger("bulk_buffer")
logger.setLevel(logging.DEBUG)


class Buffer:
    def __init__(
        self,
        schema: CollectionSchema,
        file_type: BulkFileType = BulkFileType.NUMPY,
    ):
        self._buffer = {}
        self._fields = {}
        self._file_type = file_type
        for field in schema.fields:
            if field.is_primary and field.auto_id:
                continue
            self._buffer[field.name] = []
            self._fields[field.name] = field

        if len(self._buffer) == 0:
            self._throw("Illegal collection schema: fields list is empty")

        # dynamic field, internal name is '$meta'
        if schema.enable_dynamic_field:
            self._buffer[DYNAMIC_FIELD_NAME] = []
            self._fields[DYNAMIC_FIELD_NAME] = FieldSchema(
                name=DYNAMIC_FIELD_NAME, dtype=DataType.JSON
            )

    @property
    def row_count(self) -> int:
        if len(self._buffer) == 0:
            return 0

        for k in self._buffer:
            return len(self._buffer[k])
        return None

    def _throw(self, msg: str):
        logger.error(msg)
        raise MilvusException(message=msg)

    def _raw_obj(self, x: object):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()

        return x

    def append_row(self, row: dict):
        dynamic_values = {}
        if DYNAMIC_FIELD_NAME in row and not isinstance(row[DYNAMIC_FIELD_NAME], dict):
            self._throw(f"Dynamic field '{DYNAMIC_FIELD_NAME}' value should be JSON format")

        for k in row:
            if k == DYNAMIC_FIELD_NAME:
                dynamic_values.update(row[k])
                continue

            if k not in self._buffer:
                dynamic_values[k] = self._raw_obj(row[k])
            else:
                self._buffer[k].append(row[k])

        if DYNAMIC_FIELD_NAME in self._buffer:
            self._buffer[DYNAMIC_FIELD_NAME].append(dynamic_values)

    def persist(self, local_path: str, **kwargs) -> list:
        # verify row count of fields are equal
        row_count = -1
        for k in self._buffer:
            if row_count < 0:
                row_count = len(self._buffer[k])
            elif row_count != len(self._buffer[k]):
                buffer_k_len = len(self._buffer[k])
                self._throw(
                    f"Column {k} row count {buffer_k_len} doesn't equal to the first column row count {row_count}"
                )

        # output files
        if self._file_type == BulkFileType.NUMPY:
            return self._persist_npy(local_path, **kwargs)
        if self._file_type == BulkFileType.JSON:
            return self._persist_json_rows(local_path, **kwargs)
        if self._file_type == BulkFileType.PARQUET:
            return self._persist_parquet(local_path, **kwargs)

        self._throw(f"Unsupported file tpye: {self._file_type}")
        return []

    def _persist_npy(self, local_path: str, **kwargs):
        file_list = []
        row_count = len(next(iter(self._buffer.values())))
        for k in self._buffer:
            full_file_name = Path(local_path).joinpath(k + ".npy")
            file_list.append(str(full_file_name))
            try:
                Path(local_path).mkdir(exist_ok=True)

                # numpy data type specify
                dt = None
                field_schema = self._fields[k]
                if field_schema.dtype == DataType.ARRAY:
                    element_type = field_schema.element_type
                    dt = NUMPY_TYPE_CREATOR[element_type.name]
                elif field_schema.dtype.name in NUMPY_TYPE_CREATOR:
                    dt = NUMPY_TYPE_CREATOR[field_schema.dtype.name]

                # for JSON field, convert to string array
                if field_schema.dtype == DataType.JSON:
                    str_arr = []
                    for val in self._buffer[k]:
                        str_arr.append(json.dumps(val))
                    self._buffer[k] = str_arr

                # currently, milvus server doesn't support numpy for sparse vector
                if field_schema.dtype == DataType.SPARSE_FLOAT_VECTOR:
                    self._throw(
                        f"Failed to persist file {full_file_name},"
                        f" error: milvus doesn't support parsing sparse vectors from numpy file"
                    )

                # special process for float16 vector, the self._buffer stores bytes for
                # float16 vector, convert the bytes to uint8 array
                if field_schema.dtype in {DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR}:
                    a = []
                    for b in self._buffer[k]:
                        a.append(np.frombuffer(b, dtype=dt).tolist())
                    arr = np.array(a, dtype=dt)
                else:
                    arr = np.array(self._buffer[k], dtype=dt)
                np.save(str(full_file_name), arr)
            except Exception as e:
                self._throw(f"Failed to persist file {full_file_name}, error: {e}")

            logger.info(f"Successfully persist file {full_file_name}, row count: {row_count}")

        if len(file_list) != len(self._buffer):
            logger.error("Some of fields were not persisted successfully, abort the files")
            for f in file_list:
                Path(f).unlink()
            Path(local_path).rmdir()
            file_list.clear()
            self._throw("Some of fields were not persisted successfully, abort the files")

        return file_list

    def _persist_json_rows(self, local_path: str, **kwargs):
        rows = []
        row_count = len(next(iter(self._buffer.values())))
        row_index = 0
        while row_index < row_count:
            row = {}
            for k, v in self._buffer.items():
                # special process for float16 vector, the self._buffer stores bytes for
                # float16 vector, convert the bytes to float list
                field_schema = self._fields[k]
                if field_schema.dtype in {DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR}:
                    dt = (
                        np.dtype("bfloat16")
                        if (field_schema.dtype == DataType.BFLOAT16_VECTOR)
                        else np.float16
                    )
                    row[k] = np.frombuffer(v[row_index], dtype=dt).tolist()
                else:
                    row[k] = v[row_index]
            rows.append(row)
            row_index = row_index + 1

        data = {
            "rows": rows,
        }
        file_path = Path(local_path + ".json")
        try:
            with file_path.open("w") as json_file:
                json.dump(data, json_file, indent=2)
        except Exception as e:
            self._throw(f"Failed to persist file {file_path}, error: {e}")

        logger.info(f"Successfully persist file {file_path}, row count: {len(rows)}")
        return [str(file_path)]

    def _persist_parquet(self, local_path: str, **kwargs):
        file_path = Path(local_path + ".parquet")

        data = {}
        for k in self._buffer:
            field_schema = self._fields[k]
            if field_schema.dtype in {DataType.JSON, DataType.SPARSE_FLOAT_VECTOR}:
                # for JSON and SPARSE_VECTOR field, store as string array
                str_arr = []
                for val in self._buffer[k]:
                    str_arr.append(json.dumps(val))
                data[k] = pd.Series(str_arr, dtype=None)
            elif field_schema.dtype in {DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR}:
                arr = []
                for val in self._buffer[k]:
                    arr.append(np.array(val, dtype=NUMPY_TYPE_CREATOR[field_schema.dtype.name]))
                data[k] = pd.Series(arr)
            elif field_schema.dtype in {DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR}:
                # special process for float16 vector, the self._buffer stores bytes for
                # float16 vector, convert the bytes to uint8 array
                arr = []
                for val in self._buffer[k]:
                    arr.append(
                        np.frombuffer(val, dtype=NUMPY_TYPE_CREATOR[field_schema.dtype.name])
                    )
                data[k] = pd.Series(arr)
            elif field_schema.dtype == DataType.ARRAY:
                dt = NUMPY_TYPE_CREATOR[field_schema.element_type.name]
                arr = []
                for val in self._buffer[k]:
                    arr.append(np.array(val, dtype=dt))
                data[k] = pd.Series(arr)
            elif field_schema.dtype.name in NUMPY_TYPE_CREATOR:
                dt = NUMPY_TYPE_CREATOR[field_schema.dtype.name]
                data[k] = pd.Series(self._buffer[k], dtype=dt)
            else:
                # dtype is null, let pandas deduce the type, might not work
                data[k] = pd.Series(self._buffer[k])

        # calculate a proper row group size
        row_group_size_min = 1000
        row_group_size = 10000
        row_group_size_max = 1000000
        if "buffer_size" in kwargs and "buffer_row_count" in kwargs:
            row_group_bytes = kwargs.get(
                "row_group_bytes", 32 * MB
            )  # 32MB is an experience value that avoid high memory usage of parquet reader on server-side
            buffer_size = kwargs.get("buffer_size", 1)
            buffer_row_count = kwargs.get("buffer_row_count", 1)
            size_per_row = int(buffer_size / buffer_row_count) + 1
            row_group_size = int(row_group_bytes / size_per_row)
            if row_group_size < row_group_size_min:
                row_group_size = row_group_size_min
            if row_group_size > row_group_size_max:
                row_group_size = row_group_size_max

        # write to Parquet file
        data_frame = pd.DataFrame(data=data)
        data_frame.to_parquet(
            file_path, row_group_size=row_group_size, engine="pyarrow"
        )  # don't use fastparquet

        logger.info(
            f"Successfully persist file {file_path}, total size: {buffer_size},"
            f" row count: {buffer_row_count}, row group size: {row_group_size}"
        )
        return [str(file_path)]
