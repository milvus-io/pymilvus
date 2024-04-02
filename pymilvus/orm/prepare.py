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

import copy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from pymilvus.client import entity_helper
from pymilvus.exceptions import (
    DataNotMatchException,
    DataTypeNotSupportException,
    ExceptionsMessage,
    UpsertAutoIDTrueException,
)
from pymilvus.client.types import DataType

from .schema import CollectionSchema


class Prepare:
    @classmethod
    def prepare_insert_data(
        cls,
        data: Union[List, Tuple, pd.DataFrame],
        schema: CollectionSchema,
    ) -> List:
        if not isinstance(data, (list, tuple, pd.DataFrame)):
            raise DataTypeNotSupportException(message=ExceptionsMessage.DataTypeNotSupport)

        fields = schema.fields
        entities = []  # Entities

        if isinstance(data, pd.DataFrame):
            if (
                schema.auto_id
                and schema.primary_field.name in data
                and not data[schema.primary_field.name].isnull().all()
            ):
                raise DataNotMatchException(message=ExceptionsMessage.AutoIDWithData)
            # TODO(SPARSE): support pd.SparseDtype for sparse float vector field
            for field in fields:
                if field.is_primary and field.auto_id:
                    continue
                values = []
                if field.name in list(data.columns):
                    values = list(data[field.name])
                entities.append({"name": field.name, "type": field.dtype, "values": values})
            return entities

        tmp_fields = copy.deepcopy(fields)
        for i, field in enumerate(tmp_fields):
            #  TODO Goose: Checking auto_id and is_primary only, maybe different than
            #  schema.is_primary, schema.auto_id, need to check why and how schema is built.
            if field.is_primary and field.auto_id:
                tmp_fields.pop(i)

        for i, field in enumerate(tmp_fields):
            try:
                f_data = data[i]
            # the last missing part of data is also completed in order according to the schema
            except IndexError:
                entities.append({"name": field.name, "type": field.dtype, "values": []})

            if isinstance(f_data, np.ndarray):
                d = f_data.tolist()

            elif (
                isinstance(f_data[0], np.ndarray)
                and field.dtype in (DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR)
            ):
                d = [bytes(ndarr.view(np.uint8).tolist()) for ndarr in f_data]

            else:
                d = f_data if f_data is not None else []

            entities.append({"name": field.name, "type": field.dtype, "values": d})

        return entities

    @classmethod
    def prepare_upsert_data(
        cls,
        data: Union[List, Tuple, pd.DataFrame, entity_helper.SparseMatrixInputType],
        schema: CollectionSchema,
    ) -> List:
        if schema.auto_id:
            raise UpsertAutoIDTrueException(message=ExceptionsMessage.UpsertAutoIDTrue)

        return cls.prepare_insert_data(data, schema)
