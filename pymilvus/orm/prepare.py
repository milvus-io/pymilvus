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

import numpy
import pandas

from ..exceptions import (
    DataNotMatchException,
    DataTypeNotSupportException,
    ExceptionsMessage,
    UpsertAutoIDTrueException,
)


class Prepare:
    @classmethod
    def prepare_insert_or_upsert_data(cls, data, schema, is_insert=True):
        if not isinstance(data, (list, tuple, pandas.DataFrame)):
            raise DataTypeNotSupportException(message=ExceptionsMessage.DataTypeNotSupport)

        fields = schema.fields
        entities = []  # Entities

        if isinstance(data, pandas.DataFrame):
            if schema.auto_id:
                if is_insert is False:
                    raise UpsertAutoIDTrueException(message=ExceptionsMessage.UpsertAutoIDTrue)
                if schema.primary_field.name in data:
                    if not data[schema.primary_field.name].isnull().all():
                        raise DataNotMatchException(message=ExceptionsMessage.AutoIDWithData)
            for i, field in enumerate(fields):
                if field.is_primary and field.auto_id:
                    continue
                values = []
                if field.name in list(data.columns):
                    values = list(data[field.name])
                entities.append({"name": field.name,
                                 "type": field.dtype,
                                 "values": values})
        else:
            if schema.auto_id:
                if is_insert is False:
                    raise UpsertAutoIDTrueException(message=ExceptionsMessage.UpsertAutoIDTrue)

            tmp_fields = copy.deepcopy(fields)
            for i, field in enumerate(tmp_fields):
                #  TODO Goose: Checking auto_id and is_primary only, maybe different than
                #  schema.is_primary, schema.auto_id, need to check why and how schema is built.
                if field.is_primary and field.auto_id:
                    tmp_fields.pop(i)

            for i, field in enumerate(tmp_fields):
                # TODO: check string.
                if isinstance(data[i], numpy.ndarray):
                    data[i] = data[i].tolist()

                entities.append({
                    "name": field.name,
                    "type": field.dtype,
                    "values": data[i]})

        return entities
