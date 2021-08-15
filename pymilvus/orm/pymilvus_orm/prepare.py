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

from pymilvus_orm.exceptions import DataNotMatchException, DataTypeNotSupportException, ExceptionsMessage


class Prepare:
    @classmethod
    def prepare_insert_data(cls, data, schema):
        if not isinstance(data, (list, tuple, pandas.DataFrame)):
            raise DataTypeNotSupportException(0, ExceptionsMessage.DataTypeNotSupport)

        fields = schema.fields
        entities = []
        raw_lengths = []
        if isinstance(data, pandas.DataFrame):
            if schema.auto_id:
                if schema.primary_field.name in data:
                    if len(fields) != len(data.columns):
                        raise DataNotMatchException(0, ExceptionsMessage.FieldsNumInconsistent)
                    if not data[schema.primary_field.name].isnull().all():
                        raise DataNotMatchException(0, ExceptionsMessage.AutoIDWithData)
                else:
                    if len(fields) != len(data.columns)+1:
                        raise DataNotMatchException(0, ExceptionsMessage.FieldsNumInconsistent)
            else:
                if len(fields) != len(data.columns):
                    raise DataNotMatchException(0, ExceptionsMessage.FieldsNumInconsistent)
            for i, field in enumerate(fields):
                if field.is_primary and field.auto_id:
                    continue
                entities.append({"name": field.name,
                                 "type": field.dtype,
                                 "values": list(data[field.name])})
                raw_lengths.append(len(data[field.name]))
        else:
            if schema.auto_id:
                if len(data) + 1 != len(fields):
                    raise DataNotMatchException(0, ExceptionsMessage.FieldsNumInconsistent)

            tmp_fields = copy.deepcopy(fields)
            for i, field in enumerate(tmp_fields):
                if field.is_primary and field.auto_id:
                    tmp_fields.pop(i)

            for i, field in enumerate(tmp_fields):
                if isinstance(data[i], numpy.ndarray):
                    raise DataTypeNotSupportException(0, ExceptionsMessage.NdArrayNotSupport)

                entities.append({
                    "name": field.name,
                    "type": field.dtype,
                    "values": data[i]})
                raw_lengths.append(len(data[i]))

        lengths = list(set(raw_lengths))
        if len(lengths) > 1:
            raise DataNotMatchException(0, ExceptionsMessage.DataLengthsInconsistent)

        return entities
