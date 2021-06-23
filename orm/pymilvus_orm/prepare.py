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

from pymilvus_orm.exceptions import DataNotMatch, DataTypeNotSupport


class Prepare:
    @classmethod
    def prepare_insert_data(cls, data, schema):
        if not isinstance(data, (list, tuple, pandas.DataFrame)):
            raise DataNotMatch(0, "data is not valid")

        fields = schema.fields
        entities = []
        raw_lengths = []
        if isinstance(data, pandas.DataFrame):
            if schema.auto_id:
                if schema.primary_field.name in data:
                    if len(fields) != len(data.columns):
                        raise DataNotMatch(0, f"collection has {len(fields)} fields, and auto_id is True"
                                              f", but got {len(data.columns)} fields")
                    if not data[schema.primary_field.name].isnull().all():
                        raise DataNotMatch(0, "Auto_id is True, primary field should not have data.")
                else:
                    if len(fields) != len(data.columns)+1:
                        raise DataNotMatch(0, f"collection has {len(fields)} fields, and auto_id is True"
                                              f", but got {len(data.columns)} fields")
            else:
                if len(fields) != len(data.columns):
                    raise DataNotMatch(0, f"collection has {len(fields)} fields, and auto_id is False"
                                          f", but got {len(data.columns)} fields")
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
                    raise DataNotMatch(0, f"collection has {len(fields)} fields, "
                                          f"but got {len(data)} fields")

            tmp_fields = copy.deepcopy(fields)
            for i, field in enumerate(tmp_fields):
                if field.is_primary and field.auto_id:
                    tmp_fields.pop(i)

            for i, field in enumerate(tmp_fields):
                if isinstance(data[i], numpy.ndarray):
                    raise DataTypeNotSupport(0, "Data type not support numpy.ndarray")

                entities.append({
                    "name": field.name,
                    "type": field.dtype,
                    "values": data[i]})
                raw_lengths.append(len(data[i]))

        lengths = list(set(raw_lengths))
        if len(lengths) > 1:
            raise DataNotMatch(0, "arrays must all be same length")

        return entities
