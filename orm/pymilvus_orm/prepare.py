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

class Prepare(object):
    @classmethod
    def prepare_insert_data_for_list_or_tuple(cls, data, schema):
        import pandas
        if not isinstance(data, list) and not isinstance(data, tuple) and not isinstance(data, pandas.DataFrame):
            raise Exception("data is not invalid")

        fields = schema.fields
        if isinstance(data, pandas.DataFrame):
            if len(fields) != len(data.columns):
                raise Exception(f"collection has {len(fields)} fields, but go {len(data.columns)} fields")
        elif len(data) != len(fields):
            raise Exception(f"collection has {len(fields)} fields, but go {len(data)} fields")

        if isinstance(data, pandas.DataFrame):
            entities = [{
                "name": field.name,
                "type": field.dtype,
                "values": list(data[field.name]),
            } for i, field in enumerate(fields)]
        else:
            entities = [{
                "name": field.name,
                "type": field.dtype,
                "values": data[i],
            } for i, field in enumerate(fields)]

        ids = None
        for i, field in enumerate(fields):
            if field.is_primary:
                ids = data[i]

        return entities, ids

