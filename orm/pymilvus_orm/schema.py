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

from pymilvus_orm.types import DataType
from pymilvus_orm.constants import *
import copy
import json


class CollectionSchema(object):
    def __init__(self, fields, description="", **kwargs):
        self.fields = fields
        self.description = description
        self._kwargs = kwargs

    def __repr__(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return str(json.dumps(self.to_dict()))

    @classmethod
    def construct_from_dict(cls, raw):
        fields = [FieldSchema.construct_from_dict(field_raw) for field_raw in raw['fields']]
        return CollectionSchema(fields, raw.get('description', ""))

    @property
    def primary_field(self):
        primary_field = self._kwargs.get("primary_field", None)
        for f in self.fields:
            if f.is_primary or f.name == primary_field:
                f.is_primary = True
                return f

    @property
    def auto_id(self):
        return self.primary_field is None

    def to_dict(self):
        _dict = {
            "auto_id": self.primary_field is None,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields]
        }
        return _dict


class FieldSchema(object):
    def __init__(self, name, dtype, description="", **kwargs):
        self.name = name
        self._dtype = dtype
        self.description = description
        self._type_params = None
        self._kwargs = kwargs
        self.is_primary = kwargs.get("is_primary", False)
        self._parse_type_params()

    def _parse_type_params(self):
        # update self._type_params according to self._kwargs
        if self._dtype not in (DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR):
            return
        if not self._kwargs:
            return
        # currently only support ndim
        if self._kwargs:
            for k in VECTOR_COMMON_TYPE_PARAMS:
                if k in self._kwargs:
                    if self._type_params is None:
                        self._type_params = {}
                    self._type_params[k] = self._kwargs[k]

    @classmethod
    def construct_from_dict(cls, raw):
        kwargs = {}
        kwargs.update(raw.get("params", {}))
        kwargs['is_primary'] = raw.get("is_primary", False)
        return FieldSchema(raw['name'], raw['type'], raw['description'], **kwargs)

    def to_dict(self):
        _dict = dict()
        _dict["name"] = self.name
        _dict["description"] = self.description
        _dict["type"] = self.dtype
        if self._type_params:
            _dict["params"] = copy.deepcopy(self.params)
        if self.is_primary:
            _dict["is_primary"] = True
        return _dict

    def __getattr__(self, item):
        if self._type_params and item in self._type_params:
            return self._type_params[item]

    @property
    def params(self):
        return self._type_params

    @property
    def dtype(self):
        return self._dtype
