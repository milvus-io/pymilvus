from pymilvus_orm.types import DataType
from pymilvus_orm.constants import *
import copy

class CollectionSchema(object):
    def __init__(self, fields, description="", **kwargs):
        self.fields = fields
        self.description = description
        self._kwargs = kwargs

    @classmethod
    def construct_from_dict(cls, raw):
        fields = [FieldSchema.construct_from_dict(field_raw) for field_raw in raw['fields']]
        return CollectionSchema(fields, raw.get('description', ""))

    @property
    def primary_field(self):
        for f in self.fields:
            if f.is_primary:
                return f

    @property
    def auto_id(self):
        return self.primary_field is None

    def to_dict(self):
        _dict = {}
        _dict["auto_id"] = self.primary_field is None
        _dict["description"] = self.description
        _dict["fields"] = [f.to_dict() for f in self.fields]
        return _dict

class FieldSchema(object):
    def __init__(self, name, dtype, description="", **kwargs):
        self.name = name
        self._dtype = dtype
        self.description = description
        self._type_params = None
        self._kwargs = kwargs
        self._is_primary = kwargs.get("is_primary", False)
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

    @property
    def is_primary(self):
        return self._is_primary

    @is_primary.setter
    def is_primary(self, primary):
        self._is_primary = primary