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

from typing import List
import pandas
from pandas.api.types import is_list_like

from .types import DataType, map_numpy_dtype_to_datatype, infer_dtype_bydata
from ..client.configs import DefaultConfigs
from ..exceptions import (
    MilvusException,
    CannotInferSchemaException,
    DataTypeNotSupportException,
    PrimaryKeyException,
    FieldsTypeException,
    AutoIDException,
    ExceptionsMessage
)


class FieldSchema:
    name: str
    auto_id: bool = False
    is_primary: bool = False

    def __init__(self, name, dtype: DataType, description="", **kwargs):
        if not DataType.is_valid(dtype):
            raise DataTypeNotSupportException(0, ExceptionsMessage.FieldDtype) from None

        is_primary = kwargs.get("is_primary", False)
        self._init_is_primary(is_primary, dtype)

        auto_id = kwargs.get("auto_id", False)
        self._init_auto_id(auto_id, is_primary, dtype)

        self.name = name
        self.auto_id = auto_id
        self.is_primary = is_primary

        self._dtype = dtype
        self._description = description
        self._type_params = {}

        self._parse_type_params(**kwargs)

    def set_primary_field(self):
        self._init_is_primary(True, self._dtype)
        self.is_primary = True

    @staticmethod
    def _init_is_primary(is_primary, dtype: DataType):
        if not isinstance(is_primary, bool):
            raise PrimaryKeyException(0, ExceptionsMessage.IsPrimaryType)
        if is_primary and (dtype != DataType.INT64 and dtype != DataType.VARCHAR):
            raise PrimaryKeyException(0, ExceptionsMessage.PrimaryKeyType)

    @staticmethod
    def _init_auto_id(auto_id, is_primary: bool, dtype: DataType):
        if auto_id is True and not is_primary:
            raise PrimaryKeyException(0, ExceptionsMessage.AutoIDOnlyOnPK)

        if is_primary and dtype == DataType.INT64 and not isinstance(auto_id, bool):
            raise AutoIDException(0, ExceptionsMessage.AutoIDType)

    def _parse_type_params(self, **kwargs):
        def _is_valid_number(n) -> bool:
            try:
                int(n)
            except (TypeError, ValueError):
                return False
            return True

        if self._type_params is None:
            self._type_params = {}

        if self._dtype in (DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR):
            dim = kwargs.get("dim", None)
            if dim is not None and not _is_valid_number(dim):
                raise MilvusException(0, "Please provide valid dim for BINARY_VECTOR or FLOAT_VECTOR field")
            self._type_params["dim"] = dim

        if self._dtype == DataType.VARCHAR:
            length = kwargs.get(DefaultConfigs.MaxVarCharLengthKey, None)
            # if length is not None and _is_valid_number(length) and int(length) > DefaultConfigs.MaxVarCharLength: this is handled at the server side
            if length is not None and not _is_valid_number(length):
                raise MilvusException(0, f"Please provide valid {DefaultConfigs.MaxVarCharLengthKey} for VARCHAR field")
            self._type_params[DefaultConfigs.MaxVarCharLengthKey] = length

    def __repr__(self):
        r = ["{\n"]
        s = "    {}: {}\n"
        for k, v in self.to_dict().items():
            r.append(s.format(k, v))
        r.append("  }")
        return "".join(r)

    def __getattr__(self, item):
        if self._type_params and item in self._type_params:
            return self._type_params[item]
        return None

    def __eq__(self, other):
        if not isinstance(other, FieldSchema):
            return False
        return self.to_dict() == other.to_dict()

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return self.construct_from_dict(self.to_dict())

    @classmethod
    def construct_from_dict(cls, raw):
        kwargs = {}
        kwargs.update(raw.get("params", {}))
        kwargs['is_primary'] = raw.get("is_primary", False)

        if "auto_id" in raw:
            kwargs["auto_id"] = raw["auto_id"]
        return FieldSchema(raw['name'], raw['type'], raw['description'], **kwargs)

    def to_dict(self):
        d = {
            "name": self.name,
            "type": self.dtype,
            "params": self.params,
            "description": self.description,
        }
        if self.is_primary:
            d.update({
                "is_primary": self.is_primary,
                "auto_id": self.auto_id,
            })

        return d

    @property
    def description(self) -> str:
        """
        Returns the text description of the FieldSchema.

        :return str:
            FieldSchema description text, returned when the operation is successful.

        :example:
        >>> from pymilvus import FieldSchema, DataType
        >>> field = FieldSchema("int64", DataType.INT64, description="int64", is_primary=False)
        >>> field.description
        'int64'
        """
        return self._description

    @property
    def params(self) -> dict:
        """
        Returns the parameters of the field.

        :return list:
            List of the parameter.

        :example:
        >>> from pymilvus import FieldSchema, DataType
        >>> field = FieldSchema("int64", DataType.INT64, description="int64", is_primary=False)
        >>> field.params
        {}
        >>> fvec_field = FieldSchema("fvec", DataType.FLOAT_VECTOR, description="float vector", is_primary=False, dim=128)
        >>> fvec_field.params
        {'dim': 128}
        """
        return self._type_params

    @property
    def dtype(self) -> DataType:
        return self._dtype


class CollectionSchema:
    def __init__(self, fields: List[FieldSchema], description="", **kwargs):
        if not isinstance(fields, list):
            raise FieldsTypeException(0, ExceptionsMessage.FieldsType)
        self._fields = [field for field in fields]
        self._description = description

        self._primary_field = self._init_primary_field(kwargs.get("primary_field", None), self._fields)
        self._auto_id = self._init_auto_id(kwargs.get("auto_id", False), self._primary_field)

    def __repr__(self):
        _dict = {
            "auto_id": self.auto_id,
            "description": self._description,
            "fields": self._fields,
        }
        r = ["{\n"]
        s = "  {}: {}\n"
        for k, v in _dict.items():
            r.append(s.format(k, v))
        r.append("}")
        return "".join(r)

    def __len__(self):
        return len(self.fields)

    def __eq__(self, other):
        """
        The order of the fields of schema must be consistent.
        """
        return self.to_dict() == other.to_dict()

    @classmethod
    def construct_from_dict(cls, raw):
        fields = [FieldSchema.construct_from_dict(field_raw) for field_raw in raw['fields']]
        return CollectionSchema(fields, raw.get('description', ""))

    @property
    def primary_field(self):
        return self._primary_field

    @property
    def fields(self):
        """ Returns the fields about the CollectionSchema.

        :return list:
            List of FieldSchema, return when operation is successful.

        :example:
            >>> from pymilvus import FieldSchema, CollectionSchema, DataType
            >>> field = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
            >>> schema = CollectionSchema(fields=[field])
            >>> schema.fields
            [<pymilvus.schema.FieldSchema object at 0x7fd3716ffc50>]
        """
        return self._fields

    @property
    def description(self):
        """
        Returns a text description of the CollectionSchema.

        :return str:
            CollectionSchema description text, return when operation is successful.

        :example:
            >>> from pymilvus import FieldSchema, CollectionSchema, DataType
            >>> field = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
            >>> schema = CollectionSchema(fields=[field], description="test get description")
            >>> schema.description
            'test get description'
        """
        return self._description

    @property
    def auto_id(self):
        """
        Whether the primary keys are automatically generated.

        :return bool:
            * True: If the primary keys are automatically generated,
            * False: Otherwise.

        :example:
            >>> from pymilvus import FieldSchema, CollectionSchema, DataType
            >>> field = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
            >>> schema = CollectionSchema(fields=[field])
            >>> schema.auto_id
            False
        """
        return self._auto_id

    def to_dict(self):
        _dict = {
            "auto_id": self.auto_id,
            "description": self._description,
            "fields": [s.to_dict() for s in self._fields],
        }
        return _dict

    def _init_primary_field(self, name: str, fields: List[FieldSchema]) -> FieldSchema:
        if name is None:
            pass
        elif name not in [f.name for f in fields]:
            raise MilvusException(0, f"The primary_field [{name}] is not in fields")
        else:
            for f in fields:
                if f.name == name:
                    f.set_primary_field()
                    break

        primary_fields = [f for f in fields if f.is_primary]
        if len(primary_fields) > 1:
            raise PrimaryKeyException(0, ExceptionsMessage.PrimaryKeyOnlyOne)
        elif len(primary_fields) < 1:
            raise PrimaryKeyException(0, ExceptionsMessage.PrimaryKeyNotExist)
        return primary_fields[0]

    def _init_auto_id(self, auto_id: bool, primary_field: FieldSchema) -> bool:
        if primary_field.auto_id:
            return True

        primary_field.auto_id = auto_id
        return auto_id

    # Now the absense of vector field is checked at the server side.
    def _init_vector_field(self, fields: List[FieldSchema]):
        for f in fields:
            if f.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                return f
        raise MilvusException(0, "please provide at least one field of FLOAT_VECTOR or BINARY_VECTOR")


def parse_fields_from_data(datas):
    if isinstance(datas, pandas.DataFrame):
        return parse_fields_from_dataframe(datas)
    fields = []
    if not isinstance(datas, list):
        raise DataTypeNotSupportException(0, ExceptionsMessage.DataTypeNotSupport)
    for d in datas:
        if not is_list_like(d):
            raise DataTypeNotSupportException(0, ExceptionsMessage.DataTypeNotSupport)
        d_type = infer_dtype_bydata(d[0])
        fields.append(FieldSchema("", d_type))
    return fields


def parse_fields_from_dataframe(dataframe: pandas.DataFrame) -> List[FieldSchema]:
    if not isinstance(dataframe, pandas.DataFrame):
        return None
    d_types = list(dataframe.dtypes)
    data_types = list(map(map_numpy_dtype_to_datatype, d_types))
    col_names = list(dataframe.columns)

    column_params_map = {}

    if DataType.UNKNOWN in data_types:
        if len(dataframe) == 0:
            raise CannotInferSchemaException(0, ExceptionsMessage.DataFrameInvalid)
        values = dataframe.head(1).values[0]
        for i, dtype in enumerate(data_types):
            if dtype == DataType.UNKNOWN:
                new_dtype = infer_dtype_bydata(values[i])
                if new_dtype in (DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR):
                    vector_type_params = {}
                    if new_dtype == DataType.BINARY_VECTOR:
                        vector_type_params['dim'] = len(values[i]) * 8
                    else:
                        vector_type_params['dim'] = len(values[i])
                    column_params_map[col_names[i]] = vector_type_params
                # TODO: currently we cannot get max_len_per_row from pandas.DataFrame
                # if new_dtype in (DataType.VARCHAR,):
                #     str_type_params = {}
                #     str_type_params[DefaultConfigs.MaxVarCharLengthKey] = DefaultConfigs.MaxVarCharLength
                data_types[i] = new_dtype

    if DataType.UNKNOWN in data_types:
        raise CannotInferSchemaException(0, ExceptionsMessage.DataFrameInvalid)

    fields = []
    for name, dtype in zip(col_names, data_types):
        type_params = column_params_map.get(name, {})
        field_schema = FieldSchema(name, dtype, **type_params)
        fields.append(field_schema)

    return fields
