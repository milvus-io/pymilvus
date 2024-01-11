import numpy
import pytest

from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusException
from utils import *
from pymilvus.orm import schema as s


class TestCollectionSchema:
    @pytest.fixture(scope="function")
    def raw_dict(self):
        _dict = {}
        _dict["description"] = "TestCollectionSchema_description"
        fields = [
            {
                "name": "vec1",
                "description": "desc1",
                "type": DataType.FLOAT_VECTOR,
                "params": {"dim": 128},
            },

            {
                "name": "vec2",
                "description": "desc2",
                "type": DataType.BINARY_VECTOR,
                "params": {"dim": 128},
            },
            {
                "name": "ID",
                "description": "ID",
                "type": DataType.INT64,
                "is_primary": True,
                "auto_id": False
            },
        ]
        _dict["fields"] = fields
        _dict["enable_dynamic_field"] = True

        return _dict

    def test_constructor_from_dict(self, raw_dict):
        schema = CollectionSchema.construct_from_dict(raw_dict)
        assert schema.enable_dynamic_field == raw_dict.get("enable_dynamic_field", False)
        assert schema.description, raw_dict['description']
        assert len(schema.fields) == len(raw_dict['fields'])
        f = schema.primary_field
        assert isinstance(f, FieldSchema)
        assert f.name == raw_dict['fields'][2]['name']

    def test_to_dict(self, raw_dict):
        schema = CollectionSchema.construct_from_dict(raw_dict)
        target = schema.to_dict()
        target.pop("auto_id", None)
        assert target == raw_dict
        assert target is not raw_dict


class TestFieldSchema:
    @pytest.fixture(scope="function")
    def raw_dict_float_vector(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_floatvector"
        _dict["description"] = "TestFieldSchema_description_floatvector"
        _dict["type"] = DataType.FLOAT_VECTOR
        _dict["params"] = {"dim": 128}
        return _dict

    @pytest.fixture(scope="function")
    def raw_dict_binary_vector(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_binary_vector"
        _dict["description"] = "TestFieldSchema_description_binary_vector"
        _dict["type"] = DataType.BINARY_VECTOR
        _dict["params"] = {"dim": 128}
        return _dict

    @pytest.fixture(scope="function")
    def raw_dict_float16_vector(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_float16_vector"
        _dict["description"] = "TestFieldSchema_description_float16_vector"
        _dict["type"] = DataType.FLOAT16_VECTOR
        _dict["params"] = {"dim": 128}
        return _dict
    
    @pytest.fixture(scope="function")
    def raw_dict_bfloat16_vector(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_bfloat16_vector"
        _dict["description"] = "TestFieldSchema_description_bfloat16_vector"
        _dict["type"] = DataType.BFLOAT16_VECTOR
        _dict["params"] = {"dim": 128}
        return _dict

    @pytest.fixture(scope="function")
    def raw_dict_norm(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_norm"
        _dict["description"] = "TestFieldSchema_description_norm"
        _dict["type"] = DataType.INT64
        return _dict

    @pytest.fixture(scope="function")
    def dataframe1(self):
        import pandas
        data = {
            'float': [1.0],
            'int32': [2],
            'float_vec': [numpy.array([3, 4.0], numpy.float32)]
        }
        df1 = pandas.DataFrame(data)
        return df1

    def test_constructor_from_float_dict(self, raw_dict_float_vector):
        field = FieldSchema.construct_from_dict(raw_dict_float_vector)
        assert field.dtype == DataType.FLOAT_VECTOR
        assert field.description == raw_dict_float_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_float_vector['name']
        assert field.dim == raw_dict_float_vector['params']['dim']

    def test_constructor_from_binary_dict(self, raw_dict_binary_vector):
        field = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        assert field.dtype == DataType.BINARY_VECTOR
        assert field.description == raw_dict_binary_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_binary_vector['name']
        assert field.dim == raw_dict_binary_vector['params']['dim']

    def test_constructor_from_float16_dict(self, raw_dict_float16_vector):
        field = FieldSchema.construct_from_dict(raw_dict_float16_vector)
        assert field.dtype == DataType.FLOAT16_VECTOR
        assert field.description == raw_dict_float16_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_float16_vector['name']
        assert field.dim == raw_dict_float16_vector['params']['dim']

    def test_constructor_from_bfloat16_dict(self, raw_dict_bfloat16_vector):
        field = FieldSchema.construct_from_dict(raw_dict_bfloat16_vector)
        assert field.dtype == DataType.BFLOAT16_VECTOR
        assert field.description == raw_dict_bfloat16_vector['description']
        assert field.is_primary is False
        assert field.name == raw_dict_bfloat16_vector['name']
        assert field.dim == raw_dict_bfloat16_vector['params']['dim']

    def test_constructor_from_norm_dict(self, raw_dict_norm):
        field = FieldSchema.construct_from_dict(raw_dict_norm)
        assert field.dtype == DataType.INT64
        assert field.description == raw_dict_norm['description']
        assert field.is_primary is False
        assert field.name == raw_dict_norm['name']
        assert field.dim is None
        assert field.dummy is None

    def test_cmp(self, raw_dict_binary_vector):
        import copy
        field1 = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        field2 = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        assert field1 == field2
        dict1 = copy.deepcopy(raw_dict_binary_vector)
        dict1["name"] = dict1["name"] + "_"
        field3 = FieldSchema.construct_from_dict(dict1)
        assert field1 != field3

    def test_to_dict(self, raw_dict_norm, raw_dict_float_vector, raw_dict_binary_vector):
        fields = []
        dicts = [raw_dict_norm, raw_dict_float_vector, raw_dict_binary_vector]
        fields.append(FieldSchema.construct_from_dict(raw_dict_norm))
        fields.append(FieldSchema.construct_from_dict(raw_dict_float_vector))
        fields.append(FieldSchema.construct_from_dict(raw_dict_binary_vector))

        for i, f in enumerate(fields):
            target = f.to_dict()
            assert target == dicts[i]
            assert target is not dicts[i]
