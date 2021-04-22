import logging
import pytest
from utils import *
from pymilvus_orm.schema import CollectionSchema, FieldSchema

LOGGER = logging.getLogger(__name__)


class TestCollectionSchema:
    @pytest.fixture(
        scope="function",
    )
    def raw_dict(self):
        _dict = {}
        _dict["description"] = "TestCollectionSchema_description"
        fields = [
            {
                "name": "vec1",
                "description": "desc1",
                "type": DataType.FLOAT_VECTOR,
                "params": {"ndim": 128},
            },

            {
                "name": "vec2",
                "description": "desc2",
                "type": DataType.BINARY_VECTOR,
                "params": {"ndim": 128},
            },
            {
                "name": "ID",
                "description": "ID",
                "type": DataType.INT64,
                "is_primary": True,
            },
        ]
        _dict["fields"] = fields

        return _dict

    def test_constructor_from_dict(self, raw_dict):
        schema = CollectionSchema.construct_from_dict(raw_dict)
        assert schema.description, raw_dict['description']
        assert len(schema.fields) == len(raw_dict['fields'])
        f = schema.primary_field
        assert isinstance(f, FieldSchema)
        assert f.name == raw_dict['fields'][2]['name']


class TestFieldSchema:
    @pytest.fixture(
        scope="function",
    )
    def raw_dict_float_vector(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_floatvector"
        _dict["description"] = "TestFieldSchema_description_floatvector"
        _dict["type"] = DataType.FLOAT_VECTOR
        _dict["params"] = {"ndim": 128}
        return _dict

    @pytest.fixture(
        scope="function",
    )
    def raw_dict_binary_vector(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_binary_vector"
        _dict["description"] = "TestFieldSchema_description_binary_vector"
        _dict["type"] = DataType.BINARY_VECTOR
        _dict["params"] = {"ndim": 128}
        return _dict

    @pytest.fixture(
        scope="function",
    )
    def raw_dict_norm(self):
        _dict = dict()
        _dict["name"] = "TestFieldSchema_name_norm"
        _dict["description"] = "TestFieldSchema_description_norm"
        _dict["type"] = DataType.INT64
        return _dict

    def test_constructor_from_float_dict(self, raw_dict_float_vector):
        field = FieldSchema.construct_from_dict(raw_dict_float_vector)
        assert field.dtype == DataType.FLOAT_VECTOR
        assert field.description == raw_dict_float_vector['description']
        assert field.is_primary == False
        assert field.name == raw_dict_float_vector['name']
        assert field.ndim == raw_dict_float_vector['params']['ndim']

    def test_constructor_from_binary_dict(self, raw_dict_binary_vector):
        field = FieldSchema.construct_from_dict(raw_dict_binary_vector)
        assert field.dtype == DataType.BINARY_VECTOR
        assert field.description == raw_dict_binary_vector['description']
        assert field.is_primary == False
        assert field.name == raw_dict_binary_vector['name']
        assert field.ndim == raw_dict_binary_vector['params']['ndim']

    def test_constructor_from_norm_dict(self, raw_dict_norm):
        field = FieldSchema.construct_from_dict(raw_dict_norm)
        assert field.dtype == DataType.INT64
        assert field.description == raw_dict_norm['description']
        assert field.is_primary == False
        assert field.name == raw_dict_norm['name']
