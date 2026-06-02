from unittest.mock import patch

import numpy as np
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.entity_helper import convert_to_array, extract_array_row_data_no_validity
from pymilvus.client.search_result import extract_array_row_data as extract_search_array_row_data
from pymilvus.client.type_info import get_array_element_attr
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.grpc_gen import schema_pb2 as schema_types


class TestLogicalTypeArrayHelpers:
    @pytest.mark.parametrize(
        "element_type,values",
        [
            (DataType.BOOL, [True, False]),
            (DataType.INT8, [1, 2]),
            (DataType.INT16, [100, 200]),
            (DataType.INT32, [1000, 2000]),
            (DataType.INT64, [10000, 20000]),
            (DataType.FLOAT, [1.5, 2.5]),
            (DataType.DOUBLE, [1.25, 2.75]),
            (DataType.STRING, ["a", "b"]),
            (DataType.VARCHAR, ["varchar-a", "varchar-b"]),
        ],
    )
    def test_convert_to_array_uses_registry_backed_element_attrs(self, element_type, values):
        field_info = {"name": "array_field", "element_type": element_type}

        with patch(
            "pymilvus.client.entity_helper.type_info.get_array_element_attr",
            wraps=entity_helper.type_info.get_array_element_attr,
        ) as get_array_element_attr_spy:
            array_data = convert_to_array(values, field_info)

        get_array_element_attr_spy.assert_any_call(element_type)
        attr = get_array_element_attr(element_type)
        assert attr is not None
        data = list(getattr(array_data, attr).data)
        if attr in {"float_data", "double_data"}:
            assert data == pytest.approx(values)
        else:
            assert data == values

    def test_array_element_attr_registry(self):
        assert get_array_element_attr(DataType.INT64) == "long_data"
        assert get_array_element_attr(DataType.VARCHAR) == "string_data"
        assert get_array_element_attr(DataType.TIMESTAMPTZ) is None

    def test_extract_array_row_data_unsupported_element_returns_none(self):
        field_data = schema_types.FieldData(type=DataType.ARRAY, field_name="arr")
        field_data.scalars.array_data.element_type = 999
        field_data.scalars.array_data.data.add()

        assert entity_helper.extract_array_row_data(field_data, 0) is None

    def test_extract_array_rows_unsupported_element_raises_milvus_exception(self):
        field_data = schema_types.FieldData(type=DataType.ARRAY, field_name="arr")
        field_data.scalars.array_data.element_type = 999
        field_data.scalars.array_data.data.add()

        with pytest.raises(MilvusException, match="Unsupported data type: 999"):
            extract_array_row_data_no_validity(field_data, [{}], 1)

    def test_entity_helper_rejects_timestamptz_array_element_type(self):
        field_data = schema_types.FieldData(type=DataType.ARRAY, field_name="ts_arr")
        field_data.scalars.array_data.element_type = DataType.TIMESTAMPTZ
        field_data.scalars.array_data.data.append(
            schema_types.ScalarField(
                string_data=schema_types.StringArray(data=["2026-05-27T00:00:00Z"])
            )
        )

        with pytest.raises(MilvusException, match="Unsupported data type"):
            extract_array_row_data_no_validity(field_data, [{}], 1)

    @pytest.mark.parametrize(
        "element_type",
        [
            DataType.BINARY_VECTOR,
            DataType.FLOAT16_VECTOR,
            DataType.BFLOAT16_VECTOR,
            DataType.INT8_VECTOR,
        ],
    )
    def test_scalar_array_rejects_vector_element_types(self, element_type):
        field_info = {"name": "array_field", "element_type": element_type}

        with pytest.raises(ParamError, match="Unsupported element type"):
            convert_to_array([b"\x00"], field_info)

    def test_search_result_array_extraction_uses_type_info(self):
        scalar = schema_types.ScalarField()
        scalar.string_data.data.extend(["a", "b"])

        assert extract_search_array_row_data([scalar], DataType.VARCHAR) == [["a", "b"]]

    def test_search_result_rejects_timestamptz_array_element_type(self):
        scalar = schema_types.ScalarField()
        scalar.string_data.data.extend(["2026-05-27T00:00:00Z"])

        with pytest.raises(MilvusException, match="Unsupported data type"):
            extract_search_array_row_data([scalar], DataType.TIMESTAMPTZ)

    @pytest.mark.parametrize(
        "element_type,values,attr_name,expected",
        [
            (
                DataType.BINARY_VECTOR,
                [b"\x0f", b"\xf0"],
                "binary_vector",
                b"\x0f\xf0",
            ),
            (
                DataType.FLOAT16_VECTOR,
                [
                    np.array([1.0, 2.0], dtype=np.float16),
                    np.array([3.0, 4.0], dtype=np.float16),
                ],
                "float16_vector",
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).tobytes(),
            ),
            (
                DataType.INT8_VECTOR,
                [
                    np.array([1, 2, 3, 4], dtype=np.int8),
                    np.array([5, 6, 7, 8], dtype=np.int8),
                ],
                "int8_vector",
                np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8).tobytes(),
            ),
        ],
    )
    def test_registry_backed_array_of_vector_destination(
        self, element_type, values, attr_name, expected
    ):
        field_info = {
            "name": "array_vec",
            "element_type": element_type,
            "params": {"dim": 4},
        }

        with patch(
            "pymilvus.client.entity_helper.type_info.get_vector_attr",
            wraps=entity_helper.type_info.get_vector_attr,
        ) as get_vector_attr:
            vector_data = entity_helper.convert_to_array_of_vector(values, field_info)

        get_vector_attr.assert_any_call(element_type)
        assert vector_data.dim == 4
        assert getattr(vector_data, attr_name) == expected

    def test_registry_backed_array_of_bfloat16_vector_destination(self):
        ml_dtypes = pytest.importorskip("ml_dtypes")
        values = [
            np.array([1.0, 2.0], dtype=ml_dtypes.bfloat16),
            np.array([3.0, 4.0], dtype=ml_dtypes.bfloat16),
        ]
        field_info = {
            "name": "array_vec",
            "element_type": DataType.BFLOAT16_VECTOR,
            "params": {"dim": 4},
        }

        vector_data = entity_helper.convert_to_array_of_vector(values, field_info)

        assert vector_data.dim == 4
        assert vector_data.bfloat16_vector == b"".join(
            value.view(np.uint8).tobytes() for value in values
        )

    def test_array_of_vector_accepts_raw_bytes_for_byte_vector_destination(self):
        field_info = {
            "name": "array_vec",
            "element_type": DataType.FLOAT16_VECTOR,
            "params": {"dim": 2},
        }

        vector_data = entity_helper.convert_to_array_of_vector([b"\x00<\x00@"], field_info)

        assert vector_data.dim == 2
        assert vector_data.float16_vector == b"\x00<\x00@"

    @pytest.mark.parametrize(
        "element_type,value,match",
        [
            (
                DataType.FLOAT16_VECTOR,
                np.array([1.0, 2.0], dtype=np.float32),
                "dtype=float16",
            ),
            (
                DataType.FLOAT16_VECTOR,
                [1, 2],
                "np.ndarray\\(dtype=float16\\)",
            ),
            (
                DataType.BFLOAT16_VECTOR,
                np.array([1.0, 2.0], dtype=np.float32),
                "dtype=bfloat16",
            ),
            (
                DataType.BFLOAT16_VECTOR,
                [1, 2],
                "np.ndarray\\(dtype=bfloat16\\)",
            ),
            (
                DataType.INT8_VECTOR,
                np.array([1, 2], dtype=np.int16),
                "dtype=int8",
            ),
            (
                DataType.INT8_VECTOR,
                [1, 2],
                "Expected bytes or np.ndarray",
            ),
        ],
    )
    def test_array_of_vector_rejects_invalid_byte_vector_values(self, element_type, value, match):
        field_info = {
            "name": "array_vec",
            "element_type": element_type,
            "params": {"dim": 2},
        }

        with pytest.raises(ParamError, match=match):
            entity_helper.convert_to_array_of_vector([value], field_info)
