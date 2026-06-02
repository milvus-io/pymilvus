from unittest.mock import patch

import numpy as np
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.entity_helper import convert_to_array, extract_array_row_data_no_validity
from pymilvus.client.search_result import extract_array_row_data as extract_search_array_row_data
from pymilvus.client.type_info import get_array_element_attr
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
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
            (DataType.TEXT, ["text-a", "text-b"]),
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
        assert get_array_element_attr(DataType.TEXT) == "string_data"
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

    def test_extract_array_rows_supports_text_element_type(self):
        field_data = schema_types.FieldData(type=DataType.ARRAY, field_name="text_arr")
        field_data.scalars.array_data.element_type = DataType.TEXT
        field_data.scalars.array_data.data.append(
            schema_types.ScalarField(string_data=schema_types.StringArray(data=["alpha", "beta"]))
        )

        entity_rows = [{}]
        extract_array_row_data_no_validity(field_data, entity_rows, 1)

        assert entity_rows == [{"text_arr": ["alpha", "beta"]}]

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

    def test_search_result_array_extraction_uses_type_info(self):
        scalar = schema_types.ScalarField()
        scalar.string_data.data.extend(["a", "b"])

        assert extract_search_array_row_data([scalar], DataType.TEXT) == [["a", "b"]]

    def test_search_result_rejects_timestamptz_array_element_type(self):
        scalar = schema_types.ScalarField()
        scalar.string_data.data.extend(["2026-05-27T00:00:00Z"])

        with pytest.raises(MilvusException, match="Unsupported data type"):
            extract_search_array_row_data([scalar], DataType.TIMESTAMPTZ)

    def test_registry_backed_array_of_vector_destination(self):
        field_info = {
            "name": "array_vec",
            "element_type": DataType.INT8_VECTOR,
            "params": {"dim": 4},
        }
        values = [
            np.array([1, 2, 3, 4], dtype=np.int8),
            np.array([5, 6, 7, 8], dtype=np.int8),
        ]

        with patch(
            "pymilvus.client.entity_helper.type_info.get_vector_attr",
            wraps=entity_helper.type_info.get_vector_attr,
        ) as get_vector_attr:
            vector_data = entity_helper.convert_to_array_of_vector(values, field_info)

        get_vector_attr.assert_any_call(DataType.INT8_VECTOR)
        assert vector_data.dim == 4
        assert vector_data.int8_vector == b"".join(value.tobytes() for value in values)
