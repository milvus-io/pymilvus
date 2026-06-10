import numpy as np
import pandas as pd
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.exceptions import DataNotMatchException, DataTypeNotSupportException, ParamError
from pymilvus.orm.prepare import Prepare

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _scalar_schema(auto_id: bool = False):
    """Schema with pk + varchar field."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=auto_id),
            FieldSchema("text", DataType.VARCHAR, max_length=128),
        ]
    )


def _float_vec_schema(auto_id: bool = False):
    """Schema with pk + float-vector field."""
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=auto_id),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=4),
        ]
    )


def _float16_vec_schema():
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT16_VECTOR, dim=4),
        ]
    )


def _bfloat16_vec_schema():
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.BFLOAT16_VECTOR, dim=4),
        ]
    )


def _schema_with_function_output():
    """Schema containing a field marked as function output."""
    schema = CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("text", DataType.VARCHAR, max_length=128),
            FieldSchema("sparse", DataType.SPARSE_FLOAT_VECTOR),
        ]
    )
    # Manually mark the sparse field as function output (normally done by Function validation)
    schema.fields[2].is_function_output = True
    return schema


# ---------------------------------------------------------------------------
# 1. Invalid data type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_data",
    [
        {"key": "value"},
        42,
        "a string",
        3.14,
        None,
    ],
    ids=["dict", "int", "str", "float", "None"],
)
def test_invalid_data_type_raises(bad_data):
    schema = _scalar_schema()
    with pytest.raises(DataTypeNotSupportException):
        Prepare.prepare_data(bad_data, schema)


# ---------------------------------------------------------------------------
# 2. DataFrame path
# ---------------------------------------------------------------------------


class TestDataFramePath:
    def test_normal_dataframe_insert(self):
        schema = _scalar_schema()
        df = pd.DataFrame({"pk": [1, 2, 3], "text": ["a", "b", "c"]})

        result = Prepare.prepare_data(df, schema)

        assert len(result) == 2
        assert result[0] == {"name": "pk", "type": DataType.INT64, "values": [1, 2, 3]}
        assert result[1] == {"name": "text", "type": DataType.VARCHAR, "values": ["a", "b", "c"]}

    def test_auto_id_pk_column_skipped_on_insert(self):
        schema = _scalar_schema(auto_id=True)
        df = pd.DataFrame({"text": ["a", "b"]})

        result = Prepare.prepare_data(df, schema, is_insert=True)

        assert len(result) == 1
        assert result[0]["name"] == "text"
        assert result[0]["values"] == ["a", "b"]

    def test_auto_id_with_non_null_pk_raises(self):
        schema = _scalar_schema(auto_id=True)
        df = pd.DataFrame({"pk": [1, 2], "text": ["a", "b"]})

        with pytest.raises(DataNotMatchException):
            Prepare.prepare_data(df, schema, is_insert=True)

    def test_auto_id_with_all_null_pk_allowed(self):
        """When pk column exists but all values are null, no exception is raised."""
        schema = _scalar_schema(auto_id=True)
        df = pd.DataFrame({"pk": [None, None], "text": ["a", "b"]})

        result = Prepare.prepare_data(df, schema, is_insert=True)

        # pk should be skipped because auto_id + is_insert
        assert len(result) == 1
        assert result[0]["name"] == "text"

    def test_auto_id_upsert_pk_not_skipped(self):
        """On upsert (is_insert=False), pk field should NOT be skipped even with auto_id."""
        schema = _scalar_schema(auto_id=True)
        df = pd.DataFrame({"pk": [10, 20], "text": ["x", "y"]})

        result = Prepare.prepare_data(df, schema, is_insert=False)

        names = [e["name"] for e in result]
        assert "pk" in names
        assert "text" in names

    def test_function_output_field_skipped(self):
        schema = _schema_with_function_output()
        df = pd.DataFrame({"pk": [1], "text": ["hello"]})

        result = Prepare.prepare_data(df, schema)

        names = [e["name"] for e in result]
        assert "sparse" not in names
        assert "pk" in names
        assert "text" in names

    def test_dataframe_missing_column_gives_empty_values(self):
        """If a field's column is absent from the DataFrame, values should be []."""
        schema = _scalar_schema()
        df = pd.DataFrame({"pk": [1, 2]})  # "text" column missing

        result = Prepare.prepare_data(df, schema)

        text_entity = next(e for e in result if e["name"] == "text")
        assert text_entity["values"] == []


# ---------------------------------------------------------------------------
# 3. List / Tuple path — scalar fields
# ---------------------------------------------------------------------------


class TestListPathScalar:
    def test_normal_list_data(self):
        schema = _scalar_schema()
        data = [[1, 2, 3], ["a", "b", "c"]]

        result = Prepare.prepare_data(data, schema)

        assert len(result) == 2
        assert result[0] == {"name": "pk", "type": DataType.INT64, "values": [1, 2, 3]}
        assert result[1] == {"name": "text", "type": DataType.VARCHAR, "values": ["a", "b", "c"]}

    def test_tuple_data_accepted(self):
        schema = _scalar_schema()
        data = ([1, 2], ["a", "b"])

        result = Prepare.prepare_data(data, schema)

        assert len(result) == 2
        assert result[0]["values"] == [1, 2]

    def test_auto_id_skips_pk_field(self):
        schema = _scalar_schema(auto_id=True)
        data = [["a", "b"]]  # Only text field data

        result = Prepare.prepare_data(data, schema, is_insert=True)

        assert len(result) == 1
        assert result[0]["name"] == "text"

    def test_function_output_field_skipped(self):
        schema = _schema_with_function_output()
        data = [[1, 2], ["hello", "world"]]

        result = Prepare.prepare_data(data, schema)

        names = [e["name"] for e in result]
        assert "sparse" not in names

    def test_missing_data_columns_gives_empty_values(self):
        """When data has fewer columns than expected fields, trailing fields get empty values."""
        schema = _scalar_schema()
        data = [[1, 2]]  # Only pk data, text data missing

        result = Prepare.prepare_data(data, schema)

        # The first field gets data, the second should get empty values from IndexError
        text_entity = next(e for e in result if e["name"] == "text")
        assert text_entity["values"] == []

    def test_non_vector_field_with_none_data(self):
        schema = _scalar_schema()
        data = [[1, 2], None]  # text field data is None

        result = Prepare.prepare_data(data, schema)

        text_entity = next(e for e in result if e["name"] == "text")
        assert text_entity["values"] == []


# ---------------------------------------------------------------------------
# 4. List path — FLOAT_VECTOR
# ---------------------------------------------------------------------------


class TestFloatVector:
    def test_ndarray_float32_valid(self):
        schema = _float_vec_schema()
        vec_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        data = [[1], vec_data]

        result = Prepare.prepare_data(data, schema)

        vec_entity = next(e for e in result if e["name"] == "vec")
        assert vec_entity["values"] == vec_data.tolist()

    def test_ndarray_float64_valid(self):
        schema = _float_vec_schema()
        vec_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
        data = [[1], vec_data]

        result = Prepare.prepare_data(data, schema)

        vec_entity = next(e for e in result if e["name"] == "vec")
        assert vec_entity["values"] == vec_data.tolist()

    def test_ndarray_int32_raises(self):
        schema = _float_vec_schema()
        vec_data = np.array([[1, 2, 3, 4]], dtype=np.int32)
        data = [[1], vec_data]

        with pytest.raises(ParamError, match=r"Wrong type for np.ndarray"):
            Prepare.prepare_data(data, schema)

    def test_list_of_ndarrays_float32_valid(self):
        schema = _float_vec_schema()
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        data = [[1, 2], [arr1, arr2]]

        result = Prepare.prepare_data(data, schema)

        vec_entity = next(e for e in result if e["name"] == "vec")
        assert vec_entity["values"] == [arr1.tolist(), arr2.tolist()]

    def test_list_of_ndarrays_int32_raises(self):
        schema = _float_vec_schema()
        arr1 = np.array([1, 2, 3, 4], dtype=np.int32)
        data = [[1], [arr1]]

        with pytest.raises(ParamError, match=r"Wrong type for np.ndarray"):
            Prepare.prepare_data(data, schema)

    def test_plain_list_valid(self):
        schema = _float_vec_schema()
        data = [[1], [[1.0, 2.0, 3.0, 4.0]]]

        result = Prepare.prepare_data(data, schema)

        vec_entity = next(e for e in result if e["name"] == "vec")
        assert vec_entity["values"] == [[1.0, 2.0, 3.0, 4.0]]

    def test_ndarray_auto_id_float_vector(self):
        """With auto_id, pk is skipped so vec is at index 0 of data."""
        schema = _float_vec_schema(auto_id=True)
        vec_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        data = [vec_data]

        result = Prepare.prepare_data(data, schema, is_insert=True)

        assert len(result) == 1
        assert result[0]["name"] == "vec"
        assert result[0]["values"] == vec_data.tolist()


# ---------------------------------------------------------------------------
# 5. List path — FLOAT16_VECTOR
# ---------------------------------------------------------------------------


class TestFloat16Vector:
    def test_list_of_ndarrays_float16_valid(self):
        schema = _float16_vec_schema()
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        arr2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float16)
        data = [[1, 2], [arr1, arr2]]

        result = Prepare.prepare_data(data, schema)

        vec_entity = next(e for e in result if e["name"] == "vec")
        assert len(vec_entity["values"]) == 2
        assert vec_entity["values"][0] == arr1.view(np.uint8).tobytes()
        assert vec_entity["values"][1] == arr2.view(np.uint8).tobytes()

    def test_list_of_ndarrays_wrong_dtype_raises(self):
        schema = _float16_vec_schema()
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data = [[1], [arr]]

        with pytest.raises(ParamError, match=r"Wrong type for np.ndarray"):
            Prepare.prepare_data(data, schema)

    def test_non_ndarray_raises(self):
        schema = _float16_vec_schema()
        data = [[1], [[1.0, 2.0, 3.0, 4.0]]]

        with pytest.raises(ParamError, match="Wrong type for vector field"):
            Prepare.prepare_data(data, schema)


# ---------------------------------------------------------------------------
# 6. List path — BFLOAT16_VECTOR
# ---------------------------------------------------------------------------


class TestBfloat16Vector:
    def test_list_of_ndarrays_bfloat16_valid(self):
        ml_dtypes = pytest.importorskip("ml_dtypes")
        bfloat16 = ml_dtypes.bfloat16

        schema = _bfloat16_vec_schema()
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=bfloat16)
        arr2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=bfloat16)
        data = [[1, 2], [arr1, arr2]]

        result = Prepare.prepare_data(data, schema)

        vec_entity = next(e for e in result if e["name"] == "vec")
        assert len(vec_entity["values"]) == 2
        assert vec_entity["values"][0] == arr1.view(np.uint8).tobytes()
        assert vec_entity["values"][1] == arr2.view(np.uint8).tobytes()

    def test_list_of_ndarrays_wrong_dtype_raises(self):
        schema = _bfloat16_vec_schema()
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        data = [[1], [arr]]

        with pytest.raises(ParamError, match=r"Wrong type for np.ndarray"):
            Prepare.prepare_data(data, schema)

    def test_non_ndarray_raises(self):
        schema = _bfloat16_vec_schema()
        data = [[1], [[1.0, 2.0, 3.0, 4.0]]]

        with pytest.raises(ParamError, match="Wrong type for vector field"):
            Prepare.prepare_data(data, schema)
