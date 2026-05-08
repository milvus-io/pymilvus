import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from pymilvus.client.type_info import (
    TYPE_INFO,
    ArrowLayout,
    TypeFamily,
    get_arrow_layout,
    get_bulk_numpy_dtype,
    get_field_attr,
    get_numpy_dtype,
    get_numpy_fallback_dtype,
    get_placeholder_type,
    get_protobuf_attr,
    get_scalar_attr,
    get_type_info,
    get_vector_attr,
    is_byte_vector_type,
    is_dense_vector_type,
    is_scalar_type,
    is_sparse_vector_type,
    is_vector_type,
    row_width,
)
from pymilvus.client.types import DataType, PlaceholderType
from pymilvus.exceptions import ParamError

REQUIRED_TYPES = {
    DataType.BOOL,
    DataType.INT8,
    DataType.INT16,
    DataType.INT32,
    DataType.INT64,
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.STRING,
    DataType.VARCHAR,
    DataType.TEXT,
    DataType.JSON,
    DataType.ARRAY,
    DataType.GEOMETRY,
    DataType.TIMESTAMPTZ,
    DataType.FLOAT_VECTOR,
    DataType.BINARY_VECTOR,
    DataType.FLOAT16_VECTOR,
    DataType.BFLOAT16_VECTOR,
    DataType.SPARSE_FLOAT_VECTOR,
    DataType.INT8_VECTOR,
    DataType.STRUCT,
    DataType._ARRAY_OF_VECTOR,
    DataType._ARRAY_OF_STRUCT,
}


def test_registry_contains_required_logical_model_types():
    assert set(TYPE_INFO) >= REQUIRED_TYPES
    assert DataType.NONE not in TYPE_INFO
    assert DataType.UNKNOWN not in TYPE_INFO


def test_registry_and_descriptors_are_immutable():
    info = get_type_info(DataType.FLOAT_VECTOR)

    with pytest.raises(TypeError):
        TYPE_INFO[DataType.UNKNOWN] = info

    with pytest.raises(FrozenInstanceError):
        info.vector_attr = "changed"


@pytest.mark.parametrize(
    "dtype,expected_family",
    [
        (DataType.BOOL, TypeFamily.SCALAR),
        (DataType.JSON, TypeFamily.SCALAR),
        (DataType.ARRAY, TypeFamily.SCALAR),
        (DataType.GEOMETRY, TypeFamily.SCALAR),
        (DataType.TIMESTAMPTZ, TypeFamily.SCALAR),
        (DataType.FLOAT_VECTOR, TypeFamily.DENSE_VECTOR),
        (DataType.BINARY_VECTOR, TypeFamily.DENSE_VECTOR),
        (DataType.SPARSE_FLOAT_VECTOR, TypeFamily.SPARSE_VECTOR),
        (DataType.STRUCT, TypeFamily.COMPLEX),
        (DataType._ARRAY_OF_VECTOR, TypeFamily.COMPLEX),
        (DataType._ARRAY_OF_STRUCT, TypeFamily.COMPLEX),
    ],
)
def test_type_family_metadata(dtype, expected_family):
    assert get_type_info(dtype).family == expected_family


@pytest.mark.parametrize(
    "dtype,scalar,dense,sparse,vector,byte_vector",
    [
        (DataType.INT64, True, False, False, False, False),
        (DataType.JSON, True, False, False, False, False),
        (DataType.FLOAT_VECTOR, False, True, False, True, False),
        (DataType.BINARY_VECTOR, False, True, False, True, True),
        (DataType.FLOAT16_VECTOR, False, True, False, True, True),
        (DataType.BFLOAT16_VECTOR, False, True, False, True, True),
        (DataType.INT8_VECTOR, False, True, False, True, True),
        (DataType.SPARSE_FLOAT_VECTOR, False, False, True, True, False),
        (DataType.STRUCT, False, False, False, False, False),
        (DataType.UNKNOWN, False, False, False, False, False),
    ],
)
def test_type_predicates(dtype, scalar, dense, sparse, vector, byte_vector):
    assert is_scalar_type(dtype) is scalar
    assert is_dense_vector_type(dtype) is dense
    assert is_sparse_vector_type(dtype) is sparse
    assert is_vector_type(dtype) is vector
    assert is_byte_vector_type(dtype) is byte_vector


@pytest.mark.parametrize(
    "dtype,attr",
    [
        (DataType.BOOL, "bool_data"),
        (DataType.INT8, "int_data"),
        (DataType.INT16, "int_data"),
        (DataType.INT32, "int_data"),
        (DataType.INT64, "long_data"),
        (DataType.FLOAT, "float_data"),
        (DataType.DOUBLE, "double_data"),
        (DataType.STRING, "string_data"),
        (DataType.VARCHAR, "string_data"),
        (DataType.TEXT, "string_data"),
        (DataType.TIMESTAMPTZ, "string_data"),
        (DataType.GEOMETRY, "geometry_wkt_data"),
        (DataType.JSON, "json_data"),
        (DataType.ARRAY, "array_data"),
    ],
)
def test_scalar_protobuf_attributes_match_existing_maps(dtype, attr):
    assert get_scalar_attr(dtype) == attr
    assert get_protobuf_attr(dtype) == attr


@pytest.mark.parametrize(
    "dtype,attr",
    [
        (DataType.FLOAT_VECTOR, "float_vector"),
        (DataType.BINARY_VECTOR, "binary_vector"),
        (DataType.BFLOAT16_VECTOR, "bfloat16_vector"),
        (DataType.FLOAT16_VECTOR, "float16_vector"),
        (DataType.INT8_VECTOR, "int8_vector"),
        (DataType.SPARSE_FLOAT_VECTOR, "sparse_float_vector"),
        (DataType._ARRAY_OF_VECTOR, "vector_array"),
    ],
)
def test_vector_protobuf_attributes_match_existing_maps(dtype, attr):
    assert get_vector_attr(dtype) == attr
    assert get_protobuf_attr(dtype) == attr


def test_top_level_protobuf_attributes_for_struct_arrays():
    assert get_field_attr(DataType._ARRAY_OF_STRUCT) == "struct_arrays"
    assert get_protobuf_attr(DataType._ARRAY_OF_STRUCT) == "struct_arrays"


@pytest.mark.parametrize(
    "dtype,dim,expected",
    [
        (DataType.FLOAT_VECTOR, 4, 16),
        (DataType.FLOAT16_VECTOR, 4, 8),
        (DataType.BFLOAT16_VECTOR, 4, 8),
        (DataType.INT8_VECTOR, 4, 4),
        (DataType.BINARY_VECTOR, 16, 2),
        (DataType.SPARSE_FLOAT_VECTOR, 4, None),
    ],
)
def test_row_width(dtype, dim, expected):
    assert row_width(dtype, dim) == expected


@pytest.mark.parametrize("dim", [1, 7, 15])
def test_row_width_rejects_invalid_binary_dimensions(dim):
    with pytest.raises(ParamError, match="multiple of 8"):
        row_width(DataType.BINARY_VECTOR, dim)


@pytest.mark.parametrize("dim", [0, -1, True, 2.5])
def test_row_width_rejects_invalid_dimensions(dim):
    with pytest.raises(ParamError, match="positive integer"):
        row_width(DataType.FLOAT_VECTOR, dim)


@pytest.mark.parametrize("dtype", [DataType.INT64, DataType.STRUCT, DataType._ARRAY_OF_VECTOR])
def test_row_width_rejects_unsupported_types(dtype):
    with pytest.raises(ParamError, match="fixed vector row width"):
        row_width(dtype, 8)


def test_unknown_type_lookup_has_deterministic_error():
    with pytest.raises(ParamError, match="Unsupported DataType"):
        get_type_info(DataType.UNKNOWN)
    with pytest.raises(ParamError, match="Unsupported DataType"):
        get_type_info(123456)


@pytest.mark.parametrize(
    "dtype,regular,embedding_list",
    [
        (DataType.FLOAT_VECTOR, PlaceholderType.FloatVector, PlaceholderType.EmbListFloatVector),
        (DataType.BINARY_VECTOR, PlaceholderType.BinaryVector, PlaceholderType.EmbListBinaryVector),
        (
            DataType.FLOAT16_VECTOR,
            PlaceholderType.FLOAT16_VECTOR,
            PlaceholderType.EmbListFloat16Vector,
        ),
        (
            DataType.BFLOAT16_VECTOR,
            PlaceholderType.BFLOAT16_VECTOR,
            PlaceholderType.EmbListBFloat16Vector,
        ),
        (DataType.INT8_VECTOR, PlaceholderType.Int8Vector, PlaceholderType.EmbListInt8Vector),
        (
            DataType.SPARSE_FLOAT_VECTOR,
            PlaceholderType.SparseFloatVector,
            PlaceholderType.EmbListSparseFloatVector,
        ),
    ],
)
def test_placeholder_metadata_matches_search_prepare(dtype, regular, embedding_list):
    assert get_placeholder_type(dtype) == regular
    assert get_placeholder_type(dtype, is_embedding_list=True) == embedding_list


@pytest.mark.parametrize(
    "dtype,numpy_dtype,fallback_dtype,bulk_numpy_dtype",
    [
        (DataType.FLOAT_VECTOR, "float32", None, "float32"),
        (DataType.BINARY_VECTOR, "uint8", None, "uint8"),
        (DataType.FLOAT16_VECTOR, "float16", None, "uint8"),
        (DataType.BFLOAT16_VECTOR, "bfloat16", "float16", "uint8"),
        (DataType.INT8_VECTOR, "int8", None, "int8"),
        (DataType.JSON, "str", None, "str"),
        (DataType.SPARSE_FLOAT_VECTOR, None, None, None),
    ],
)
def test_numpy_metadata_matches_embedding_list_and_bulk_facts(
    dtype, numpy_dtype, fallback_dtype, bulk_numpy_dtype
):
    assert get_numpy_dtype(dtype) == numpy_dtype
    assert get_numpy_fallback_dtype(dtype) == fallback_dtype
    assert get_bulk_numpy_dtype(dtype) == bulk_numpy_dtype


@pytest.mark.parametrize(
    "dtype,layout",
    [
        (DataType.BOOL, ArrowLayout("scalar", "bool")),
        (DataType.FLOAT, ArrowLayout("scalar", "float32")),
        (DataType.VARCHAR, ArrowLayout("string")),
        (DataType.JSON, ArrowLayout("string")),
        (DataType.FLOAT_VECTOR, ArrowLayout("list", "float32")),
        (DataType.BINARY_VECTOR, ArrowLayout("list", "uint8")),
        (DataType.FLOAT16_VECTOR, ArrowLayout("list", "uint8")),
        (DataType.BFLOAT16_VECTOR, ArrowLayout("list", "uint8")),
        (DataType.INT8_VECTOR, ArrowLayout("list", "int8")),
        (DataType.SPARSE_FLOAT_VECTOR, ArrowLayout("string")),
        (DataType.ARRAY, ArrowLayout("unsupported")),
        (DataType.STRUCT, ArrowLayout("unsupported")),
    ],
)
def test_arrow_layouts_are_symbolic(dtype, layout):
    assert get_arrow_layout(dtype) == layout


@pytest.mark.parametrize(
    "dtype",
    [
        DataType.JSON,
        DataType.ARRAY,
        DataType.GEOMETRY,
        DataType.TIMESTAMPTZ,
        DataType.STRUCT,
        DataType._ARRAY_OF_VECTOR,
        DataType._ARRAY_OF_STRUCT,
    ],
)
def test_complex_and_nested_types_have_static_metadata_only(dtype):
    info = get_type_info(dtype)
    assert info.placeholder is None
    assert info.embedding_list_placeholder is None
    assert info.arrow_layout is not None


def test_type_info_and_search_result_imports_do_not_load_pyarrow():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    script = """
import importlib
import sys

# Package initialization may import optional dataframe integrations in some
# environments. Clear that noise before checking these internal modules.
import pymilvus.client

for name in list(sys.modules):
    if name == "pyarrow" or name.startswith("pyarrow."):
        del sys.modules[name]

sys.modules.pop("pymilvus.client.type_info", None)
sys.modules.pop("pymilvus.client.search_result", None)

importlib.import_module("pymilvus.client.type_info")
importlib.import_module("pymilvus.client.search_result")

loaded = sorted(name for name in sys.modules if name == "pyarrow" or name.startswith("pyarrow."))
if loaded:
    raise SystemExit("pyarrow imported: " + ", ".join(loaded[:5]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
