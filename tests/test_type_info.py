from dataclasses import FrozenInstanceError, fields

import pytest
from pymilvus.client.type_info import (
    TYPE_INFO,
    ArrowLayout,
    ProtobufSlot,
    ProtobufSlotKind,
    TypeFamily,
    TypeInfo,
    get_arrow_layout,
    get_bulk_numpy_dtype,
    get_field_attr,
    get_numpy_dtype,
    get_numpy_fallback_dtype,
    get_placeholder_type,
    get_protobuf_attr,
    get_protobuf_slot,
    get_scalar_attr,
    get_type_info,
    get_vector_attr,
    get_vector_type_for_numpy_dtype,
    is_byte_vector_type,
    is_dense_vector_type,
    is_scalar_type,
    is_sparse_vector_type,
    is_vector_type,
    require_numpy_dtype,
    require_placeholder_type,
    require_vector_type_for_numpy_dtype,
    row_width,
)
from pymilvus.client.types import DataType, PlaceholderType
from pymilvus.exceptions import ParamError


def test_registry_covers_all_supported_data_types():
    unsupported_types = {DataType.NONE, DataType.UNKNOWN}
    assert set(TYPE_INFO) == set(DataType) - unsupported_types
    assert all(info.dtype == dtype for dtype, info in TYPE_INFO.items())


def test_registry_and_descriptors_are_immutable():
    info = get_type_info(DataType.FLOAT_VECTOR)

    with pytest.raises(TypeError):
        TYPE_INFO[DataType.UNKNOWN] = info

    with pytest.raises(FrozenInstanceError):
        info.protobuf_slot = ProtobufSlot(ProtobufSlotKind.VECTOR, "changed")

    with pytest.raises(FrozenInstanceError):
        info.protobuf_slot.attr = "changed"


def test_type_info_uses_single_protobuf_slot_field():
    info = get_type_info(DataType.FLOAT_VECTOR)
    field_names = {field.name for field in fields(TypeInfo)}

    assert "protobuf_slot" in field_names
    assert {"scalar_attr", "vector_attr", "field_attr"}.isdisjoint(field_names)
    assert not hasattr(info, "scalar_attr")
    assert not hasattr(info, "vector_attr")
    assert not hasattr(info, "field_attr")


def test_populated_protobuf_slots_are_valid():
    for info in TYPE_INFO.values():
        if info.protobuf_slot is None:
            continue
        assert isinstance(info.protobuf_slot.kind, ProtobufSlotKind)
        assert info.protobuf_slot.attr


def test_vector_width_metadata_matches_family_extension_rules():
    for info in TYPE_INFO.values():
        if info.family == TypeFamily.DENSE_VECTOR:
            assert info.bytes_per_dim is not None or info.binary_packed
        else:
            assert info.bytes_per_dim is None
            assert info.binary_packed is False


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
        (DataType.TIMESTAMPTZ, "string_data"),
        (DataType.GEOMETRY, "geometry_wkt_data"),
        (DataType.JSON, "json_data"),
        (DataType.ARRAY, "array_data"),
    ],
)
def test_scalar_protobuf_attributes_match_existing_maps(dtype, attr):
    assert get_protobuf_slot(dtype) == ProtobufSlot(ProtobufSlotKind.SCALAR, attr)
    assert get_scalar_attr(dtype) == attr
    assert get_vector_attr(dtype) is None
    assert get_field_attr(dtype) is None
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
    assert get_protobuf_slot(dtype) == ProtobufSlot(ProtobufSlotKind.VECTOR, attr)
    assert get_scalar_attr(dtype) is None
    assert get_vector_attr(dtype) == attr
    assert get_field_attr(dtype) is None
    assert get_protobuf_attr(dtype) == attr


def test_top_level_protobuf_attributes_for_struct_arrays():
    assert get_protobuf_slot(DataType._ARRAY_OF_STRUCT) == ProtobufSlot(
        ProtobufSlotKind.FIELD, "struct_arrays"
    )
    assert get_scalar_attr(DataType._ARRAY_OF_STRUCT) is None
    assert get_vector_attr(DataType._ARRAY_OF_STRUCT) is None
    assert get_field_attr(DataType._ARRAY_OF_STRUCT) == "struct_arrays"
    assert get_protobuf_attr(DataType._ARRAY_OF_STRUCT) == "struct_arrays"


def test_types_without_single_static_protobuf_slot_return_none():
    assert get_protobuf_slot(DataType.STRUCT) is None
    assert get_scalar_attr(DataType.STRUCT) is None
    assert get_vector_attr(DataType.STRUCT) is None
    assert get_field_attr(DataType.STRUCT) is None
    assert get_protobuf_attr(DataType.STRUCT) is None


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
    assert require_placeholder_type(dtype) == regular
    assert require_placeholder_type(dtype, is_embedding_list=True) == embedding_list


@pytest.mark.parametrize("dtype", [DataType.INT64, DataType.JSON, DataType.ARRAY])
def test_require_placeholder_type_rejects_types_without_placeholder_metadata(dtype):
    with pytest.raises(ParamError, match="unsupported data type"):
        require_placeholder_type(dtype)


def test_require_placeholder_type_preserves_unsupported_dtype_error():
    with pytest.raises(ParamError, match="Unsupported DataType"):
        require_placeholder_type(DataType.UNKNOWN)


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
    "dtype,numpy_dtype",
    [
        (DataType.FLOAT_VECTOR, "float32"),
        (DataType.BINARY_VECTOR, "uint8"),
        (DataType.FLOAT16_VECTOR, "float16"),
        (DataType.BFLOAT16_VECTOR, "bfloat16"),
        (DataType.INT8_VECTOR, "int8"),
    ],
)
def test_require_numpy_dtype_returns_present_metadata(dtype, numpy_dtype):
    assert require_numpy_dtype(dtype) == numpy_dtype


def test_require_numpy_dtype_rejects_missing_metadata():
    with pytest.raises(ParamError, match="Unsupported DataType"):
        require_numpy_dtype(DataType.SPARSE_FLOAT_VECTOR)


@pytest.mark.parametrize(
    "dtype_name,vector_type",
    [
        ("float32", DataType.FLOAT_VECTOR),
        ("float64", DataType.FLOAT_VECTOR),
        ("float16", DataType.FLOAT16_VECTOR),
        ("bfloat16", DataType.BFLOAT16_VECTOR),
        ("int8", DataType.INT8_VECTOR),
        ("uint8", DataType.BINARY_VECTOR),
        ("byte", DataType.BINARY_VECTOR),
    ],
)
def test_numpy_dtype_name_vector_type_metadata(dtype_name, vector_type):
    assert get_vector_type_for_numpy_dtype(dtype_name) == vector_type
    assert require_vector_type_for_numpy_dtype(dtype_name) == vector_type


def test_require_vector_type_for_numpy_dtype_rejects_unknown_dtype_name():
    assert get_vector_type_for_numpy_dtype("complex64") is None
    with pytest.raises(ParamError, match="unsupported data type: complex64"):
        require_vector_type_for_numpy_dtype("complex64")


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
