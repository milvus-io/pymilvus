"""Internal static metadata for PyMilvus ``DataType`` values.

This module is a data-only registry for facts that are already duplicated by
client consumers. It is intentionally not a public API, and it must stay cheap
to import from Search V1 paths: keep Arrow facts symbolic and do not import
``pyarrow`` or operation-specific packing/decoding code here.

When adding a new supported ``DataType``, add exactly one ``TYPE_INFO`` entry,
choose the narrowest ``TypeFamily``, fill only static protobuf/storage metadata,
and extend ``tests/test_type_info.py`` for family, attributes, row width, and
symbolic Arrow/NumPy facts as applicable.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Union

from pymilvus.client.types import DataType, PlaceholderType
from pymilvus.exceptions import ParamError


class TypeFamily(str, Enum):
    """Coarse type categories shared by later internal consumer migrations."""

    SCALAR = "scalar"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    COMPLEX = "complex"


@dataclass(frozen=True)
class ArrowLayout:
    """Symbolic Arrow shape, translated to concrete ``pyarrow`` objects later."""

    kind: str
    value_type: Optional[str] = None


@dataclass(frozen=True)
class TypeInfo:
    """Immutable static facts for one ``DataType``.

    Keep fields descriptive, not behavioral. Attributes name existing protobuf
    containers, byte metadata describes fixed-width vector storage, placeholder
    fields mirror existing search enums, and dtype/layout strings are symbolic
    facts that consumers may translate in later feature slices.
    """

    dtype: DataType
    family: TypeFamily
    scalar_attr: Optional[str] = None
    vector_attr: Optional[str] = None
    field_attr: Optional[str] = None
    bytes_per_dim: Optional[int] = None
    binary_packed: bool = False
    byte_storage: bool = False
    placeholder: Optional[PlaceholderType] = None
    embedding_list_placeholder: Optional[PlaceholderType] = None
    numpy_dtype: Optional[str] = None
    numpy_fallback_dtype: Optional[str] = None
    bulk_numpy_dtype: Optional[str] = None
    arrow_layout: Optional[ArrowLayout] = None


# Accept enum values and raw enum integers so registry helpers match callers
# that receive serialized schema/type values before coercion.
DataTypeLike = Union[DataType, int]


def _scalar(
    dtype: DataType,
    scalar_attr: str,
    numpy_dtype: Optional[str],
    arrow_layout: ArrowLayout,
) -> TypeInfo:
    """Build scalar descriptors whose runtime and bulk NumPy facts match."""

    return TypeInfo(
        dtype=dtype,
        family=TypeFamily.SCALAR,
        scalar_attr=scalar_attr,
        numpy_dtype=numpy_dtype,
        bulk_numpy_dtype=numpy_dtype,
        arrow_layout=arrow_layout,
    )


# Registry maintenance rules:
# - Every supported DataType enum except NONE and UNKNOWN should have one entry.
# - Keep entries data-only; operation-specific validation, packing, result
#   decoding, and Arrow conversion stay in their consumer modules.
# - Use one protobuf attribute slot per entry: scalar_attr, vector_attr, or
#   field_attr. Complex/nested types may leave behavior metadata unset.
# - Dense fixed-width vectors use bytes_per_dim; binary vectors also set
#   binary_packed so row_width validates the dimension multiple-of-8 rule.
# - byte_storage marks vectors whose payload is represented in byte-oriented
#   protobuf fields, which is broader than binary_packed.
# - ArrowLayout and NumPy dtype values are strings/descriptors only; this module
#   must remain importable without optional Arrow dependencies.
TYPE_INFO: Mapping[DataType, TypeInfo] = MappingProxyType(
    {
        DataType.BOOL: _scalar(DataType.BOOL, "bool_data", "bool", ArrowLayout("scalar", "bool")),
        DataType.INT8: _scalar(DataType.INT8, "int_data", "int8", ArrowLayout("scalar", "int8")),
        DataType.INT16: _scalar(
            DataType.INT16, "int_data", "int16", ArrowLayout("scalar", "int16")
        ),
        DataType.INT32: _scalar(
            DataType.INT32, "int_data", "int32", ArrowLayout("scalar", "int32")
        ),
        DataType.INT64: _scalar(
            DataType.INT64, "long_data", "int64", ArrowLayout("scalar", "int64")
        ),
        DataType.FLOAT: _scalar(
            DataType.FLOAT, "float_data", "float32", ArrowLayout("scalar", "float32")
        ),
        DataType.DOUBLE: _scalar(
            DataType.DOUBLE, "double_data", "float64", ArrowLayout("scalar", "float64")
        ),
        DataType.STRING: _scalar(DataType.STRING, "string_data", "str", ArrowLayout("string")),
        DataType.VARCHAR: _scalar(DataType.VARCHAR, "string_data", "str", ArrowLayout("string")),
        DataType.TEXT: _scalar(DataType.TEXT, "string_data", "str", ArrowLayout("string")),
        DataType.JSON: _scalar(DataType.JSON, "json_data", "str", ArrowLayout("string")),
        DataType.ARRAY: TypeInfo(
            dtype=DataType.ARRAY,
            family=TypeFamily.SCALAR,
            scalar_attr="array_data",
            arrow_layout=ArrowLayout("unsupported"),
        ),
        DataType.GEOMETRY: _scalar(
            DataType.GEOMETRY, "geometry_wkt_data", "str", ArrowLayout("string")
        ),
        DataType.TIMESTAMPTZ: _scalar(
            DataType.TIMESTAMPTZ, "string_data", "str", ArrowLayout("string")
        ),
        DataType.FLOAT_VECTOR: TypeInfo(
            dtype=DataType.FLOAT_VECTOR,
            family=TypeFamily.DENSE_VECTOR,
            vector_attr="float_vector",
            bytes_per_dim=4,
            placeholder=PlaceholderType.FloatVector,
            embedding_list_placeholder=PlaceholderType.EmbListFloatVector,
            numpy_dtype="float32",
            bulk_numpy_dtype="float32",
            arrow_layout=ArrowLayout("list", "float32"),
        ),
        DataType.BINARY_VECTOR: TypeInfo(
            dtype=DataType.BINARY_VECTOR,
            family=TypeFamily.DENSE_VECTOR,
            vector_attr="binary_vector",
            binary_packed=True,
            byte_storage=True,
            placeholder=PlaceholderType.BinaryVector,
            embedding_list_placeholder=PlaceholderType.EmbListBinaryVector,
            numpy_dtype="uint8",
            bulk_numpy_dtype="uint8",
            arrow_layout=ArrowLayout("list", "uint8"),
        ),
        DataType.FLOAT16_VECTOR: TypeInfo(
            dtype=DataType.FLOAT16_VECTOR,
            family=TypeFamily.DENSE_VECTOR,
            vector_attr="float16_vector",
            bytes_per_dim=2,
            byte_storage=True,
            placeholder=PlaceholderType.FLOAT16_VECTOR,
            embedding_list_placeholder=PlaceholderType.EmbListFloat16Vector,
            numpy_dtype="float16",
            bulk_numpy_dtype="uint8",
            arrow_layout=ArrowLayout("list", "uint8"),
        ),
        DataType.BFLOAT16_VECTOR: TypeInfo(
            dtype=DataType.BFLOAT16_VECTOR,
            family=TypeFamily.DENSE_VECTOR,
            vector_attr="bfloat16_vector",
            bytes_per_dim=2,
            byte_storage=True,
            placeholder=PlaceholderType.BFLOAT16_VECTOR,
            embedding_list_placeholder=PlaceholderType.EmbListBFloat16Vector,
            numpy_dtype="bfloat16",
            numpy_fallback_dtype="float16",
            bulk_numpy_dtype="uint8",
            arrow_layout=ArrowLayout("list", "uint8"),
        ),
        DataType.INT8_VECTOR: TypeInfo(
            dtype=DataType.INT8_VECTOR,
            family=TypeFamily.DENSE_VECTOR,
            vector_attr="int8_vector",
            bytes_per_dim=1,
            byte_storage=True,
            placeholder=PlaceholderType.Int8Vector,
            embedding_list_placeholder=PlaceholderType.EmbListInt8Vector,
            numpy_dtype="int8",
            bulk_numpy_dtype="int8",
            arrow_layout=ArrowLayout("list", "int8"),
        ),
        DataType.SPARSE_FLOAT_VECTOR: TypeInfo(
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            family=TypeFamily.SPARSE_VECTOR,
            vector_attr="sparse_float_vector",
            placeholder=PlaceholderType.SparseFloatVector,
            embedding_list_placeholder=PlaceholderType.EmbListSparseFloatVector,
            arrow_layout=ArrowLayout("string"),
        ),
        DataType.STRUCT: TypeInfo(
            dtype=DataType.STRUCT,
            family=TypeFamily.COMPLEX,
            arrow_layout=ArrowLayout("unsupported"),
        ),
        DataType._ARRAY_OF_VECTOR: TypeInfo(
            dtype=DataType._ARRAY_OF_VECTOR,
            family=TypeFamily.COMPLEX,
            vector_attr="vector_array",
            arrow_layout=ArrowLayout("unsupported"),
        ),
        DataType._ARRAY_OF_STRUCT: TypeInfo(
            dtype=DataType._ARRAY_OF_STRUCT,
            family=TypeFamily.COMPLEX,
            field_attr="struct_arrays",
            arrow_layout=ArrowLayout("unsupported"),
        ),
    }
)


def _coerce_dtype(dtype: DataTypeLike) -> DataType:
    """Normalize raw enum integers while rejecting unknown values deterministically."""

    if isinstance(dtype, DataType):
        return dtype
    try:
        return DataType(dtype)
    except (TypeError, ValueError) as exc:
        msg = f"Unsupported DataType: {dtype}"
        raise ParamError(message=msg) from exc


def get_type_info(dtype: DataTypeLike) -> TypeInfo:
    """Return the descriptor for a supported ``DataType`` or raise ``ParamError``."""

    dtype = _coerce_dtype(dtype)
    try:
        return TYPE_INFO[dtype]
    except KeyError as exc:
        msg = f"Unsupported DataType: {dtype}"
        raise ParamError(message=msg) from exc


def _has_family(dtype: DataTypeLike, family: TypeFamily) -> bool:
    try:
        return get_type_info(dtype).family == family
    except ParamError:
        return False


def is_scalar_type(dtype: DataTypeLike) -> bool:
    return _has_family(dtype, TypeFamily.SCALAR)


def is_dense_vector_type(dtype: DataTypeLike) -> bool:
    return _has_family(dtype, TypeFamily.DENSE_VECTOR)


def is_sparse_vector_type(dtype: DataTypeLike) -> bool:
    return _has_family(dtype, TypeFamily.SPARSE_VECTOR)


def is_vector_type(dtype: DataTypeLike) -> bool:
    return is_dense_vector_type(dtype) or is_sparse_vector_type(dtype)


def is_byte_vector_type(dtype: DataTypeLike) -> bool:
    try:
        return get_type_info(dtype).byte_storage
    except ParamError:
        return False


def get_scalar_attr(dtype: DataTypeLike) -> Optional[str]:
    return get_type_info(dtype).scalar_attr


def get_vector_attr(dtype: DataTypeLike) -> Optional[str]:
    return get_type_info(dtype).vector_attr


def get_field_attr(dtype: DataTypeLike) -> Optional[str]:
    return get_type_info(dtype).field_attr


def get_protobuf_attr(dtype: DataTypeLike) -> Optional[str]:
    info = get_type_info(dtype)
    return info.scalar_attr or info.vector_attr or info.field_attr


def get_placeholder_type(
    dtype: DataTypeLike, is_embedding_list: bool = False
) -> Optional[PlaceholderType]:
    info = get_type_info(dtype)
    if is_embedding_list:
        return info.embedding_list_placeholder
    return info.placeholder


def get_numpy_dtype(dtype: DataTypeLike) -> Optional[str]:
    return get_type_info(dtype).numpy_dtype


def get_numpy_fallback_dtype(dtype: DataTypeLike) -> Optional[str]:
    return get_type_info(dtype).numpy_fallback_dtype


def get_bulk_numpy_dtype(dtype: DataTypeLike) -> Optional[str]:
    return get_type_info(dtype).bulk_numpy_dtype


def get_arrow_layout(dtype: DataTypeLike) -> Optional[ArrowLayout]:
    return get_type_info(dtype).arrow_layout


def row_width(dtype: DataTypeLike, dim: int) -> Optional[int]:
    """Return fixed vector row width in bytes.

    Sparse vectors return ``None`` because their rows are not fixed-width
    slices. Non-vector types and invalid dimensions raise ``ParamError`` so
    callers do not silently compute inconsistent offsets.
    """

    info = get_type_info(dtype)
    if info.family == TypeFamily.SPARSE_VECTOR:
        return None
    if info.family != TypeFamily.DENSE_VECTOR:
        msg = f"DataType {info.dtype} does not have a fixed vector row width"
        raise ParamError(message=msg)
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        msg = f"Vector dimension must be a positive integer, got {dim!r}"
        raise ParamError(message=msg)
    if info.binary_packed:
        if dim % 8 != 0:
            msg = f"Binary vector dimension must be a multiple of 8, got {dim}"
            raise ParamError(message=msg)
        return dim // 8
    if info.bytes_per_dim is None:
        msg = f"DataType {info.dtype} does not have bytes-per-dimension metadata"
        raise ParamError(message=msg)
    return dim * info.bytes_per_dim
