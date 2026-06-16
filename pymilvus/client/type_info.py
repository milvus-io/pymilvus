"""Internal static metadata for PyMilvus ``DataType`` values.

This module is a data-only registry for facts that are already duplicated by
client consumers. It is intentionally not a public API, and it must stay cheap
to import from Search V1 paths: keep Arrow facts symbolic and do not import
``pyarrow`` or operation-specific packing/decoding code here.

Extension rules:
    1. Add exactly one ``TYPE_INFO`` entry for each supported ``DataType``.
    2. Choose the narrowest ``TypeFamily`` that describes existing behavior.
    3. Use ``protobuf_slot`` only for known static protobuf payload locations.
       Leave it as ``None`` when no single payload slot exists.
    4. Fill only static protobuf/storage metadata; keep validation, packing,
       decoding, and Arrow conversion in consumer modules.
    5. Extend ``tests/test_type_info.py`` for family, protobuf slots, row
       width, placeholder, NumPy, and symbolic Arrow facts as applicable.

The registry intentionally stores facts, not operation handlers. Empty optional
fields mean the registry has no shared static fact for that concept; they do
not grant or deny runtime support in insert, search, result, or bulk paths.
"""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Union

from pymilvus.client.types import DataType, PlaceholderType
from pymilvus.exceptions import ParamError


class TypeFamily(str, Enum):
    """High-level static family for a ``DataType``.

    Attributes:
        SCALAR: Field data stored as scalar data, including scalar-shaped
            payloads such as JSON.
        DENSE_VECTOR: Fixed-width vector rows whose row width can be derived
            from dimension metadata.
        SPARSE_VECTOR: Sparse vector rows whose storage is not fixed-width per
            dimension.
        COMPLEX: Nested or structural types that have static registry facts but
            keep operation-specific behavior in existing consumers.
    """

    SCALAR = "scalar"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    COMPLEX = "complex"


class ProtobufSlotKind(str, Enum):
    """Namespace that owns a protobuf payload attribute.

    Attributes:
        SCALAR: Attribute under ``FieldData.scalars``.
        VECTOR: Attribute under ``FieldData.vectors``.
        FIELD: Top-level ``FieldData`` attribute, such as ``struct_arrays``.
    """

    SCALAR = "scalar"
    VECTOR = "vector"
    FIELD = "field"


@dataclass(frozen=True)
class ProtobufSlot:
    """Static protobuf storage location for one ``DataType``.

    Attributes:
        kind: Required protobuf namespace. It tells helpers whether ``attr``
            belongs under scalar data, vector data, or the top-level field data
            object.
        attr: Required attribute name in that namespace. Empty strings are
            invalid. If a type has no single static storage location, leave
            ``TypeInfo.protobuf_slot`` as ``None`` instead of using an empty
            attr.
    """

    kind: ProtobufSlotKind
    attr: str


@dataclass(frozen=True)
class ArrowLayout:
    """Symbolic Arrow layout fact.

    Attributes:
        kind: Required symbolic layout category, such as ``"scalar"``,
            ``"list"``, ``"string"``, or ``"unsupported"``.
        value_type: Optional symbolic element type. It is present only when the
            layout needs an element scalar, for example ``"float32"`` for a
            list of float vectors. Empty means the layout has no element type.
    """

    kind: str
    value_type: Optional[str] = None


@dataclass(frozen=True)
class TypeInfo:
    """Immutable static facts for one ``DataType``.

    Attributes:
        dtype: Required ``DataType`` key. It must match the key used in
            ``TYPE_INFO``.
        family: Required high-level family used by predicate helpers.
        protobuf_slot: Optional static protobuf payload location. ``None``
            means this registry does not record one universal payload location
            for the type, so protobuf attr helpers return ``None``. Use
            ``ProtobufSlot``, not empty strings, when a location is known.
        bytes_per_dim: Optional fixed-width dense-vector storage size. It is
            required for dense vectors except binary-packed vectors, unused for
            scalar and sparse vector types, and invalid for non-vector types.
        binary_packed: True only for ``BINARY_VECTOR``, where row width is
            ``dim // 8`` and ``dim`` must be divisible by 8.
        byte_storage: True for vector payloads stored in byte-oriented
            protobuf fields, including ``BINARY_VECTOR``, ``FLOAT16_VECTOR``,
            ``BFLOAT16_VECTOR``, and ``INT8_VECTOR``.
        placeholder: Optional regular search placeholder enum. Empty means the
            type has no placeholder metadata in this registry.
        embedding_list_placeholder: Optional EmbeddingList placeholder enum.
            Empty means EmbeddingList does not have a distinct placeholder fact
            for the type.
        numpy_dtype: Optional symbolic runtime NumPy dtype string. Empty means
            there is no current runtime dtype fact to share.
        numpy_fallback_dtype: Optional symbolic fallback dtype string used when
            an optional dtype is unavailable.
        bulk_numpy_dtype: Optional symbolic bulk-writer NumPy dtype string.
            Empty means current bulk behavior has no NumPy dtype creator for
            the type.
        array_element_attr: Optional scalar protobuf attr when this type is
            supported as an ARRAY element. Empty means this type is not a
            supported ARRAY element, even if it has scalar protobuf storage.
        arrow_layout: Optional symbolic Arrow layout. Empty means no Arrow
            layout fact is known; use ``ArrowLayout("unsupported")`` when the
            type is known but intentionally unsupported for Arrow conversion.
    """

    dtype: DataType
    family: TypeFamily
    protobuf_slot: Optional[ProtobufSlot] = None
    bytes_per_dim: Optional[int] = None
    binary_packed: bool = False
    byte_storage: bool = False
    placeholder: Optional[PlaceholderType] = None
    embedding_list_placeholder: Optional[PlaceholderType] = None
    numpy_dtype: Optional[str] = None
    numpy_fallback_dtype: Optional[str] = None
    bulk_numpy_dtype: Optional[str] = None
    array_element_attr: Optional[str] = None
    arrow_layout: Optional[ArrowLayout] = None


# Accept enum values and raw enum integers so registry helpers match callers
# that receive serialized schema/type values before coercion.
DataTypeLike = Union[DataType, int]


def _scalar(
    dtype: DataType,
    attr: str,
    numpy_dtype: Optional[str],
    arrow_layout: ArrowLayout,
    *,
    array_element: bool = False,
) -> TypeInfo:
    """Build scalar descriptors whose runtime and bulk NumPy facts match.

    Args:
        dtype: Scalar ``DataType`` key.
        attr: Attribute under ``FieldData.scalars``.
        numpy_dtype: Shared runtime and bulk NumPy dtype string, if known.
        arrow_layout: Symbolic Arrow layout descriptor.
        array_element: Whether this scalar type is supported as an ARRAY
            element and should expose ``attr`` for array-cell decode.

    Returns:
        Immutable scalar ``TypeInfo`` using a scalar protobuf slot.
    """

    return TypeInfo(
        dtype=dtype,
        family=TypeFamily.SCALAR,
        protobuf_slot=ProtobufSlot(ProtobufSlotKind.SCALAR, attr),
        numpy_dtype=numpy_dtype,
        bulk_numpy_dtype=numpy_dtype,
        array_element_attr=attr if array_element else None,
        arrow_layout=arrow_layout,
    )


# Registry maintenance rules:
# - Every supported DataType enum except NONE and UNKNOWN should have one entry.
# - Keep entries data-only; operation-specific validation, packing, result
#   decoding, and Arrow conversion stay in their consumer modules.
# - Use protobuf_slot for known static payload locations. Complex/nested types
#   may leave behavior metadata unset.
# - Dense fixed-width vectors use bytes_per_dim; binary vectors also set
#   binary_packed so row_width validates the dimension multiple-of-8 rule.
# - byte_storage marks vectors whose payload is represented in byte-oriented
#   protobuf fields, which is broader than binary_packed.
# - ArrowLayout and NumPy dtype values are strings/descriptors only; this module
#   must remain importable without optional Arrow dependencies.
TYPE_INFO: Mapping[DataType, TypeInfo] = MappingProxyType(
    {
        DataType.BOOL: _scalar(
            DataType.BOOL,
            "bool_data",
            "bool",
            ArrowLayout("scalar", "bool"),
            array_element=True,
        ),
        DataType.INT8: _scalar(
            DataType.INT8,
            "int_data",
            "int8",
            ArrowLayout("scalar", "int8"),
            array_element=True,
        ),
        DataType.INT16: _scalar(
            DataType.INT16,
            "int_data",
            "int16",
            ArrowLayout("scalar", "int16"),
            array_element=True,
        ),
        DataType.INT32: _scalar(
            DataType.INT32,
            "int_data",
            "int32",
            ArrowLayout("scalar", "int32"),
            array_element=True,
        ),
        DataType.INT64: _scalar(
            DataType.INT64,
            "long_data",
            "int64",
            ArrowLayout("scalar", "int64"),
            array_element=True,
        ),
        DataType.FLOAT: _scalar(
            DataType.FLOAT,
            "float_data",
            "float32",
            ArrowLayout("scalar", "float32"),
            array_element=True,
        ),
        DataType.DOUBLE: _scalar(
            DataType.DOUBLE,
            "double_data",
            "float64",
            ArrowLayout("scalar", "float64"),
            array_element=True,
        ),
        DataType.STRING: _scalar(
            DataType.STRING,
            "string_data",
            "str",
            ArrowLayout("string"),
            array_element=True,
        ),
        DataType.VARCHAR: _scalar(
            DataType.VARCHAR,
            "string_data",
            "str",
            ArrowLayout("string"),
            array_element=True,
        ),
        DataType.TEXT: _scalar(
            DataType.TEXT,
            "string_data",
            "str",
            ArrowLayout("string"),
            array_element=True,
        ),
        DataType.JSON: _scalar(DataType.JSON, "json_data", "str", ArrowLayout("string")),
        DataType.ARRAY: TypeInfo(
            dtype=DataType.ARRAY,
            family=TypeFamily.COMPLEX,
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.SCALAR, "array_data"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "float_vector"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "binary_vector"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "float16_vector"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "bfloat16_vector"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "int8_vector"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "sparse_float_vector"),
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
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.VECTOR, "vector_array"),
            arrow_layout=ArrowLayout("unsupported"),
        ),
        DataType._ARRAY_OF_STRUCT: TypeInfo(
            dtype=DataType._ARRAY_OF_STRUCT,
            family=TypeFamily.COMPLEX,
            protobuf_slot=ProtobufSlot(ProtobufSlotKind.FIELD, "struct_arrays"),
            arrow_layout=ArrowLayout("unsupported"),
        ),
    }
)

_ARRAY_ELEMENT_TYPE_TO_ATTR: Mapping[DataType, str] = MappingProxyType(
    {
        dtype: info.array_element_attr
        for dtype, info in TYPE_INFO.items()
        if info.array_element_attr is not None
    }
)


def _coerce_dtype(dtype: DataTypeLike) -> DataType:
    """Normalize raw enum integers while rejecting unknown values.

    Args:
        dtype: ``DataType`` or raw enum integer to coerce.

    Returns:
        Coerced ``DataType`` value.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``.
    """

    if isinstance(dtype, DataType):
        return dtype
    try:
        return DataType(dtype)
    except (TypeError, ValueError) as exc:
        msg = f"Unsupported DataType: {dtype}"
        raise ParamError(message=msg) from exc


def get_type_info(dtype: DataTypeLike) -> TypeInfo:
    """Return the registry entry for a supported ``DataType``.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Immutable ``TypeInfo`` for supported values.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry, such as
            ``DataType.NONE`` or ``DataType.UNKNOWN``.
    """

    dtype = _coerce_dtype(dtype)
    try:
        return TYPE_INFO[dtype]
    except KeyError as exc:
        msg = f"Unsupported DataType: {dtype}"
        raise ParamError(message=msg) from exc


def _has_family(dtype: DataTypeLike, family: TypeFamily) -> bool:
    """Return family membership without surfacing unsupported-type errors.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect.
        family: Family to compare against.

    Returns:
        True when ``dtype`` is supported and belongs to ``family``; otherwise
        False.
    """

    try:
        return get_type_info(dtype).family == family
    except ParamError:
        return False


def is_scalar_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` is represented as scalar field data.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect.

    Returns:
        True for scalar-family registry entries; otherwise False.
    """

    return _has_family(dtype, TypeFamily.SCALAR)


def is_dense_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` has fixed-width dense vector row semantics.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect.

    Returns:
        True for dense vector registry entries; otherwise False.
    """

    return _has_family(dtype, TypeFamily.DENSE_VECTOR)


def is_sparse_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` stores sparse vector rows without fixed width.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect.

    Returns:
        True for sparse vector registry entries; otherwise False.
    """

    return _has_family(dtype, TypeFamily.SPARSE_VECTOR)


def is_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` is any vector family known to the registry.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect.

    Returns:
        True for dense or sparse vector entries; otherwise False.
    """

    return is_dense_vector_type(dtype) or is_sparse_vector_type(dtype)


def is_byte_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether vector payloads use byte-oriented protobuf storage.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect.

    Returns:
        True when the entry records byte-oriented vector protobuf storage;
        otherwise False.
    """

    try:
        return get_type_info(dtype).byte_storage
    except ParamError:
        return False


_DENSE_FLOAT_VECTOR_PLACEHOLDERS = frozenset(
    {
        PlaceholderType.FloatVector,
        PlaceholderType.FLOAT16_VECTOR,
        PlaceholderType.BFLOAT16_VECTOR,
    }
)


def _has_regular_placeholder(dtype: DataTypeLike, *placeholders: PlaceholderType) -> bool:
    try:
        return get_type_info(dtype).placeholder in placeholders
    except ParamError:
        return False


def is_dense_float_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` is a dense vector with float search semantics."""

    try:
        info = get_type_info(dtype)
    except ParamError:
        return False
    return (
        info.family == TypeFamily.DENSE_VECTOR
        and info.placeholder in _DENSE_FLOAT_VECTOR_PLACEHOLDERS
    )


def is_float_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` is a dense or sparse float vector."""

    return is_sparse_vector_type(dtype) or is_dense_float_vector_type(dtype)


def is_binary_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` is the binary vector type."""

    return _has_regular_placeholder(dtype, PlaceholderType.BinaryVector)


def is_int_vector_type(dtype: DataTypeLike) -> bool:
    """Return whether ``dtype`` is the int vector type."""

    return _has_regular_placeholder(dtype, PlaceholderType.Int8Vector)


def get_protobuf_slot(dtype: DataTypeLike) -> Optional[ProtobufSlot]:
    """Return the static protobuf storage slot for a ``DataType``.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        ``ProtobufSlot`` when a single static payload location is known;
        ``None`` when the type has no universal payload slot in this registry.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    return get_type_info(dtype).protobuf_slot


def get_scalar_attr(dtype: DataTypeLike) -> Optional[str]:
    """Return the scalar namespace protobuf attr for a ``DataType``.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Slot attr only when ``protobuf_slot.kind`` is ``SCALAR``; otherwise
        ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    slot = get_protobuf_slot(dtype)
    if slot is not None and slot.kind == ProtobufSlotKind.SCALAR:
        return slot.attr
    return None


def get_array_element_attr(dtype: DataTypeLike) -> Optional[str]:
    """Return the scalar protobuf attr for a supported ARRAY element type.

    Args:
        dtype: ``DataType`` or raw enum integer to inspect as an array element.

    Returns:
        Scalar data attr for supported scalar array elements; otherwise
        ``None``. Unsupported enum values are intentionally treated as
        unsupported array elements so existing callers can decide whether to
        raise or return ``None``.
    """

    try:
        dtype = _coerce_dtype(dtype)
    except ParamError:
        return None
    return _ARRAY_ELEMENT_TYPE_TO_ATTR.get(dtype)


def get_vector_attr(dtype: DataTypeLike) -> Optional[str]:
    """Return the vector namespace protobuf attr for a ``DataType``.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Slot attr only when ``protobuf_slot.kind`` is ``VECTOR``; otherwise
        ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    slot = get_protobuf_slot(dtype)
    if slot is not None and slot.kind == ProtobufSlotKind.VECTOR:
        return slot.attr
    return None


def get_field_attr(dtype: DataTypeLike) -> Optional[str]:
    """Return the top-level field-data protobuf attr for a ``DataType``.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Slot attr only when ``protobuf_slot.kind`` is ``FIELD``; otherwise
        ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    slot = get_protobuf_slot(dtype)
    if slot is not None and slot.kind == ProtobufSlotKind.FIELD:
        return slot.attr
    return None


def get_protobuf_attr(dtype: DataTypeLike) -> Optional[str]:
    """Return the protobuf attr name for compatibility with existing callers.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Slot attr when ``protobuf_slot`` exists; otherwise ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    slot = get_protobuf_slot(dtype)
    return None if slot is None else slot.attr


def get_placeholder_type(
    dtype: DataTypeLike, is_embedding_list: bool = False
) -> Optional[PlaceholderType]:
    """Return search placeholder metadata without constructing placeholders.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.
        is_embedding_list: Whether to return EmbeddingList-specific placeholder
            metadata.

    Returns:
        Placeholder enum when this registry records one; otherwise ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    info = get_type_info(dtype)
    if is_embedding_list:
        return info.embedding_list_placeholder
    return info.placeholder


def require_placeholder_type(
    dtype: DataTypeLike, is_embedding_list: bool = False
) -> PlaceholderType:
    """Return placeholder metadata or raise when the fact is absent.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.
        is_embedding_list: Whether to require EmbeddingList-specific placeholder
            metadata.

    Returns:
        Required placeholder enum.

    Raises:
        ParamError: If ``dtype`` is unsupported or has no placeholder metadata
            for the requested placeholder kind.
    """

    coerced_dtype = _coerce_dtype(dtype)
    placeholder = get_placeholder_type(coerced_dtype, is_embedding_list=is_embedding_list)
    if placeholder is None:
        raise ParamError(message=f"unsupported data type: {coerced_dtype}")
    return placeholder


def get_numpy_dtype(dtype: DataTypeLike) -> Optional[str]:
    """Return the runtime NumPy dtype string used by vector helpers, if known.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Symbolic NumPy dtype string when known; otherwise ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    return get_type_info(dtype).numpy_dtype


def require_numpy_dtype(dtype: DataTypeLike) -> str:
    """Return runtime NumPy dtype metadata or raise when the fact is absent.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Required symbolic NumPy dtype string.

    Raises:
        ParamError: If ``dtype`` is unsupported or has no runtime NumPy dtype
            metadata.
    """

    coerced_dtype = _coerce_dtype(dtype)
    numpy_dtype = get_numpy_dtype(coerced_dtype)
    if numpy_dtype is None:
        raise ParamError(message=f"Unsupported DataType: {coerced_dtype}")
    return numpy_dtype


def get_numpy_fallback_dtype(dtype: DataTypeLike) -> Optional[str]:
    """Return fallback runtime NumPy dtype metadata for optional dtype support.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Fallback symbolic NumPy dtype string when known; otherwise ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    return get_type_info(dtype).numpy_fallback_dtype


def get_bulk_numpy_dtype(dtype: DataTypeLike) -> Optional[str]:
    """Return the bulk-writer NumPy dtype string, if current behavior has one.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Symbolic bulk-writer NumPy dtype string when known; otherwise ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    return get_type_info(dtype).bulk_numpy_dtype


_NUMPY_DTYPE_TO_VECTOR_TYPE: Mapping[str, DataType] = MappingProxyType(
    {
        "bfloat16": DataType.BFLOAT16_VECTOR,
        "float16": DataType.FLOAT16_VECTOR,
        "float32": DataType.FLOAT_VECTOR,
        "float64": DataType.FLOAT_VECTOR,
        "int8": DataType.INT8_VECTOR,
        "byte": DataType.BINARY_VECTOR,
        "uint8": DataType.BINARY_VECTOR,
    }
)


def get_vector_type_for_numpy_dtype(dtype_name: object) -> Optional[DataType]:
    """Return vector ``DataType`` metadata for a symbolic NumPy dtype name.

    Args:
        dtype_name: Symbolic dtype name or object whose string form is a dtype
            name.

    Returns:
        Vector ``DataType`` when this registry records a search placeholder
        mapping for the dtype name; otherwise ``None``.
    """

    return _NUMPY_DTYPE_TO_VECTOR_TYPE.get(str(dtype_name))


def require_vector_type_for_numpy_dtype(dtype_name: object) -> DataType:
    """Return vector ``DataType`` metadata or raise for unsupported dtype names.

    Args:
        dtype_name: Symbolic dtype name or object whose string form is a dtype
            name.

    Returns:
        Required vector ``DataType``.

    Raises:
        ParamError: If the dtype name has no vector placeholder mapping.
    """

    vector_type = get_vector_type_for_numpy_dtype(dtype_name)
    if vector_type is None:
        raise ParamError(message=f"unsupported data type: {dtype_name}")
    return vector_type


def get_arrow_layout(dtype: DataTypeLike) -> Optional[ArrowLayout]:
    """Return symbolic Arrow layout facts without importing ``pyarrow``.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.

    Returns:
        Symbolic Arrow layout when known; otherwise ``None``.

    Raises:
        ParamError: If ``dtype`` cannot be coerced to ``DataType``, or if the
            enum value is intentionally unsupported by this registry.
    """

    return get_type_info(dtype).arrow_layout


def row_width(dtype: DataTypeLike, dim: int) -> Optional[int]:
    """Return fixed vector row width in bytes.

    Sparse vectors return ``None`` because their rows are not fixed-width
    slices. Non-vector types and invalid dimensions raise ``ParamError`` so
    callers do not silently compute inconsistent offsets.

    Args:
        dtype: ``DataType`` or raw enum integer to look up.
        dim: Positive vector dimension. Binary vectors require a multiple of 8.

    Returns:
        Fixed dense-vector row width in bytes, or ``None`` for sparse vectors.

    Raises:
        ParamError: If ``dtype`` is unsupported, not a vector with fixed row
            width, or ``dim`` is not valid for the vector storage format.
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
