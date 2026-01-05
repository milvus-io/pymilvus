"""
ColumnarSearchResult - A drop-in replacement for SearchResult with columnar storage.

Design Principles:
1. Reduce object creation: No pre-creation of nq x topk Hit objects
2. Read-only: Data is referenced from protobuf, not copied
3. Type compatible: All field return types match SearchResult exactly
4. Lazy access: Data is extracted on-demand, not at initialization

Performance Benefits:
- Initialization is O(1) instead of O(nq x topk)
- Memory usage is minimal (just references)
- Ideal for scenarios where only a subset of results is accessed
"""

import collections.abc
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import orjson

from pymilvus.client import entity_helper
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, schema_pb2

if TYPE_CHECKING:
    import numpy as np

# Optional dependencies
try:
    import numpy as np
except ImportError:
    np = None

# ==============================================================================
# Helper Accessor Classes (Performance Optimization)
# ==============================================================================


@runtime_checkable
class FieldAccessor(Protocol):
    """Protocol defining the interface for obtaining a value at a specific relative index."""

    def __call__(self, i: int) -> Any: ...


class BaseAccessor:
    __slots__ = ("data", "start")

    def __init__(self, data: Any, start: int):
        self.data = data
        self.start = start


class ScalarAccessor(BaseAccessor):
    __slots__ = ()

    def get(self, i: int) -> Any:
        return self.data[i + self.start]


class BytesVectorAccessor(BaseAccessor):
    __slots__ = ("bpv",)

    def __init__(self, data: Any, start: int, bpv: int):
        super().__init__(data, start)
        self.bpv = bpv

    def get(self, i: int) -> bytes:
        start_idx = (i + self.start) * self.bpv
        return self.data[start_idx : start_idx + self.bpv]


class FloatVectorAccessor(BaseAccessor):
    __slots__ = ("dim",)

    def __init__(self, data: Any, start: int, dim: int):
        super().__init__(data, start)
        self.dim = dim

    def get(self, i: int) -> List[float]:
        start_idx = (i + self.start) * self.dim
        return self.data[start_idx : start_idx + self.dim]


class Int8VectorAccessor(BaseAccessor):
    __slots__ = ("dim",)

    def __init__(self, data: Any, start: int, dim: int):
        super().__init__(data, start)
        self.dim = dim

    def get(self, i: int) -> bytes:
        start_idx = (i + self.start) * self.dim
        return self.data[start_idx : start_idx + self.dim]


class JsonAccessor(BaseAccessor):
    __slots__ = ()

    def get(self, i: int) -> Any:
        # Optimization: parsing JSON parsing could be cached if likely to be re-accessed,
        # but for now we keep the stateless approach to minimize memory overhead per hit.
        val = self.data[i + self.start]
        return orjson.loads(val) if val else None


class NullableAccessor:
    __slots__ = ("raw_acc", "start", "valid_data")

    def __init__(self, raw_accessor: FieldAccessor, valid_data: Any, start: int):
        self.raw_acc = raw_accessor
        self.valid_data = valid_data
        self.start = start

    def get(self, i: int) -> Any:
        if self.valid_data[i + self.start]:
            return self.raw_acc(i)
        return None


class AccessorFactory:
    """Factory for creating optimized field accessors based on DataType."""

    # Strategy dictionary to map DataType to builder methods
    _BUILDERS: ClassVar[
        Dict[int, Callable[[str, schema_pb2.FieldData, int, Dict[str, Any]], FieldAccessor]]
    ] = {}

    @staticmethod
    def _get_payload(cache: Dict[str, Any], key: str, extractor_func: Callable[[], Any]) -> Any:
        """Helper to get cached payload or extract it."""
        if key in cache:
            return cache[key]
        payload = extractor_func()
        cache[key] = payload
        return payload

    @classmethod
    def _build_float_vector(
        cls, field_name: str, field_data: schema_pb2.FieldData, start: int, cache: Dict[str, Any]
    ) -> FieldAccessor:
        data = cls._get_payload(cache, field_name, lambda: field_data.vectors.float_vector.data)
        dim = field_data.vectors.dim
        return FloatVectorAccessor(data, start, dim).get

    @classmethod
    def _build_binary_vector(
        cls, field_name: str, field_data: schema_pb2.FieldData, start: int, cache: Dict[str, Any]
    ) -> FieldAccessor:
        data = cls._get_payload(cache, field_name, lambda: field_data.vectors.binary_vector)
        bpv = field_data.vectors.dim // 8
        return BytesVectorAccessor(data, start, bpv).get

    @classmethod
    def _build_float16_family_vector(
        cls, field_name: str, field_data: schema_pb2.FieldData, start: int, cache: Dict[str, Any]
    ) -> FieldAccessor:
        field_attr = (
            "float16_vector" if field_data.type == DataType.FLOAT16_VECTOR else "bfloat16_vector"
        )
        data = cls._get_payload(cache, field_name, lambda: getattr(field_data.vectors, field_attr))
        bpv = field_data.vectors.dim * 2
        return BytesVectorAccessor(data, start, bpv).get

    @classmethod
    def _build_int8_vector(
        cls, field_name: str, field_data: schema_pb2.FieldData, start: int, cache: Dict[str, Any]
    ) -> FieldAccessor:
        data = cls._get_payload(cache, field_name, lambda: field_data.vectors.int8_vector)
        dim = field_data.vectors.dim
        return Int8VectorAccessor(data, start, dim).get

    @staticmethod
    def _build_sparse_float_vector(
        _field_name: str, field_data: schema_pb2.FieldData, start: int, _cache: Dict[str, Any]
    ) -> FieldAccessor:
        data = field_data.vectors.sparse_float_vector

        def sparse_accessor(i: int) -> Any:
            return entity_helper.sparse_proto_to_rows(data, i + start, i + start + 1)[0]

        return sparse_accessor

    @classmethod
    def _build_scalar(
        cls,
        field_name: str,
        field_data: schema_pb2.FieldData,
        start: int,
        cache: Dict[str, Any],
        attr_name: str,
    ) -> FieldAccessor:
        data = cls._get_payload(
            cache, field_name, lambda: getattr(field_data.scalars, attr_name).data
        )
        return ScalarAccessor(data, start).get

    @classmethod
    def _build_json(
        cls, field_name: str, field_data: schema_pb2.FieldData, start: int, cache: Dict[str, Any]
    ) -> FieldAccessor:
        data = cls._get_payload(cache, field_name, lambda: field_data.scalars.json_data.data)
        return JsonAccessor(data, start).get

    @staticmethod
    def _build_array(
        _field_name: str, field_data: schema_pb2.FieldData, start: int, _cache: Dict[str, Any]
    ) -> FieldAccessor:
        def array_accessor(i: int) -> Any:
            return entity_helper.extract_array_row_data(field_data, i + start)

        return array_accessor

    @staticmethod
    def _build_struct_array(
        _field_name: str, field_data: schema_pb2.FieldData, start: int, _cache: Dict[str, Any]
    ) -> FieldAccessor:
        def struct_accessor(i: int) -> Any:
            abs_idx = i + start
            if hasattr(field_data, "struct_arrays") and field_data.struct_arrays:
                return entity_helper.extract_struct_array_from_column_data(
                    field_data.struct_arrays, abs_idx
                )
            return None

        return struct_accessor

    @staticmethod
    def _build_vector_array(
        _field_name: str, field_data: schema_pb2.FieldData, start: int, _cache: Dict[str, Any]
    ) -> FieldAccessor:
        def vector_arr_accessor(i: int) -> Any:
            abs_idx = i + start
            if (
                hasattr(field_data, "vectors")
                and hasattr(field_data.vectors, "vector_array")
                and abs_idx < len(field_data.vectors.vector_array.data)
            ):
                vector_data = field_data.vectors.vector_array.data[abs_idx]
                v_dim = vector_data.dim
                f_data = vector_data.float_vector.data
                if len(f_data) == 0:
                    return []
                num_vecs = len(f_data) // v_dim
                return [list(f_data[j * v_dim : (j + 1) * v_dim]) for j in range(num_vecs)]
            return []

        return vector_arr_accessor

    @classmethod
    def _init_builders(cls):
        """Initialize the builders map."""
        if cls._BUILDERS:
            return

        # Vector types
        cls._BUILDERS[DataType.FLOAT_VECTOR] = cls._build_float_vector
        cls._BUILDERS[DataType.BINARY_VECTOR] = cls._build_binary_vector
        cls._BUILDERS[DataType.FLOAT16_VECTOR] = cls._build_float16_family_vector
        cls._BUILDERS[DataType.BFLOAT16_VECTOR] = cls._build_float16_family_vector
        cls._BUILDERS[DataType.INT8_VECTOR] = cls._build_int8_vector
        cls._BUILDERS[DataType.SPARSE_FLOAT_VECTOR] = cls._build_sparse_float_vector

        # Scalar helpers
        def make_scalar_builder(attr_name: str):
            def builder(
                n: str, d: schema_pb2.FieldData, s: int, c: Dict[str, Any]
            ) -> FieldAccessor:
                return cls._build_scalar(n, d, s, c, attr_name)

            return builder

        cls._BUILDERS[DataType.BOOL] = make_scalar_builder("bool_data")
        cls._BUILDERS[DataType.INT8] = make_scalar_builder("int_data")
        cls._BUILDERS[DataType.INT16] = make_scalar_builder("int_data")
        cls._BUILDERS[DataType.INT32] = make_scalar_builder("int_data")
        cls._BUILDERS[DataType.INT64] = make_scalar_builder("long_data")
        cls._BUILDERS[DataType.FLOAT] = make_scalar_builder("float_data")
        cls._BUILDERS[DataType.DOUBLE] = make_scalar_builder("double_data")
        cls._BUILDERS[DataType.VARCHAR] = make_scalar_builder("string_data")
        cls._BUILDERS[DataType.STRING] = make_scalar_builder("string_data")
        cls._BUILDERS[DataType.TIMESTAMPTZ] = make_scalar_builder("string_data")
        cls._BUILDERS[DataType.GEOMETRY] = make_scalar_builder("geometry_wkt_data")

        # JSON
        cls._BUILDERS[DataType.JSON] = cls._build_json

        # Complex
        cls._BUILDERS[DataType.ARRAY] = cls._build_array
        cls._BUILDERS[DataType._ARRAY_OF_STRUCT] = cls._build_struct_array
        cls._BUILDERS[DataType._ARRAY_OF_VECTOR] = cls._build_vector_array

    @staticmethod
    def create_accessor(
        field_name: str,
        field_data: schema_pb2.FieldData,
        start: int,
        column_payload_cache: Dict[str, Any],
    ) -> FieldAccessor:
        """
        Create a callable accessor for the given field data.

        This method identifies the appropriate accessor builder based on the field type
        and returns a function that retrieves the value at a specific index.

        Args:
            field_name: The name of the field, used as a key for caching payload.
            field_data: The Protobuf FieldData object containing columns.
            start: The starting index offset for the current query results.
            column_payload_cache: A shared cache dictionary to store extracted column data,
                                  avoiding redundant extraction across multiple hits/queries.

        Returns:
            A callable function that takes a relative index (int) and returns the value.

        Raises:
            MilvusException: If the field type is not supported.
        """
        if not AccessorFactory._BUILDERS:
            AccessorFactory._init_builders()

        dtype = field_data.type
        builder = AccessorFactory._BUILDERS.get(dtype)

        if builder is None:
            msg = f"Unsupported field type: {dtype}"
            raise MilvusException(message=msg)

        raw_accessor_func = builder(field_name, field_data, start, column_payload_cache)

        # Wrap with nullability check if needed
        valid_data = field_data.valid_data if len(field_data.valid_data) > 0 else None
        if valid_data is not None:
            return NullableAccessor(raw_accessor_func, valid_data, start).get

        return raw_accessor_func


# ==============================================================================


class RowProxy(collections.abc.Mapping):
    """
    A lightweight read-only proxy that represents a single row in search results.

    It does not store data itself, but retrieves it from ColumnarHits on demand.
    Fully compatible with the original Hit dict-like interface.

    This is READ-ONLY - any attempt to modify will raise an error.
    """

    __slots__ = ("_hits", "_idx", "_pk_name")

    def __init__(self, hits: "ColumnarHits", idx: int, pk_name: str):
        self._hits = hits
        self._idx = idx
        self._pk_name = pk_name

    @property
    def highlight(self) -> Optional[Dict[str, Any]]:
        """Return highlight data for this hit if available."""
        return self._hits.get_highlight(self._idx)

    def __len__(self) -> int:
        return len(self.keys())

    def __getitem__(self, key: str) -> Any:
        """Get field value by key. Supports both top-level and entity fields."""
        # Top-level keys
        if key in {self._pk_name, "id"}:
            return self.id
        if key == "distance":
            return self.distance
        if key == "entity":
            return self.to_dict()["entity"]
        if key == "offset" and self._hits.offsets is not None:
            return self.offset

        # Entity field access
        return self._hits.get_value(key, self._idx)

    def __getattr__(self, key: str) -> Any:
        """Support for dot access (hit.field)."""
        return self.__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with default."""
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Materialize this row into a dictionary (creates a copy)."""
        entity = {}
        for field_name in self._hits.fields:
            if field_name == "$meta":
                continue
            entity[field_name] = self._hits.get_value(field_name, self._idx)

        # Add dynamic fields from $meta
        dynamic_names = self._get_dynamic_field_names()
        for name in dynamic_names:
            if name not in entity:
                entity[name] = self._hits.get_value(name, self._idx)

        result = {self._pk_name: self.id, "distance": self.distance, "entity": entity}
        if self._hits.offsets is not None:
            result["offset"] = self.offset
        return result

    @property
    def id(self) -> Union[int, str]:
        """Primary key value."""
        return self._hits.ids[self._idx]

    @property
    def distance(self) -> float:
        """Distance/score value."""
        return self._hits.distances[self._idx]

    @property
    def offset(self) -> Optional[int]:
        """Offset/element_index value (if available)."""
        if self._hits.offsets is not None:
            return self._hits.offsets[self._idx]
        return None

    @property
    def entity(self):
        """Returns self for compatibility with hit.entity.field access."""
        return self

    @property
    def pk(self) -> Union[int, str]:
        """Alias for id."""
        return self.id

    @property
    def score(self) -> float:
        """Alias for distance."""
        return self.distance

    # ===== Dict-like compatibility methods =====

    def _get_dynamic_field_names(self) -> List[str]:
        """Get field names from $meta JSON for dynamic fields."""
        abs_idx = self._idx + self._hits.start

        # Check cache first
        if abs_idx in self._hits._meta_cache:
            return list(self._hits._meta_cache[abs_idx].keys())

        # If not in cache, parse and cache
        meta_field = self._hits._fields_data_map.get("$meta")
        if meta_field is not None and meta_field.type == DataType.JSON:
            json_bytes = meta_field.scalars.json_data.data[abs_idx]
            if json_bytes:
                meta_dict = orjson.loads(json_bytes)
                self._hits._meta_cache[abs_idx] = meta_dict
                return list(meta_dict.keys())

        return []

    def keys(self) -> List[str]:
        """Return field names (compatible with dict.keys())."""
        keys = [self._pk_name, "distance"]
        if self._pk_name != "id":
            keys.append("id")
        if self._hits.offsets is not None:
            keys.append("offset")

        field_names = [f for f in self._hits.fields if f != "$meta"]
        keys.extend(field_names)
        keys.extend(self._get_dynamic_field_names())
        return keys

    def values(self) -> List[Any]:
        """Return field values (compatible with dict.values())."""
        return [self.get(f) for f in self.keys()]

    def items(self) -> List[tuple]:
        """Return (field_name, value) pairs (compatible with dict.items())."""
        return [(f, self.get(f)) for f in self.keys()]

    def __contains__(self, key: str) -> bool:
        """Support 'field in hit' syntax."""
        if key in (self._pk_name, "id", "distance", "entity"):
            return True
        if key == "offset" and self._hits.offsets is not None:
            return True
        if key in self._hits.fields:
            return True
        return key in self._get_dynamic_field_names()

    def __iter__(self):
        """Iterate over field names (compatible with dict iteration)."""
        return iter(self.keys())

    def __str__(self) -> str:
        return str(self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()

    # Read-only enforcement
    def __setitem__(self, key: str, value: Any) -> None:
        msg = "RowProxy is read-only"
        raise TypeError(msg)


class ColumnarHits:
    """
    Holds search results for a single query in columnar format.

    Instead of creating topk Hit objects upfront, it stores references to
    the underlying protobuf data and creates lightweight RowProxy objects
    on demand.

    This class is READ-ONLY.
    """

    __slots__ = (
        "_accessor_cache",
        "_all_offsets",
        "_all_pks",
        "_all_scores",
        "_column_payload_cache",
        "_distances_cache",
        "_dynamic_fields",
        "_fields",
        "_fields_data_map",
        "_highlight_results",
        "_ids_cache",
        "_meta_cache",
        "_offsets_cache",
        "end",
        "output_fields",
        "pk_name",
        "start",
    )

    def __init__(
        self,
        start: int,
        end: int,
        all_pks: List[Union[str, int]],
        all_scores: List[float],
        all_offsets: Optional[List[int]],
        fields_data_map: Dict[str, schema_pb2.FieldData],
        fields: List[str],
        output_fields: List[str],
        pk_name: str,
        column_payload_cache: Dict[str, Any],
        highlight_results: Optional[List] = None,
    ):
        self.start = start
        self.end = end
        self._all_pks = all_pks
        self._all_scores = all_scores
        self._all_offsets = all_offsets
        self.pk_name = pk_name
        self.output_fields = output_fields

        # Shared references from parent (no dict/list creation per instance)
        self._fields_data_map = fields_data_map
        self._fields = fields

        # Shared cache for raw column payloads (avoids redundant extraction/copying)
        self._column_payload_cache = column_payload_cache

        # Dynamic fields = output_fields - actual fields
        self._dynamic_fields = set(output_fields) - set(fields)

        # Highlight results
        self._highlight_results = highlight_results

        # Lazy caches
        self._ids_cache = None
        self._distances_cache = None
        self._offsets_cache = None
        self._meta_cache = {}
        # Accessor cache for O(1) field access (bypasses branching and map lookups)
        self._accessor_cache: Dict[str, FieldAccessor] = {}

    @property
    def ids(self) -> List[Union[str, int]]:
        """Return ids for this query, slicing lazily."""
        if self._ids_cache is None:
            self._ids_cache = self._all_pks[self.start : self.end]
        return self._ids_cache

    @property
    def distances(self) -> List[float]:
        """Return distances for this query, slicing lazily."""
        if self._distances_cache is None:
            self._distances_cache = self._all_scores[self.start : self.end]
        return self._distances_cache

    @property
    def offsets(self) -> Optional[List[int]]:
        """Return offsets for this query, slicing lazily (or None if not present)."""
        if self._all_offsets is None:
            return None
        if self._offsets_cache is None:
            self._offsets_cache = self._all_offsets[self.start : self.end]
        return self._offsets_cache

    @property
    def fields(self) -> List[str]:
        """Field names available in results."""
        return self._fields

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [RowProxy(self, i, self.pk_name) for i in indices]
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            msg = "Index out of range"
            raise IndexError(msg)
        return RowProxy(self, idx, self.pk_name)

    def __iter__(self):
        for i in range(len(self)):
            yield RowProxy(self, i, self.pk_name)

    def __str__(self) -> str:
        """Only print at most 10 query results."""
        items = [str(self[i]) for i in range(min(10, len(self)))]
        reminder = f" ... and {len(self) - 10} entities remaining" if len(self) > 10 else ""
        return f"{items}{reminder}"

    __repr__ = __str__

    def get_highlight(self, idx: int) -> Optional[Dict[str, Any]]:
        """Return highlight data for a specific index."""
        if not self._highlight_results:
            return None

        abs_idx = idx + self.start
        highlight_data = {}
        for result in self._highlight_results:
            if abs_idx < len(result.datas):
                highlight_data[result.field_name] = {
                    "fragments": list(result.datas[abs_idx].fragments),
                    "scores": list(result.datas[abs_idx].scores),
                }
        return highlight_data if highlight_data else None

    def get_value(self, field_name: str, idx: int) -> Any:
        "Retrieve a single value for a field at a specific relative index."
        accessor = self._accessor_cache.get(field_name)
        if accessor is not None:
            return accessor(idx)

        # Slow path: Bind accessor first
        return self._bind_accessor(field_name)(idx)

    def _bind_accessor(self, field_name: str) -> FieldAccessor:
        "Determine field type and bind a fast accessor function for this field."
        field_data = self._fields_data_map.get(field_name)

        if field_data is None:
            # Check dynamic fields ($meta)
            meta_data = self._fields_data_map.get("$meta")
            if meta_data is not None and meta_data.type == DataType.JSON:
                if field_name not in self._column_payload_cache:
                    self._column_payload_cache[field_name] = meta_data.scalars.json_data.data

                json_data = self._column_payload_cache[field_name]
                start = self.start

                def meta_accessor(i: int) -> Any:
                    abs_idx = i + start
                    if abs_idx in self._meta_cache:
                        meta_dict = self._meta_cache[abs_idx]
                    else:
                        json_bytes = json_data[abs_idx]
                        meta_dict = orjson.loads(json_bytes) if json_bytes else {}
                        self._meta_cache[abs_idx] = meta_dict
                    return meta_dict.get(field_name)

                self._accessor_cache[field_name] = meta_accessor
                return meta_accessor

            msg = f"Field '{field_name}' not found"
            raise KeyError(msg)

        # Delegate to AccessorFactory
        accessor = AccessorFactory.create_accessor(
            field_name, field_data, self.start, self._column_payload_cache
        )
        self._accessor_cache[field_name] = accessor
        return accessor

    # ===== Batch access API for advanced users =====

    def get_all_ids(self) -> List[Union[str, int]]:
        """Return all IDs for this query."""
        return self.ids

    def get_all_distances(self) -> List[float]:
        """Return all distances for this query."""
        return self.distances

    def get_column(
        self,
        field_name: str,
        return_type: Literal["list", "numpy"] = "list",
    ) -> Union[List[Any], "np.ndarray"]:
        """
        Get all values for a specific field as a columnar array.

        Args:
            field_name: Name of the field to retrieve.
            return_type: Output format, one of:
                - "list": Python list (default, always works)
                - "numpy": NumPy ndarray (best for numeric types)

        Returns:
            Column data in requested format.

        Raises:
            KeyError: If field_name not found.
            ImportError: If numpy not installed when requested.

        Note:
            For types not compatible with numpy (JSON, ARRAY, SPARSE, dynamic fields),
            requesting "numpy" will log a warning and return "list" instead.
        """
        # Handle special fields
        if field_name in ("id", self.pk_name):
            return self._get_column_formatted(self.ids, return_type, is_pk=True)
        if field_name == "distance":
            return self._get_column_formatted(self.distances, return_type, dtype_hint="float")
        if field_name == "offset":
            if self.offsets is None:
                msg = "Field 'offset' not found (element_indices not enabled)"
                raise KeyError(msg)
            return self._get_column_formatted(self.offsets, return_type, dtype_hint="int")

        # Check if field exists
        field_data = self._fields_data_map.get(field_name)

        if field_data is None:
            # Check if dynamic field
            if field_name in self._dynamic_fields or self._fields_data_map.get("$meta"):
                return self._get_dynamic_column(field_name, return_type)
            msg = f"Field '{field_name}' not found"
            raise KeyError(msg)

        dtype = field_data.type

        if return_type == "list":
            return self._get_column_as_list(field_name, field_data)

        if return_type == "numpy":
            return self._get_column_as_numpy(field_name, field_data, dtype)

        msg = f"Invalid return_type: {return_type}. Must be 'list' or 'numpy'."
        raise ValueError(msg)

    def _get_column_as_list(self, field_name: str, field_data: schema_pb2.FieldData) -> List[Any]:
        """Get column as Python list using existing accessor."""
        accessor = self._accessor_cache.get(field_name) or self._bind_accessor(field_name)
        return [accessor(i) for i in range(len(self))]

    def _get_column_as_numpy(
        self, field_name: str, field_data: schema_pb2.FieldData, dtype: int
    ) -> "np.ndarray":
        """Get column as NumPy array with optimized conversion."""
        if np is None:
            msg = "numpy is required for return_type='numpy'"
            raise ImportError(msg)

        # NumPy-compatible scalar types
        numpy_dtype_map = {
            DataType.BOOL: np.bool_,
            DataType.INT8: np.int8,
            DataType.INT16: np.int16,
            DataType.INT32: np.int32,
            DataType.INT64: np.int64,
            DataType.FLOAT: np.float32,
            DataType.DOUBLE: np.float64,
        }

        start, end = self.start, self.end

        # Scalar types
        # Scalar types - Optimized direct access
        if dtype in numpy_dtype_map:
            np_dtype = numpy_dtype_map[dtype]
            data = None
            if dtype == DataType.BOOL:
                data = field_data.scalars.bool_data.data
            elif dtype in (DataType.INT8, DataType.INT16, DataType.INT32):
                data = field_data.scalars.int_data.data
            elif dtype == DataType.INT64:
                data = field_data.scalars.long_data.data
            elif dtype == DataType.FLOAT:
                data = field_data.scalars.float_data.data
            elif dtype == DataType.DOUBLE:
                data = field_data.scalars.double_data.data

            if data is not None:
                # Direct slice from proto repeated field (acts as list) to numpy
                # This is faster than iterating with accessor
                return np.array(data[start:end], dtype=np_dtype)

            # Fallback for types not handled above (should be covered though)
            accessor = self._accessor_cache.get(field_name) or self._bind_accessor(field_name)
            return np.array([accessor(i) for i in range(len(self))], dtype=np_dtype)

        # Float vector - reshape to 2D (n, dim)
        if dtype == DataType.FLOAT_VECTOR:
            dim = field_data.vectors.dim
            data = field_data.vectors.float_vector.data
            arr = np.asarray(data, dtype=np.float32)
            return arr.reshape(-1, dim)[start:end].copy()

        # Binary vector - reshape to 2D (n, dim//8)
        if dtype == DataType.BINARY_VECTOR:
            dim = field_data.vectors.dim
            bpv = dim // 8
            data = field_data.vectors.binary_vector
            arr = np.frombuffer(data, dtype=np.uint8)
            return arr.reshape(-1, bpv)[start:end].copy()

        # Float16 vector
        if dtype == DataType.FLOAT16_VECTOR:
            dim = field_data.vectors.dim
            data = field_data.vectors.float16_vector
            arr = np.frombuffer(data, dtype=np.float16)
            return arr.reshape(-1, dim)[start:end].copy()

        # INT8 vector
        if dtype == DataType.INT8_VECTOR:
            dim = field_data.vectors.dim
            data = field_data.vectors.int8_vector
            arr = np.frombuffer(data, dtype=np.int8)
            return arr.reshape(-1, dim)[start:end].copy()

        # BFloat16 vector - NumPy doesn't natively support, try ml_dtypes
        if dtype == DataType.BFLOAT16_VECTOR:
            try:
                from ml_dtypes import bfloat16  # noqa: PLC0415

                dim = field_data.vectors.dim
                data = field_data.vectors.bfloat16_vector
                arr = np.frombuffer(data, dtype=bfloat16)
                return arr.reshape(-1, dim)[start:end].copy()
            except ImportError:
                warnings.warn(
                    f"Field '{field_name}' is BFLOAT16_VECTOR which requires 'ml_dtypes' package. "
                    "Falling back to list. Install with: pip install ml_dtypes",
                    UserWarning,
                    stacklevel=3,
                )
                return self._get_column_as_list(field_name, field_data)

        # VARCHAR - use object dtype (no performance benefit, but works)
        if dtype in (DataType.VARCHAR, DataType.STRING):
            accessor = self._accessor_cache.get(field_name) or self._bind_accessor(field_name)
            return np.array([accessor(i) for i in range(len(self))], dtype=object)

        # Non-compatible types - warn and fallback
        warnings.warn(
            f"Field '{field_name}' (type={dtype}) is not numpy-compatible. "
            "Returning list instead.",
            UserWarning,
            stacklevel=3,
        )
        return self._get_column_as_list(field_name, field_data)

    def _get_dynamic_column(
        self, field_name: str, return_type: Literal["list", "numpy"]
    ) -> Union[List[Any], "np.ndarray"]:
        """Get column for dynamic field from $meta JSON."""
        accessor = self._accessor_cache.get(field_name) or self._bind_accessor(field_name)
        values = [accessor(i) for i in range(len(self))]

        if return_type == "list":
            return values

        if return_type == "numpy":
            if np is None:
                msg = "numpy is required for return_type='numpy'"
                raise ImportError(msg)

            # Return object array to maintain consistent return type
            return np.array(values, dtype=object)

        return values

    def _get_column_formatted(
        self,
        data: List[Any],
        return_type: Literal["list", "numpy"],
        is_pk: bool = False,
        dtype_hint: Optional[str] = None,
    ) -> Union[List[Any], "np.ndarray"]:
        """Format pre-extracted column data to requested type."""
        if return_type == "list":
            return list(data)

        if return_type == "numpy":
            if np is None:
                msg = "numpy is required for return_type='numpy'"
                raise ImportError(msg)

            if not data:
                return np.array(data)
            if dtype_hint == "float":
                return np.array(data, dtype=np.float64)
            if is_pk and isinstance(data[0], int):
                return np.array(data, dtype=np.int64)
            return np.array(data)

        # Should be unreachable if called from get_column with valid return_type
        return list(data)


# ==============================================================================
# Main Result Class
# ==============================================================================


class ColumnarSearchResult(list):
    """
    A drop-in replacement for SearchResult that uses columnar storage.

    Key differences from SearchResult:
    1. Initialization is O(1) - no pre-creation of Hit objects
    2. Data is stored in columnar format (references to protobuf)
    3. RowProxy objects are created on-demand during iteration/access
    4. This is READ-ONLY - data cannot be modified

    API Compatibility:
    - Fully compatible with SearchResult iteration patterns
    - All field types return the same Python types as SearchResult
    - Supports indexing, slicing, iteration

    Usage:
        # Works exactly like SearchResult
        for hits in result:
            for hit in hits:
                print(hit.id, hit.distance, hit['field_name'])
    """

    def __init__(
        self,
        res: schema_pb2.SearchResultData,
        round_decimal: Optional[int] = None,
        status: Optional[common_pb2.Status] = None,
        session_ts: Optional[int] = 0,
    ):
        self._res = res
        self._round_decimal = round_decimal
        pk_name = res.primary_field_name or "id"

        # Parse IDs
        if res.ids.HasField("int_id"):
            all_pks = res.ids.int_id.data
        elif res.ids.HasField("str_id"):
            all_pks = res.ids.str_id.data
        else:
            all_pks = []

        # Parse scores with optional rounding
        if isinstance(round_decimal, int) and round_decimal > 0:
            all_scores = [round(x, round_decimal) for x in res.scores]
        else:
            all_scores = res.scores

        # Extract element_indices if present (for offsets)
        all_offsets: Optional[List[int]] = None
        if res.element_indices and len(res.element_indices.data) > 0:
            all_offsets = list(res.element_indices.data)

        # Create shared field map ONCE (not per ColumnarHits)
        fields_data_map = {fd.field_name: fd for fd in res.fields_data}
        fields = list(fields_data_map.keys())

        # Shared payload cache for all hits/queries
        column_payload_cache = {}

        # Get highlight results
        highlight_results = list(res.highlight_results) if res.highlight_results else None

        # Create ColumnarHits for each query
        data = []
        nq_thres = 0
        for topk in res.topks:
            start, end = nq_thres, nq_thres + topk
            data.append(
                ColumnarHits(
                    start,
                    end,
                    all_pks,
                    all_scores,
                    all_offsets,
                    fields_data_map,
                    fields,
                    list(res.output_fields),
                    pk_name,
                    column_payload_cache,
                    highlight_results,
                )
            )
            nq_thres += topk

        super().__init__(data)

        # Set recalls
        self.recalls = res.recalls if len(res.recalls) > 0 else None

        # Set extra info
        self.extra = {}
        if status and status.extra_info:
            if "report_value" in status.extra_info:
                self.extra["cost"] = int(status.extra_info["report_value"])
            if "scanned_remote_bytes" in status.extra_info:
                self.extra["scanned_remote_bytes"] = int(status.extra_info["scanned_remote_bytes"])
            if "scanned_total_bytes" in status.extra_info:
                self.extra["scanned_total_bytes"] = int(status.extra_info["scanned_total_bytes"])
            if "cache_hit_ratio" in status.extra_info:
                self.extra["cache_hit_ratio"] = float(status.extra_info["cache_hit_ratio"])

        # Iterator related
        self._session_ts = session_ts
        self._search_iterator_v2_results = res.search_iterator_v2_results

    def __str__(self) -> str:
        """Only print at most 10 results."""
        result_msg = f"data: {self[:10]}"
        recall_msg = f",recalls: {self.recalls[:10]}" if self.recalls else ""
        extra_msg = f",{self.extra}" if self.extra else ""
        reminder = f" ... and {len(self) - 10} results remaining" if len(self) > 10 else ""
        return f"{result_msg}{recall_msg}{reminder}{extra_msg}"

    __repr__ = __str__

    def materialize(self):
        """
        No-op for compatibility.

        ColumnarSearchResult doesn't need explicit materialization since
        data is accessed on-demand. This method exists for API compatibility
        with SearchResult.
        """

    def get_session_ts(self):
        """Iterator related inner method."""
        return self._session_ts

    def get_search_iterator_v2_results_info(self):
        """Iterator related inner method."""
        return self._search_iterator_v2_results
