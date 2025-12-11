"""
Vector type handlers.

This module contains handlers for all vector types:
- FLOAT_VECTOR
- FLOAT16_VECTOR, BFLOAT16_VECTOR, INT8_VECTOR, BINARY_VECTOR (bytes-based)
- SPARSE_FLOAT_VECTOR
"""

from typing import Any, Dict, Optional

import numpy as np

from pymilvus.client import utils
from pymilvus.client.data_types import DataType
from pymilvus.exceptions import (
    DataNotMatchException,
    ExceptionsMessage,
    ParamError,
)
from pymilvus.grpc_gen import schema_pb2

from .base import VectorHandler


class FloatVectorHandler(VectorHandler):
    """Handler for FLOAT_VECTOR type."""

    @property
    def data_type(self):
        return DataType.FLOAT_VECTOR

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract float vector from FieldData."""
        dim = field_data.vectors.dim
        start_pos = index * dim
        end_pos = start_pos + dim

        if len(field_data.vectors.float_vector.data) < start_pos:
            return

        if context and context.get("strict_float32", False):
            np_arrays = context.get("float_vector_np_arrays", {})
            field_name = field_data.field_name
            if field_name in np_arrays:
                row_data[field_name] = np_arrays[field_name][start_pos:end_pos]
            else:
                row_data[field_name] = field_data.vectors.float_vector.data[start_pos:end_pos]
        else:
            row_data[field_data.field_name] = field_data.vectors.float_vector.data[
                start_pos:end_pos
            ]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack float vectors into FieldData."""
        # TODO: get dimension from field_info
        field_data.vectors.dim = len(entity_values[0])
        all_floats = [f for vector in entity_values for f in vector]
        field_data.vectors.float_vector.data.extend(all_floats)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single float vector value."""
        field_name = field_info["name"]
        try:
            f_value = field_value
            if isinstance(field_value, np.ndarray):
                if field_value.dtype not in ("float32", "float64"):
                    raise ParamError(
                        message="invalid input for float32 vector. Expected an np.ndarray with dtype=float32"
                    )
                f_value = field_value.tolist()

            field_data.vectors.dim = len(f_value)
            field_data.vectors.float_vector.data.extend(f_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def get_numpy_dtype(self) -> Optional[np.dtype]:
        return np.float32

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw float vector data."""
        return field_data.vectors.float_vector.data

    def get_bytes_per_vector(self, dim: int) -> int:
        """Get bytes per vector for float vector (counted by elements)."""
        return dim


class BytesVectorHandler(VectorHandler):
    """Base handler for bytes-based vector types (FLOAT16, BFLOAT16, INT8, BINARY)."""

    def __init__(self, data_type: DataType, bytes_per_element: int):
        """
        Initialize bytes vector handler.

        Args:
            data_type: The DataType enum value
            bytes_per_element: Number of bytes per vector element
        """
        self._data_type = data_type
        self._bytes_per_element = bytes_per_element

    @property
    def data_type(self):
        return self._data_type

    def _get_vector_field(self, field_data: schema_pb2.FieldData) -> bytes:
        """Get the vector field from FieldData based on type."""
        field_map = {
            DataType.FLOAT16_VECTOR: field_data.vectors.float16_vector,
            DataType.BFLOAT16_VECTOR: field_data.vectors.bfloat16_vector,
            DataType.INT8_VECTOR: field_data.vectors.int8_vector,
            DataType.BINARY_VECTOR: field_data.vectors.binary_vector,
        }
        return field_map[self._data_type]

    def _get_vector_field_for_pack(self, field_data: schema_pb2.FieldData):
        """Get the vector field for packing based on type."""
        if self._data_type == DataType.FLOAT16_VECTOR:
            return field_data.vectors.float16_vector
        if self._data_type == DataType.BFLOAT16_VECTOR:
            return field_data.vectors.bfloat16_vector
        if self._data_type == DataType.INT8_VECTOR:
            return field_data.vectors.int8_vector
        if self._data_type == DataType.BINARY_VECTOR:
            return field_data.vectors.binary_vector
        msg = f"Unsupported bytes vector type: {self._data_type}"
        raise ValueError(msg)

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract bytes vector from FieldData."""
        dim = field_data.vectors.dim
        bytes_per_vector = dim * self._bytes_per_element
        start_pos = index * bytes_per_vector
        end_pos = start_pos + bytes_per_vector

        vector_field = self._get_vector_field(field_data)
        if len(vector_field) >= start_pos:
            # Return direct bytes (not [bytes]) to match original materialize() behavior
            row_data[field_data.field_name] = vector_field[start_pos:end_pos]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack bytes vectors into FieldData."""
        if self._data_type == DataType.BINARY_VECTOR:
            field_data.vectors.dim = len(entity_values[0]) * 8
            # BINARY_VECTOR uses assignment, not +=
            field_data.vectors.binary_vector = b"".join(entity_values)
        elif self._data_type == DataType.INT8_VECTOR:
            field_data.vectors.dim = len(entity_values[0])
            field_data.vectors.int8_vector = b"".join(entity_values)
        else:  # FLOAT16_VECTOR, BFLOAT16_VECTOR
            field_data.vectors.dim = len(entity_values[0]) // 2
            if self._data_type == DataType.FLOAT16_VECTOR:
                field_data.vectors.float16_vector = b"".join(entity_values)
            else:  # BFLOAT16_VECTOR
                field_data.vectors.bfloat16_vector = b"".join(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single bytes vector value."""
        field_name = field_info["name"]
        try:
            if self._data_type == DataType.BINARY_VECTOR:
                field_data.vectors.dim = len(field_value) * 8
                vector_field = self._get_vector_field_for_pack(field_data)
                vector_field += bytes(field_value)
            elif self._data_type == DataType.INT8_VECTOR:
                if isinstance(field_value, np.ndarray):
                    if field_value.dtype != "int8":
                        raise ParamError(
                            message="invalid input for int8 vector. Expected an np.ndarray with dtype=int8"
                        )
                    i_bytes = field_value.view(np.int8).tobytes()
                else:
                    raise ParamError(
                        message="invalid input for int8 vector. Expected an np.ndarray with dtype=int8"
                    )
                field_data.vectors.dim = len(i_bytes)
                vector_field = self._get_vector_field_for_pack(field_data)
                vector_field += i_bytes
            else:  # FLOAT16_VECTOR or BFLOAT16_VECTOR
                if isinstance(field_value, bytes):
                    v_bytes = field_value
                elif isinstance(field_value, np.ndarray):
                    expected_dtype = (
                        "float16" if self._data_type == DataType.FLOAT16_VECTOR else "bfloat16"
                    )
                    if field_value.dtype != expected_dtype:
                        raise ParamError(
                            message=f"invalid input for {expected_dtype} vector. Expected an np.ndarray with dtype={expected_dtype}"
                        )
                    v_bytes = field_value.view(np.uint8).tobytes()
                else:
                    expected_dtype = (
                        "float16" if self._data_type == DataType.FLOAT16_VECTOR else "bfloat16"
                    )
                    raise ParamError(
                        message=f"invalid input type for {expected_dtype} vector. Expected an np.ndarray with dtype={expected_dtype}"
                    )
                field_data.vectors.dim = len(v_bytes) // 2
                vector_field = self._get_vector_field_for_pack(field_data)
                vector_field += v_bytes
        except (TypeError, ValueError) as e:
            type_name = self._data_type.name.lower()
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, type_name, type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def get_numpy_dtype(self) -> Optional[np.dtype]:
        """Return numpy dtype for bytes vector types."""
        dtype_map = {
            DataType.FLOAT16_VECTOR: np.float16,
            DataType.BFLOAT16_VECTOR: np.float16,  # Approximation
            DataType.BINARY_VECTOR: np.uint8,
            DataType.INT8_VECTOR: np.int8,
        }
        return dtype_map.get(self._data_type)

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw bytes vector data."""
        return self._get_vector_field(field_data)

    def get_bytes_per_vector(self, dim: int) -> int:
        """Get bytes per vector for bytes vector."""
        if self._data_type == DataType.BINARY_VECTOR:
            return dim // 8
        if self._data_type in (DataType.BFLOAT16_VECTOR, DataType.FLOAT16_VECTOR):
            return dim * 2
        # INT8_VECTOR
        return dim


class SparseFloatVectorHandler(VectorHandler):
    """Handler for SPARSE_FLOAT_VECTOR type."""

    @property
    def data_type(self):
        return DataType.SPARSE_FLOAT_VECTOR

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract sparse float vector from FieldData."""
        row_data[field_data.field_name] = utils.sparse_parse_single_row(
            field_data.vectors.sparse_float_vector.contents[index]
        )

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack sparse float vectors into FieldData."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_data.vectors.sparse_float_vector.CopyFrom(
            entity_helper.sparse_rows_to_proto(entity_values)
        )

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single sparse float vector value."""
        from pymilvus.client import entity_helper  # noqa: PLC0415
        from pymilvus.client.utils import SciPyHelper  # noqa: PLC0415

        field_name = field_info["name"]
        try:
            if not SciPyHelper.is_scipy_sparse(field_value):
                field_value = [field_value]
            elif field_value.shape[0] != 1:
                raise ParamError(message="invalid input for sparse float vector: expect 1 row")
            if not entity_helper.entity_is_sparse_matrix(field_value):
                raise ParamError(message="invalid input for sparse float vector")
            field_data.vectors.sparse_float_vector.contents.append(
                entity_helper.sparse_rows_to_proto(field_value).contents[0]
            )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "sparse_float_vector", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def get_bytes_per_vector(self, dim: int) -> int:
        """Sparse vectors have variable length, return 0 to indicate special handling."""
        # Sparse vectors don't have fixed bytes per vector
        # This method is not used for sparse vectors in batch extraction
        return 0

    def get_numpy_dtype(self) -> Optional[np.dtype]:
        """Sparse vectors don't have a fixed numpy dtype."""
        return None

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw sparse float vector data."""
        return field_data.vectors.sparse_float_vector
