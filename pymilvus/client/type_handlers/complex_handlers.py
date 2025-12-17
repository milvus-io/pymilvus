"""
Complex type handlers.

This module contains handlers for complex data types:
- JSON (with dynamic field handling)
- ARRAY
- _ARRAY_OF_VECTOR (internal type)
- _ARRAY_OF_STRUCT (internal type)
"""

import logging
from typing import Any, Dict, Optional

import orjson

from pymilvus.client.data_types import DataType
from pymilvus.exceptions import (
    DataNotMatchException,
    ExceptionsMessage,
    MilvusException,
)
from pymilvus.grpc_gen import schema_pb2

from .base import TypeHandler

logger = logging.getLogger(__name__)


class JsonHandler(TypeHandler):
    """Handler for JSON type (with special dynamic field handling)."""

    @property
    def data_type(self):
        return DataType.JSON

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract JSON from FieldData with dynamic field handling."""
        if len(field_data.valid_data) > 0 and field_data.valid_data[index] is False:
            row_data[field_data.field_name] = None
            return

        try:
            json_dict = orjson.loads(field_data.scalars.json_data.data[index])
        except Exception as e:
            logger.error(
                f"HybridExtraList::_extract_lazy_fields::Failed to load JSON data: {e}, "
                f"original data: {field_data.scalars.json_data.data[index]}"
            )
            raise

        dynamic_fields = context.get("dynamic_fields") if context else None

        if not field_data.is_dynamic:
            row_data[field_data.field_name] = json_dict
            return

        if not dynamic_fields:
            row_data.update({k: v for k, v in json_dict.items() if k not in row_data})
            return

        row_data.update(
            {k: v for k, v in json_dict.items() if k in dynamic_fields and k not in row_data}
        )

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack JSON values into FieldData."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_data.scalars.json_data.data.extend(
            entity_helper.entity_to_json_arr(entity_values, field_info)
        )

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single JSON value."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.json_data.data.extend([])
            else:
                field_data.scalars.json_data.data.append(entity_helper.convert_to_json(field_value))
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "json", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return True

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw JSON data."""
        return field_data.scalars.json_data.data


class ArrayHandler(TypeHandler):
    """Handler for ARRAY type."""

    @property
    def data_type(self):
        return DataType.ARRAY

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract array from FieldData."""
        if index < len(field_data.scalars.array_data.data):
            array_data = field_data.scalars.array_data.data[index]
            element_type = field_data.scalars.array_data.element_type

            from .registry import get_type_registry  # noqa: PLC0415

            element_handler = get_type_registry().get_handler(element_type)
            try:
                element_data = element_handler.extract_from_scalar_field(array_data)
                # Keep empty lists as-is to match original behavior
                # (extract_array_row_data returns list)
                row_data[field_data.field_name] = element_data
            except NotImplementedError:
                msg = f"Unsupported data type: {element_type}"
                raise MilvusException(message=msg) from None

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack array values into FieldData."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_data.scalars.array_data.data.extend(
            entity_helper.entity_to_array_arr(entity_values, field_info)
        )

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single array value."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.array_data.data.extend([])
            else:
                field_data.scalars.array_data.data.append(
                    entity_helper.convert_to_array(field_value, field_info)
                )
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "array", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False  # ARRAY should be extracted immediately, not lazily

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw array data."""
        return field_data.scalars.array_data.data


class ArrayOfVectorHandler(TypeHandler):
    """Handler for _ARRAY_OF_VECTOR type."""

    @property
    def data_type(self):
        return DataType._ARRAY_OF_VECTOR

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract array of vectors from FieldData."""
        if hasattr(field_data, "vectors") and hasattr(field_data.vectors, "vector_array"):
            if index < len(field_data.vectors.vector_array.data):
                vector_data = field_data.vectors.vector_array.data[index]
                dim = vector_data.dim
                float_data = vector_data.float_vector.data
                num_vectors = len(float_data) // dim
                row_vectors = []
                for vec_idx in range(num_vectors):
                    vec_start = vec_idx * dim
                    vec_end = vec_start + dim
                    row_vectors.append(list(float_data[vec_start:vec_end]))
                row_data[field_data.field_name] = row_vectors
            else:
                row_data[field_data.field_name] = []
        else:
            row_data[field_data.field_name] = []

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack array of vectors into FieldData."""
        # This is handled by convert_to_array_of_vector in entity_helper
        # Keep existing implementation

    def is_lazy_field(self) -> bool:
        return True

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw vector array data."""
        if hasattr(field_data, "vectors") and hasattr(field_data.vectors, "vector_array"):
            return field_data.vectors.vector_array.data
        return None


class ArrayOfStructHandler(TypeHandler):
    """Handler for _ARRAY_OF_STRUCT type."""

    @property
    def data_type(self):
        return DataType._ARRAY_OF_STRUCT

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract array of structs from FieldData."""
        if hasattr(field_data, "struct_arrays") and field_data.struct_arrays:
            from pymilvus.client import entity_helper  # noqa: PLC0415

            row_data[field_data.field_name] = entity_helper.extract_struct_array_from_column_data(
                field_data.struct_arrays, index
            )
        else:
            row_data[field_data.field_name] = None

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack array of structs into FieldData."""
        # This is handled by existing struct array conversion logic

    def is_lazy_field(self) -> bool:
        return True

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw struct array data."""
        return field_data.struct_arrays if hasattr(field_data, "struct_arrays") else None
