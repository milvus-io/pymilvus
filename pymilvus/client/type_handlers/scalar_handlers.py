"""
Scalar type handlers.

This module contains handlers for scalar data types:
- BOOL, INT8, INT16, INT32, INT64
- FLOAT, DOUBLE
- VARCHAR, TIMESTAMPTZ, GEOMETRY
"""

from typing import Any, Dict, Optional

from pymilvus.client.data_types import DataType
from pymilvus.exceptions import (
    DataNotMatchException,
    ExceptionsMessage,
)
from pymilvus.grpc_gen import schema_pb2

from .base import TypeHandler


class BoolHandler(TypeHandler):
    """Handler for BOOL type."""

    @property
    def data_type(self):
        return DataType.BOOL

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract bool from FieldData."""
        if len(field_data.scalars.bool_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.bool_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack bool values into FieldData."""
        field_data.scalars.bool_data.data.extend(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single bool value."""
        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.bool_data.data.extend([])
            else:
                field_data.scalars.bool_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "bool", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw bool data."""
        return field_data.scalars.bool_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract bool data from ScalarField."""
        return list(scalar_field.bool_data.data)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """Pack bool values into ScalarField."""
        scalar_field.bool_data.data.extend(values)


class IntHandler(TypeHandler):
    """Handler for INT8, INT16, INT32 types."""

    def __init__(self, data_type: DataType):
        """
        Initialize int handler.

        Args:
            data_type: The DataType enum value (INT8, INT16, or INT32)
        """
        self._data_type = data_type

    @property
    def data_type(self):
        return self._data_type

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract int from FieldData."""
        if len(field_data.scalars.int_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.int_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack int values into FieldData."""
        field_data.scalars.int_data.data.extend(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single int value."""
        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.int_data.data.extend([])
            else:
                field_data.scalars.int_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw int data."""
        return field_data.scalars.int_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract int data from ScalarField."""
        return list(scalar_field.int_data.data)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """Pack int values into ScalarField."""
        scalar_field.int_data.data.extend(values)


class Int64Handler(TypeHandler):
    """Handler for INT64 type."""

    @property
    def data_type(self):
        return DataType.INT64

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract int64 from FieldData."""
        if len(field_data.scalars.long_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.long_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack int64 values into FieldData."""
        field_data.scalars.long_data.data.extend(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single int64 value."""
        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.long_data.data.extend([])
            else:
                field_data.scalars.long_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "int64", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw int64 data."""
        return field_data.scalars.long_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract int64 data from ScalarField."""
        return list(scalar_field.long_data.data)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """Pack int64 values into ScalarField."""
        scalar_field.long_data.data.extend(values)


class FloatHandler(TypeHandler):
    """Handler for FLOAT type."""

    @property
    def data_type(self):
        return DataType.FLOAT

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract float from FieldData."""
        if len(field_data.scalars.float_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.float_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack float values into FieldData."""
        field_data.scalars.float_data.data.extend(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single float value."""
        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.float_data.data.extend([])
            else:
                field_data.scalars.float_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "float", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw float data."""
        return field_data.scalars.float_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract float data from ScalarField."""
        return list(scalar_field.float_data.data)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """Pack float values into ScalarField."""
        scalar_field.float_data.data.extend(values)


class DoubleHandler(TypeHandler):
    """Handler for DOUBLE type."""

    @property
    def data_type(self):
        return DataType.DOUBLE

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract double from FieldData."""
        if len(field_data.scalars.double_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.double_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack double values into FieldData."""
        field_data.scalars.double_data.data.extend(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single double value."""
        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.double_data.data.extend([])
            else:
                field_data.scalars.double_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "double", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw double data."""
        return field_data.scalars.double_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract double data from ScalarField."""
        return list(scalar_field.double_data.data)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """Pack double values into ScalarField."""
        scalar_field.double_data.data.extend(values)


class VarcharHandler(TypeHandler):
    """Handler for VARCHAR type."""

    @property
    def data_type(self):
        return DataType.VARCHAR

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract varchar from FieldData."""
        if len(field_data.scalars.string_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.string_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack varchar values into FieldData."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_data.scalars.string_data.data.extend(
            entity_helper.entity_to_str_arr(
                entity_values, field_info, entity_helper.CHECK_STR_ARRAY
            )
        )

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single varchar value."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                # convert_to_str_array expects an iterable, wrap single value in list
                converted = entity_helper.convert_to_str_array(
                    [field_value], field_info, entity_helper.CHECK_STR_ARRAY
                )
                field_data.scalars.string_data.data.extend(converted)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "varchar", type(field_value))
                + f" Detail: {e!s}"
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw varchar data."""
        return field_data.scalars.string_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract varchar data from ScalarField."""
        return list(scalar_field.string_data.data)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """Pack varchar values into ScalarField."""
        scalar_field.string_data.data.extend(values)


class TimestamptzHandler(TypeHandler):
    """Handler for TIMESTAMPTZ type."""

    @property
    def data_type(self):
        return DataType.TIMESTAMPTZ

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract timestamptz from FieldData."""
        if len(field_data.scalars.string_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.string_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack timestamptz values into FieldData."""
        field_data.scalars.string_data.data.extend(entity_values)

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single timestamptz value."""
        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.string_data.data.extend([])
            else:
                field_data.scalars.string_data.data.append(field_value)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "string", type(field_value))
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw timestamptz data."""
        return field_data.scalars.string_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract timestamptz data from ScalarField."""
        return list(scalar_field.string_data.data)


class GeometryHandler(TypeHandler):
    """Handler for GEOMETRY type."""

    @property
    def data_type(self):
        return DataType.GEOMETRY

    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract geometry from FieldData."""
        if len(field_data.scalars.geometry_wkt_data.data) > index:
            row_data[field_data.field_name] = field_data.scalars.geometry_wkt_data.data[index]

    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack geometry values into FieldData."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_data.scalars.geometry_wkt_data.data.extend(
            entity_helper.entity_to_str_arr(
                entity_values, field_info, entity_helper.CHECK_STR_ARRAY
            )
        )

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """Pack a single geometry value."""
        from pymilvus.client import entity_helper  # noqa: PLC0415

        field_name = field_info["name"]
        try:
            if field_value is None:
                field_data.scalars.geometry_wkt_data.data.extend([])
            else:
                # convert_to_str_array expects an iterable, wrap single value in list
                converted = entity_helper.convert_to_str_array(
                    [field_value], field_info, entity_helper.CHECK_STR_ARRAY
                )
                field_data.scalars.geometry_wkt_data.data.extend(converted)
        except (TypeError, ValueError) as e:
            raise DataNotMatchException(
                message=ExceptionsMessage.FieldDataInconsistent
                % (field_name, "geometry", type(field_value))
            ) from e

    def is_lazy_field(self) -> bool:
        return False

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """Get raw geometry data."""
        return field_data.scalars.geometry_wkt_data.data

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """Extract geometry data from ScalarField."""
        return list(scalar_field.geometry_wkt_data.data)
