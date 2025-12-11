"""
Base classes for type handlers.

This module contains the abstract base classes and common utilities
for all type handlers in the pymilvus client.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from pymilvus.client.data_types import DataType
from pymilvus.grpc_gen import schema_pb2

logger = logging.getLogger(__name__)


class TypeHandler(ABC):
    """Base class for data type handlers.

    Handlers encapsulate type-specific logic for:
    - Reading data from protobuf (extract)
    - Writing data to protobuf (pack)
    - Type metadata (data_type, is_lazy_field)

    Non-type-specific logic (slicing, valid_data handling) should be
    handled by the caller, not in handlers.
    """

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """The data type this handler supports."""

    @abstractmethod
    def extract_from_field_data(
        self,
        field_data: schema_pb2.FieldData,
        index: int,
        row_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Extract a single row from FieldData.

        Args:
            field_data: The FieldData protobuf object
            index: Row index to extract
            row_data: Dictionary to store extracted data
            context: Optional context (e.g., strict_float32, numpy arrays cache)
        """

    @abstractmethod
    def pack_to_field_data(
        self,
        entity_values: list,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """
        Pack entity values into FieldData (for bulk operations).

        Args:
            entity_values: List of values to pack
            field_data: FieldData protobuf object to populate
            field_info: Field schema information
        """

    def pack_single_value(
        self,
        field_value: Any,
        field_data: schema_pb2.FieldData,
        field_info: Dict[str, Any],
    ) -> None:
        """
        Pack a single value into FieldData.

        Default implementation delegates to pack_to_field_data.
        Override only if single value handling requires special validation.

        Args:
            field_value: Single value to pack
            field_data: FieldData protobuf object to populate
            field_info: Field schema information
        """
        self.pack_to_field_data([field_value], field_data, field_info)

    def is_lazy_field(self) -> bool:
        """
        Whether this field type should be lazily loaded.

        Override in subclasses to customize loading policy.
        Default: True (lazy loading preferred for safety).

        Returns:
            True if lazy loading is preferred
        """
        return True

    def get_raw_data(self, field_data: schema_pb2.FieldData) -> Any:
        """
        Get the raw data container from FieldData.

        This method knows which protobuf field path to access
        for this specific type (e.g., scalars.bool_data.data for BOOL).

        Args:
            field_data: The FieldData protobuf object

        Returns:
            The raw data container
        """
        msg = f"get_raw_data not implemented for {self.data_type}"
        raise NotImplementedError(msg)

    def extract_from_scalar_field(self, scalar_field: schema_pb2.ScalarField) -> list:
        """
        Extract data from ScalarField (used for array element extraction).

        Only implement in handlers that can be array elements (scalars).

        Args:
            scalar_field: The ScalarField protobuf object

        Returns:
            List of extracted values
        """
        msg = f"extract_from_scalar_field not implemented for {self.data_type}"
        raise NotImplementedError(msg)

    def pack_to_scalar_field(self, values: list, scalar_field: schema_pb2.ScalarField) -> None:
        """
        Pack values into ScalarField (used for array element packing).

        Only implement in handlers that can be array elements (scalars).

        Args:
            values: List of values to pack
            scalar_field: The ScalarField protobuf object to populate
        """
        msg = f"pack_to_scalar_field not implemented for {self.data_type}"
        raise NotImplementedError(msg)


class VectorHandler(TypeHandler):
    """Abstract base class for vector type handlers.

    Vector handlers always return True for is_lazy_field() and must implement
    get_bytes_per_vector() and get_numpy_dtype().
    """

    def is_lazy_field(self) -> bool:
        """Vector fields are always lazily loaded."""
        return True

    @abstractmethod
    def get_bytes_per_vector(self, dim: int) -> int:
        """
        Get the number of bytes (or elements for float) per vector.

        Args:
            dim: Vector dimension

        Returns:
            Number of bytes per vector (or number of elements for FLOAT_VECTOR)
        """

    @abstractmethod
    def get_numpy_dtype(self) -> Optional[np.dtype]:
        """
        Return the numpy dtype for this vector type.

        Returns:
            numpy dtype for this vector type
        """
