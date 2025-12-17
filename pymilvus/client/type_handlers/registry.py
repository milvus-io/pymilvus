"""
Type handler registry.

This module contains the TypeHandlerRegistry class and global registry functions
for managing and accessing type handlers.
"""

from typing import Dict, Optional

import numpy as np

from pymilvus.client.data_types import DataType

from .base import TypeHandler

# Import all handlers for registration
from .complex_handlers import (
    ArrayHandler,
    ArrayOfStructHandler,
    ArrayOfVectorHandler,
    JsonHandler,
)
from .scalar_handlers import (
    BoolHandler,
    DoubleHandler,
    FloatHandler,
    GeometryHandler,
    Int64Handler,
    IntHandler,
    TimestamptzHandler,
    VarcharHandler,
)
from .vector_handlers import (
    BytesVectorHandler,
    FloatVectorHandler,
    SparseFloatVectorHandler,
)


class TypeHandlerRegistry:
    """Registry for type handlers."""

    def __init__(self):
        self._handlers: Dict[DataType, TypeHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default handlers for all supported types."""
        # Vector types
        self.register(FloatVectorHandler())
        self.register(BytesVectorHandler(DataType.FLOAT16_VECTOR, 2))
        self.register(BytesVectorHandler(DataType.BFLOAT16_VECTOR, 2))
        self.register(BytesVectorHandler(DataType.INT8_VECTOR, 1))
        self.register(BytesVectorHandler(DataType.BINARY_VECTOR, 1))
        self.register(SparseFloatVectorHandler())
        self.register(ArrayOfVectorHandler())
        self.register(ArrayOfStructHandler())

        # Scalar types
        self.register(BoolHandler())
        self.register(IntHandler(DataType.INT8))
        self.register(IntHandler(DataType.INT16))
        self.register(IntHandler(DataType.INT32))
        self.register(Int64Handler())
        self.register(FloatHandler())
        self.register(DoubleHandler())
        self.register(VarcharHandler())
        self.register(TimestamptzHandler())
        self.register(GeometryHandler())

        # Complex types
        self.register(JsonHandler())
        self.register(ArrayHandler())

    def register(self, handler: TypeHandler):
        """Register a type handler."""
        self._handlers[handler.data_type] = handler

    def get_handler(self, data_type: DataType) -> TypeHandler:
        """Get handler for a data type."""
        handler = self._handlers.get(data_type)
        if handler is None:
            msg = f"No handler registered for {data_type}"
            raise ValueError(msg)
        return handler

    def get_numpy_dtype(self, data_type: DataType) -> Optional[np.dtype]:
        """Get numpy dtype for a data type (only vector types have this)."""
        handler = self._handlers.get(data_type)
        if handler and hasattr(handler, "get_numpy_dtype"):
            return handler.get_numpy_dtype()
        return None

    def get_lazy_field_types(self):
        """Get all data types that should be lazily loaded."""
        return {dt for dt, handler in self._handlers.items() if handler.is_lazy_field()}


# Global registry instance (lazy initialization to avoid circular import)
_type_registry: Optional[TypeHandlerRegistry] = None


def _get_registry() -> TypeHandlerRegistry:
    """Get or create the global type registry (lazy initialization)."""
    global _type_registry  # noqa: PLW0603
    if _type_registry is None:
        _type_registry = TypeHandlerRegistry()
    return _type_registry


def get_type_handler(data_type: DataType) -> TypeHandler:
    """Get type handler for a data type (convenience function)."""
    return _get_registry().get_handler(data_type)


def get_type_registry() -> TypeHandlerRegistry:
    """Get the global type registry."""
    return _get_registry()
