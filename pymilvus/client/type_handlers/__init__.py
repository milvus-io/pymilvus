"""
Type handler registry for data type processing.

This package provides a strategy pattern-based approach to handle different data types,
eliminating the need for scattered if/elif branches across multiple files.

Usage:
    from pymilvus.client.type_handlers import get_type_handler, get_type_registry

    # Get a handler for a specific type
    handler = get_type_handler(DataType.FLOAT_VECTOR)

    # Get the registry for advanced operations
    registry = get_type_registry()
    lazy_types = registry.get_lazy_field_types()
"""

# Export base classes
from .base import (
    TypeHandler,
    VectorHandler,
)

# Export complex handlers
from .complex_handlers import (
    ArrayHandler,
    ArrayOfStructHandler,
    ArrayOfVectorHandler,
    JsonHandler,
)

# Export registry and convenience functions
from .registry import (
    TypeHandlerRegistry,
    get_type_handler,
    get_type_registry,
)

# Export scalar handlers
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

# Export vector handlers
from .vector_handlers import (
    BytesVectorHandler,
    FloatVectorHandler,
    SparseFloatVectorHandler,
)

__all__ = [
    "ArrayHandler",
    "ArrayOfStructHandler",
    "ArrayOfVectorHandler",
    "BoolHandler",
    "BytesVectorHandler",
    "DoubleHandler",
    "FloatHandler",
    "FloatVectorHandler",
    "GeometryHandler",
    "Int64Handler",
    "IntHandler",
    "JsonHandler",
    "SparseFloatVectorHandler",
    "TimestamptzHandler",
    "TypeHandler",
    "TypeHandlerRegistry",
    "VarcharHandler",
    "VectorHandler",
    "get_type_handler",
    "get_type_registry",
]
