"""Data type definitions - independent module to avoid circular dependencies."""

from enum import IntEnum

from pymilvus.grpc_gen import schema_pb2


class DataType(IntEnum):
    """
    DataType enumeration.

    String of DataType is str of its value, e.g.: str(DataType.BOOL) == "1"
    """

    NONE = 0  # schema_pb2.None, this is an invalid representation in python
    BOOL = schema_pb2.Bool
    INT8 = schema_pb2.Int8
    INT16 = schema_pb2.Int16
    INT32 = schema_pb2.Int32
    INT64 = schema_pb2.Int64

    FLOAT = schema_pb2.Float
    DOUBLE = schema_pb2.Double

    STRING = schema_pb2.String
    VARCHAR = schema_pb2.VarChar
    ARRAY = schema_pb2.Array
    JSON = schema_pb2.JSON
    GEOMETRY = schema_pb2.Geometry
    TIMESTAMPTZ = schema_pb2.Timestamptz

    BINARY_VECTOR = schema_pb2.BinaryVector
    FLOAT_VECTOR = schema_pb2.FloatVector
    FLOAT16_VECTOR = schema_pb2.Float16Vector
    BFLOAT16_VECTOR = schema_pb2.BFloat16Vector
    SPARSE_FLOAT_VECTOR = schema_pb2.SparseFloatVector
    INT8_VECTOR = schema_pb2.Int8Vector

    STRUCT = schema_pb2.Struct

    # Internal use only - not exposed to users
    _ARRAY_OF_VECTOR = schema_pb2.ArrayOfVector
    _ARRAY_OF_STRUCT = schema_pb2.ArrayOfStruct

    UNKNOWN = 999

    def __str__(self) -> str:
        return str(self.value)
