from enum import IntEnum

class DataType(IntEnum):
    NONE = 0
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5

    FLOAT = 10
    DOUBLE = 11

    STRING = 20

    BINARY_VECTOR = 100
    FLOAT_VECTOR = 101

    UNKNOWN = 999