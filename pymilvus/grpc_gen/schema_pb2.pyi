from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

Array: DataType
BinaryVector: DataType
Bool: DataType
DESCRIPTOR: _descriptor.FileDescriptor
Double: DataType
FieldCreated: FieldState
FieldCreating: FieldState
FieldDropped: FieldState
FieldDropping: FieldState
Float: DataType
FloatVector: DataType
Int16: DataType
Int32: DataType
Int64: DataType
Int8: DataType
JSON: DataType
None: DataType
String: DataType
VarChar: DataType

class ArrayArray(_message.Message):
    __slots__ = ["data", "element_type"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[ScalarField]
    element_type: DataType
    def __init__(self, data: _Optional[_Iterable[_Union[ScalarField, _Mapping]]] = ..., element_type: _Optional[_Union[DataType, str]] = ...) -> None: ...

class BoolArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, data: _Optional[_Iterable[bool]] = ...) -> None: ...

class BytesArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, data: _Optional[_Iterable[bytes]] = ...) -> None: ...

class CollectionSchema(_message.Message):
    __slots__ = ["autoID", "description", "enable_dynamic_field", "fields", "name"]
    AUTOID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DYNAMIC_FIELD_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    autoID: bool
    description: str
    enable_dynamic_field: bool
    fields: _containers.RepeatedCompositeFieldContainer[FieldSchema]
    name: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., autoID: bool = ..., fields: _Optional[_Iterable[_Union[FieldSchema, _Mapping]]] = ..., enable_dynamic_field: bool = ...) -> None: ...

class DoubleArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class FieldData(_message.Message):
    __slots__ = ["field_id", "field_name", "is_dynamic", "scalars", "type", "vectors"]
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_DYNAMIC_FIELD_NUMBER: _ClassVar[int]
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    field_id: int
    field_name: str
    is_dynamic: bool
    scalars: ScalarField
    type: DataType
    vectors: VectorField
    def __init__(self, type: _Optional[_Union[DataType, str]] = ..., field_name: _Optional[str] = ..., scalars: _Optional[_Union[ScalarField, _Mapping]] = ..., vectors: _Optional[_Union[VectorField, _Mapping]] = ..., field_id: _Optional[int] = ..., is_dynamic: bool = ...) -> None: ...

class FieldSchema(_message.Message):
    __slots__ = ["autoID", "data_type", "default_value", "description", "element_type", "fieldID", "index_params", "is_dynamic", "is_partition_key", "is_primary_key", "name", "state", "type_params"]
    AUTOID_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    FIELDID_FIELD_NUMBER: _ClassVar[int]
    INDEX_PARAMS_FIELD_NUMBER: _ClassVar[int]
    IS_DYNAMIC_FIELD_NUMBER: _ClassVar[int]
    IS_PARTITION_KEY_FIELD_NUMBER: _ClassVar[int]
    IS_PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TYPE_PARAMS_FIELD_NUMBER: _ClassVar[int]
    autoID: bool
    data_type: DataType
    default_value: ValueField
    description: str
    element_type: DataType
    fieldID: int
    index_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    is_dynamic: bool
    is_partition_key: bool
    is_primary_key: bool
    name: str
    state: FieldState
    type_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    def __init__(self, fieldID: _Optional[int] = ..., name: _Optional[str] = ..., is_primary_key: bool = ..., description: _Optional[str] = ..., data_type: _Optional[_Union[DataType, str]] = ..., type_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., index_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., autoID: bool = ..., state: _Optional[_Union[FieldState, str]] = ..., element_type: _Optional[_Union[DataType, str]] = ..., default_value: _Optional[_Union[ValueField, _Mapping]] = ..., is_dynamic: bool = ..., is_partition_key: bool = ...) -> None: ...

class FloatArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class IDs(_message.Message):
    __slots__ = ["int_id", "str_id"]
    INT_ID_FIELD_NUMBER: _ClassVar[int]
    STR_ID_FIELD_NUMBER: _ClassVar[int]
    int_id: LongArray
    str_id: StringArray
    def __init__(self, int_id: _Optional[_Union[LongArray, _Mapping]] = ..., str_id: _Optional[_Union[StringArray, _Mapping]] = ...) -> None: ...

class IntArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[int]] = ...) -> None: ...

class JSONArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, data: _Optional[_Iterable[bytes]] = ...) -> None: ...

class LongArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[int]] = ...) -> None: ...

class ScalarField(_message.Message):
    __slots__ = ["array_data", "bool_data", "bytes_data", "double_data", "float_data", "int_data", "json_data", "long_data", "string_data"]
    ARRAY_DATA_FIELD_NUMBER: _ClassVar[int]
    BOOL_DATA_FIELD_NUMBER: _ClassVar[int]
    BYTES_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    INT_DATA_FIELD_NUMBER: _ClassVar[int]
    JSON_DATA_FIELD_NUMBER: _ClassVar[int]
    LONG_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    array_data: ArrayArray
    bool_data: BoolArray
    bytes_data: BytesArray
    double_data: DoubleArray
    float_data: FloatArray
    int_data: IntArray
    json_data: JSONArray
    long_data: LongArray
    string_data: StringArray
    def __init__(self, bool_data: _Optional[_Union[BoolArray, _Mapping]] = ..., int_data: _Optional[_Union[IntArray, _Mapping]] = ..., long_data: _Optional[_Union[LongArray, _Mapping]] = ..., float_data: _Optional[_Union[FloatArray, _Mapping]] = ..., double_data: _Optional[_Union[DoubleArray, _Mapping]] = ..., string_data: _Optional[_Union[StringArray, _Mapping]] = ..., bytes_data: _Optional[_Union[BytesArray, _Mapping]] = ..., array_data: _Optional[_Union[ArrayArray, _Mapping]] = ..., json_data: _Optional[_Union[JSONArray, _Mapping]] = ...) -> None: ...

class SearchResultData(_message.Message):
    __slots__ = ["fields_data", "ids", "num_queries", "output_fields", "scores", "top_k", "topks"]
    FIELDS_DATA_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    NUM_QUERIES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    TOPKS_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    fields_data: _containers.RepeatedCompositeFieldContainer[FieldData]
    ids: IDs
    num_queries: int
    output_fields: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.RepeatedScalarFieldContainer[float]
    top_k: int
    topks: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, num_queries: _Optional[int] = ..., top_k: _Optional[int] = ..., fields_data: _Optional[_Iterable[_Union[FieldData, _Mapping]]] = ..., scores: _Optional[_Iterable[float]] = ..., ids: _Optional[_Union[IDs, _Mapping]] = ..., topks: _Optional[_Iterable[int]] = ..., output_fields: _Optional[_Iterable[str]] = ...) -> None: ...

class StringArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, data: _Optional[_Iterable[str]] = ...) -> None: ...

class ValueField(_message.Message):
    __slots__ = ["bool_data", "bytes_data", "double_data", "float_data", "int_data", "long_data", "string_data"]
    BOOL_DATA_FIELD_NUMBER: _ClassVar[int]
    BYTES_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    INT_DATA_FIELD_NUMBER: _ClassVar[int]
    LONG_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    bool_data: bool
    bytes_data: bytes
    double_data: float
    float_data: float
    int_data: int
    long_data: int
    string_data: str
    def __init__(self, bool_data: bool = ..., int_data: _Optional[int] = ..., long_data: _Optional[int] = ..., float_data: _Optional[float] = ..., double_data: _Optional[float] = ..., string_data: _Optional[str] = ..., bytes_data: _Optional[bytes] = ...) -> None: ...

class VectorField(_message.Message):
    __slots__ = ["binary_vector", "dim", "float_vector"]
    BINARY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VECTOR_FIELD_NUMBER: _ClassVar[int]
    binary_vector: bytes
    dim: int
    float_vector: FloatArray
    def __init__(self, dim: _Optional[int] = ..., float_vector: _Optional[_Union[FloatArray, _Mapping]] = ..., binary_vector: _Optional[bytes] = ...) -> None: ...

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class FieldState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
