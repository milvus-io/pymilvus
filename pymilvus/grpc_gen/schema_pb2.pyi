from . import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    None: _ClassVar[DataType]
    Bool: _ClassVar[DataType]
    Int8: _ClassVar[DataType]
    Int16: _ClassVar[DataType]
    Int32: _ClassVar[DataType]
    Int64: _ClassVar[DataType]
    Float: _ClassVar[DataType]
    Double: _ClassVar[DataType]
    String: _ClassVar[DataType]
    VarChar: _ClassVar[DataType]
    Array: _ClassVar[DataType]
    JSON: _ClassVar[DataType]
    BinaryVector: _ClassVar[DataType]
    FloatVector: _ClassVar[DataType]

class FieldState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    FieldCreated: _ClassVar[FieldState]
    FieldCreating: _ClassVar[FieldState]
    FieldDropping: _ClassVar[FieldState]
    FieldDropped: _ClassVar[FieldState]
None: DataType
Bool: DataType
Int8: DataType
Int16: DataType
Int32: DataType
Int64: DataType
Float: DataType
Double: DataType
String: DataType
VarChar: DataType
Array: DataType
JSON: DataType
BinaryVector: DataType
FloatVector: DataType
FieldCreated: FieldState
FieldCreating: FieldState
FieldDropping: FieldState
FieldDropped: FieldState

class FieldSchema(_message.Message):
    __slots__ = ["fieldID", "name", "is_primary_key", "description", "data_type", "type_params", "index_params", "autoID", "state", "element_type", "default_value", "is_dynamic", "is_partition_key"]
    FIELDID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    IS_PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    TYPE_PARAMS_FIELD_NUMBER: _ClassVar[int]
    INDEX_PARAMS_FIELD_NUMBER: _ClassVar[int]
    AUTOID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    IS_DYNAMIC_FIELD_NUMBER: _ClassVar[int]
    IS_PARTITION_KEY_FIELD_NUMBER: _ClassVar[int]
    fieldID: int
    name: str
    is_primary_key: bool
    description: str
    data_type: DataType
    type_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    index_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    autoID: bool
    state: FieldState
    element_type: DataType
    default_value: ValueField
    is_dynamic: bool
    is_partition_key: bool
    def __init__(self, fieldID: _Optional[int] = ..., name: _Optional[str] = ..., is_primary_key: bool = ..., description: _Optional[str] = ..., data_type: _Optional[_Union[DataType, str]] = ..., type_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., index_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., autoID: bool = ..., state: _Optional[_Union[FieldState, str]] = ..., element_type: _Optional[_Union[DataType, str]] = ..., default_value: _Optional[_Union[ValueField, _Mapping]] = ..., is_dynamic: bool = ..., is_partition_key: bool = ...) -> None: ...

class CollectionSchema(_message.Message):
    __slots__ = ["name", "description", "autoID", "fields", "enable_dynamic_field"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    AUTOID_FIELD_NUMBER: _ClassVar[int]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DYNAMIC_FIELD_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    autoID: bool
    fields: _containers.RepeatedCompositeFieldContainer[FieldSchema]
    enable_dynamic_field: bool
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., autoID: bool = ..., fields: _Optional[_Iterable[_Union[FieldSchema, _Mapping]]] = ..., enable_dynamic_field: bool = ...) -> None: ...

class BoolArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, data: _Optional[_Iterable[bool]] = ...) -> None: ...

class IntArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[int]] = ...) -> None: ...

class LongArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[int]] = ...) -> None: ...

class FloatArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class DoubleArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class BytesArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, data: _Optional[_Iterable[bytes]] = ...) -> None: ...

class StringArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, data: _Optional[_Iterable[str]] = ...) -> None: ...

class ArrayArray(_message.Message):
    __slots__ = ["data", "element_type"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[ScalarField]
    element_type: DataType
    def __init__(self, data: _Optional[_Iterable[_Union[ScalarField, _Mapping]]] = ..., element_type: _Optional[_Union[DataType, str]] = ...) -> None: ...

class JSONArray(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, data: _Optional[_Iterable[bytes]] = ...) -> None: ...

class ValueField(_message.Message):
    __slots__ = ["bool_data", "int_data", "long_data", "float_data", "double_data", "string_data", "bytes_data"]
    BOOL_DATA_FIELD_NUMBER: _ClassVar[int]
    INT_DATA_FIELD_NUMBER: _ClassVar[int]
    LONG_DATA_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    BYTES_DATA_FIELD_NUMBER: _ClassVar[int]
    bool_data: bool
    int_data: int
    long_data: int
    float_data: float
    double_data: float
    string_data: str
    bytes_data: bytes
    def __init__(self, bool_data: bool = ..., int_data: _Optional[int] = ..., long_data: _Optional[int] = ..., float_data: _Optional[float] = ..., double_data: _Optional[float] = ..., string_data: _Optional[str] = ..., bytes_data: _Optional[bytes] = ...) -> None: ...

class ScalarField(_message.Message):
    __slots__ = ["bool_data", "int_data", "long_data", "float_data", "double_data", "string_data", "bytes_data", "array_data", "json_data"]
    BOOL_DATA_FIELD_NUMBER: _ClassVar[int]
    INT_DATA_FIELD_NUMBER: _ClassVar[int]
    LONG_DATA_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    BYTES_DATA_FIELD_NUMBER: _ClassVar[int]
    ARRAY_DATA_FIELD_NUMBER: _ClassVar[int]
    JSON_DATA_FIELD_NUMBER: _ClassVar[int]
    bool_data: BoolArray
    int_data: IntArray
    long_data: LongArray
    float_data: FloatArray
    double_data: DoubleArray
    string_data: StringArray
    bytes_data: BytesArray
    array_data: ArrayArray
    json_data: JSONArray
    def __init__(self, bool_data: _Optional[_Union[BoolArray, _Mapping]] = ..., int_data: _Optional[_Union[IntArray, _Mapping]] = ..., long_data: _Optional[_Union[LongArray, _Mapping]] = ..., float_data: _Optional[_Union[FloatArray, _Mapping]] = ..., double_data: _Optional[_Union[DoubleArray, _Mapping]] = ..., string_data: _Optional[_Union[StringArray, _Mapping]] = ..., bytes_data: _Optional[_Union[BytesArray, _Mapping]] = ..., array_data: _Optional[_Union[ArrayArray, _Mapping]] = ..., json_data: _Optional[_Union[JSONArray, _Mapping]] = ...) -> None: ...

class VectorField(_message.Message):
    __slots__ = ["dim", "float_vector", "binary_vector"]
    DIM_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VECTOR_FIELD_NUMBER: _ClassVar[int]
    BINARY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    dim: int
    float_vector: FloatArray
    binary_vector: bytes
    def __init__(self, dim: _Optional[int] = ..., float_vector: _Optional[_Union[FloatArray, _Mapping]] = ..., binary_vector: _Optional[bytes] = ...) -> None: ...

class FieldData(_message.Message):
    __slots__ = ["type", "field_name", "scalars", "vectors", "field_id", "is_dynamic"]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    IS_DYNAMIC_FIELD_NUMBER: _ClassVar[int]
    type: DataType
    field_name: str
    scalars: ScalarField
    vectors: VectorField
    field_id: int
    is_dynamic: bool
    def __init__(self, type: _Optional[_Union[DataType, str]] = ..., field_name: _Optional[str] = ..., scalars: _Optional[_Union[ScalarField, _Mapping]] = ..., vectors: _Optional[_Union[VectorField, _Mapping]] = ..., field_id: _Optional[int] = ..., is_dynamic: bool = ...) -> None: ...

class IDs(_message.Message):
    __slots__ = ["int_id", "str_id"]
    INT_ID_FIELD_NUMBER: _ClassVar[int]
    STR_ID_FIELD_NUMBER: _ClassVar[int]
    int_id: LongArray
    str_id: StringArray
    def __init__(self, int_id: _Optional[_Union[LongArray, _Mapping]] = ..., str_id: _Optional[_Union[StringArray, _Mapping]] = ...) -> None: ...

class SearchResultData(_message.Message):
    __slots__ = ["num_queries", "top_k", "fields_data", "scores", "ids", "topks", "output_fields"]
    NUM_QUERIES_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    FIELDS_DATA_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    TOPKS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    num_queries: int
    top_k: int
    fields_data: _containers.RepeatedCompositeFieldContainer[FieldData]
    scores: _containers.RepeatedScalarFieldContainer[float]
    ids: IDs
    topks: _containers.RepeatedScalarFieldContainer[int]
    output_fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, num_queries: _Optional[int] = ..., top_k: _Optional[int] = ..., fields_data: _Optional[_Iterable[_Union[FieldData, _Mapping]]] = ..., scores: _Optional[_Iterable[float]] = ..., ids: _Optional[_Union[IDs, _Mapping]] = ..., topks: _Optional[_Iterable[int]] = ..., output_fields: _Optional[_Iterable[str]] = ...) -> None: ...
