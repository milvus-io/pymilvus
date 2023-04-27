from . import common_pb2 as _common_pb2
from . import schema_pb2 as _schema_pb2
from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

AddUserToRole: OperateUserRoleType
All: ShowType
DESCRIPTOR: _descriptor.FileDescriptor
DenyToRead: QuotaState
DenyToWrite: QuotaState
Grant: OperatePrivilegeType
InMemory: ShowType
MILVUS_EXT_OBJ_FIELD_NUMBER: _ClassVar[int]
ReadLimited: QuotaState
RemoveUserFromRole: OperateUserRoleType
Revoke: OperatePrivilegeType
Unknown: QuotaState
WriteLimited: QuotaState
milvus_ext_obj: _descriptor.FieldDescriptor

class AlterAliasRequest(_message.Message):
    __slots__ = ["alias", "base", "collection_name", "db_name"]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    alias: str
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class AlterCollectionRequest(_message.Message):
    __slots__ = ["base", "collectionID", "collection_name", "db_name", "properties"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionID: int
    collection_name: str
    db_name: str
    properties: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., collectionID: _Optional[int] = ..., properties: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class BoolResponse(_message.Message):
    __slots__ = ["status", "value"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    value: bool
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., value: bool = ...) -> None: ...

class CalcDistanceRequest(_message.Message):
    __slots__ = ["base", "op_left", "op_right", "params"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    OP_LEFT_FIELD_NUMBER: _ClassVar[int]
    OP_RIGHT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    op_left: VectorsArray
    op_right: VectorsArray
    params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., op_left: _Optional[_Union[VectorsArray, _Mapping]] = ..., op_right: _Optional[_Union[VectorsArray, _Mapping]] = ..., params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class CalcDistanceResults(_message.Message):
    __slots__ = ["float_dist", "int_dist", "status"]
    FLOAT_DIST_FIELD_NUMBER: _ClassVar[int]
    INT_DIST_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    float_dist: _schema_pb2.FloatArray
    int_dist: _schema_pb2.IntArray
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., int_dist: _Optional[_Union[_schema_pb2.IntArray, _Mapping]] = ..., float_dist: _Optional[_Union[_schema_pb2.FloatArray, _Mapping]] = ...) -> None: ...

class CheckHealthRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CheckHealthResponse(_message.Message):
    __slots__ = ["isHealthy", "quota_states", "reasons", "status"]
    ISHEALTHY_FIELD_NUMBER: _ClassVar[int]
    QUOTA_STATES_FIELD_NUMBER: _ClassVar[int]
    REASONS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    isHealthy: bool
    quota_states: _containers.RepeatedScalarFieldContainer[QuotaState]
    reasons: _containers.RepeatedScalarFieldContainer[str]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., isHealthy: bool = ..., reasons: _Optional[_Iterable[str]] = ..., quota_states: _Optional[_Iterable[_Union[QuotaState, str]]] = ...) -> None: ...

class CompactionMergeInfo(_message.Message):
    __slots__ = ["sources", "target"]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    sources: _containers.RepeatedScalarFieldContainer[int]
    target: int
    def __init__(self, sources: _Optional[_Iterable[int]] = ..., target: _Optional[int] = ...) -> None: ...

class ComponentInfo(_message.Message):
    __slots__ = ["extra_info", "nodeID", "role", "state_code"]
    EXTRA_INFO_FIELD_NUMBER: _ClassVar[int]
    NODEID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    STATE_CODE_FIELD_NUMBER: _ClassVar[int]
    extra_info: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    nodeID: int
    role: str
    state_code: _common_pb2.StateCode
    def __init__(self, nodeID: _Optional[int] = ..., role: _Optional[str] = ..., state_code: _Optional[_Union[_common_pb2.StateCode, str]] = ..., extra_info: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class ComponentStates(_message.Message):
    __slots__ = ["state", "status", "subcomponent_states"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SUBCOMPONENT_STATES_FIELD_NUMBER: _ClassVar[int]
    state: ComponentInfo
    status: _common_pb2.Status
    subcomponent_states: _containers.RepeatedCompositeFieldContainer[ComponentInfo]
    def __init__(self, state: _Optional[_Union[ComponentInfo, _Mapping]] = ..., subcomponent_states: _Optional[_Iterable[_Union[ComponentInfo, _Mapping]]] = ..., status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ...) -> None: ...

class CreateAliasRequest(_message.Message):
    __slots__ = ["alias", "base", "collection_name", "db_name"]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    alias: str
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class CreateCollectionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "consistency_level", "db_name", "properties", "schema", "shards_num"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    CONSISTENCY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    SHARDS_NUM_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    consistency_level: _common_pb2.ConsistencyLevel
    db_name: str
    properties: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    schema: bytes
    shards_num: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., schema: _Optional[bytes] = ..., shards_num: _Optional[int] = ..., consistency_level: _Optional[_Union[_common_pb2.ConsistencyLevel, str]] = ..., properties: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class CreateCredentialRequest(_message.Message):
    __slots__ = ["base", "created_utc_timestamps", "modified_utc_timestamps", "password", "username"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    CREATED_UTC_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_UTC_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    created_utc_timestamps: int
    modified_utc_timestamps: int
    password: str
    username: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., username: _Optional[str] = ..., password: _Optional[str] = ..., created_utc_timestamps: _Optional[int] = ..., modified_utc_timestamps: _Optional[int] = ...) -> None: ...

class CreateDatabaseRequest(_message.Message):
    __slots__ = ["base", "db_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ...) -> None: ...

class CreateIndexRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "extra_params", "field_name", "index_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    EXTRA_PARAMS_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    extra_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    field_name: str
    index_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., field_name: _Optional[str] = ..., extra_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., index_name: _Optional[str] = ...) -> None: ...

class CreatePartitionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ...) -> None: ...

class CreateResourceGroupRequest(_message.Message):
    __slots__ = ["base", "resource_group"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    resource_group: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., resource_group: _Optional[str] = ...) -> None: ...

class CreateRoleRequest(_message.Message):
    __slots__ = ["base", "entity"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    entity: RoleEntity
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., entity: _Optional[_Union[RoleEntity, _Mapping]] = ...) -> None: ...

class DeleteCredentialRequest(_message.Message):
    __slots__ = ["base", "username"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    username: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., username: _Optional[str] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "expr", "hash_keys", "partition_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    HASH_KEYS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    expr: str
    hash_keys: _containers.RepeatedScalarFieldContainer[int]
    partition_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ..., expr: _Optional[str] = ..., hash_keys: _Optional[_Iterable[int]] = ...) -> None: ...

class DescribeCollectionRequest(_message.Message):
    __slots__ = ["base", "collectionID", "collection_name", "db_name", "time_stamp"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionID: int
    collection_name: str
    db_name: str
    time_stamp: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., collectionID: _Optional[int] = ..., time_stamp: _Optional[int] = ...) -> None: ...

class DescribeCollectionResponse(_message.Message):
    __slots__ = ["aliases", "collectionID", "collection_name", "consistency_level", "created_timestamp", "created_utc_timestamp", "db_name", "physical_channel_names", "properties", "schema", "shards_num", "start_positions", "status", "virtual_channel_names"]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    CONSISTENCY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    CREATED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    CREATED_UTC_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PHYSICAL_CHANNEL_NAMES_FIELD_NUMBER: _ClassVar[int]
    PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    SHARDS_NUM_FIELD_NUMBER: _ClassVar[int]
    START_POSITIONS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VIRTUAL_CHANNEL_NAMES_FIELD_NUMBER: _ClassVar[int]
    aliases: _containers.RepeatedScalarFieldContainer[str]
    collectionID: int
    collection_name: str
    consistency_level: _common_pb2.ConsistencyLevel
    created_timestamp: int
    created_utc_timestamp: int
    db_name: str
    physical_channel_names: _containers.RepeatedScalarFieldContainer[str]
    properties: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    schema: _schema_pb2.CollectionSchema
    shards_num: int
    start_positions: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyDataPair]
    status: _common_pb2.Status
    virtual_channel_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., schema: _Optional[_Union[_schema_pb2.CollectionSchema, _Mapping]] = ..., collectionID: _Optional[int] = ..., virtual_channel_names: _Optional[_Iterable[str]] = ..., physical_channel_names: _Optional[_Iterable[str]] = ..., created_timestamp: _Optional[int] = ..., created_utc_timestamp: _Optional[int] = ..., shards_num: _Optional[int] = ..., aliases: _Optional[_Iterable[str]] = ..., start_positions: _Optional[_Iterable[_Union[_common_pb2.KeyDataPair, _Mapping]]] = ..., consistency_level: _Optional[_Union[_common_pb2.ConsistencyLevel, str]] = ..., collection_name: _Optional[str] = ..., properties: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., db_name: _Optional[str] = ...) -> None: ...

class DescribeIndexRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "field_name", "index_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    field_name: str
    index_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., field_name: _Optional[str] = ..., index_name: _Optional[str] = ...) -> None: ...

class DescribeIndexResponse(_message.Message):
    __slots__ = ["index_descriptions", "status"]
    INDEX_DESCRIPTIONS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    index_descriptions: _containers.RepeatedCompositeFieldContainer[IndexDescription]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., index_descriptions: _Optional[_Iterable[_Union[IndexDescription, _Mapping]]] = ...) -> None: ...

class DescribeResourceGroupRequest(_message.Message):
    __slots__ = ["base", "resource_group"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    resource_group: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., resource_group: _Optional[str] = ...) -> None: ...

class DescribeResourceGroupResponse(_message.Message):
    __slots__ = ["resource_group", "status"]
    RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    resource_group: ResourceGroup
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., resource_group: _Optional[_Union[ResourceGroup, _Mapping]] = ...) -> None: ...

class DescribeSegmentRequest(_message.Message):
    __slots__ = ["base", "collectionID", "segmentID"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    SEGMENTID_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionID: int
    segmentID: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., collectionID: _Optional[int] = ..., segmentID: _Optional[int] = ...) -> None: ...

class DescribeSegmentResponse(_message.Message):
    __slots__ = ["buildID", "enable_index", "fieldID", "indexID", "status"]
    BUILDID_FIELD_NUMBER: _ClassVar[int]
    ENABLE_INDEX_FIELD_NUMBER: _ClassVar[int]
    FIELDID_FIELD_NUMBER: _ClassVar[int]
    INDEXID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    buildID: int
    enable_index: bool
    fieldID: int
    indexID: int
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., indexID: _Optional[int] = ..., buildID: _Optional[int] = ..., enable_index: bool = ..., fieldID: _Optional[int] = ...) -> None: ...

class DropAliasRequest(_message.Message):
    __slots__ = ["alias", "base", "db_name"]
    ALIAS_FIELD_NUMBER: _ClassVar[int]
    BASE_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    alias: str
    base: _common_pb2.MsgBase
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., alias: _Optional[str] = ...) -> None: ...

class DropCollectionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class DropDatabaseRequest(_message.Message):
    __slots__ = ["base", "db_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ...) -> None: ...

class DropIndexRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "field_name", "index_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    field_name: str
    index_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., field_name: _Optional[str] = ..., index_name: _Optional[str] = ...) -> None: ...

class DropPartitionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ...) -> None: ...

class DropResourceGroupRequest(_message.Message):
    __slots__ = ["base", "resource_group"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    resource_group: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., resource_group: _Optional[str] = ...) -> None: ...

class DropRoleRequest(_message.Message):
    __slots__ = ["base", "role_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    ROLE_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    role_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., role_name: _Optional[str] = ...) -> None: ...

class DummyRequest(_message.Message):
    __slots__ = ["request_type"]
    REQUEST_TYPE_FIELD_NUMBER: _ClassVar[int]
    request_type: str
    def __init__(self, request_type: _Optional[str] = ...) -> None: ...

class DummyResponse(_message.Message):
    __slots__ = ["response"]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class FlushAllRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class FlushAllResponse(_message.Message):
    __slots__ = ["flush_all_ts", "status"]
    FLUSH_ALL_TS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    flush_all_ts: int
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., flush_all_ts: _Optional[int] = ...) -> None: ...

class FlushRequest(_message.Message):
    __slots__ = ["base", "collection_names", "db_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAMES_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_names: _containers.RepeatedScalarFieldContainer[str]
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_names: _Optional[_Iterable[str]] = ...) -> None: ...

class FlushResponse(_message.Message):
    __slots__ = ["coll_seal_times", "coll_segIDs", "db_name", "flush_coll_segIDs", "status"]
    class CollSealTimesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class CollSegIDsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _schema_pb2.LongArray
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_schema_pb2.LongArray, _Mapping]] = ...) -> None: ...
    class FlushCollSegIDsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _schema_pb2.LongArray
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_schema_pb2.LongArray, _Mapping]] = ...) -> None: ...
    COLL_SEAL_TIMES_FIELD_NUMBER: _ClassVar[int]
    COLL_SEGIDS_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FLUSH_COLL_SEGIDS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    coll_seal_times: _containers.ScalarMap[str, int]
    coll_segIDs: _containers.MessageMap[str, _schema_pb2.LongArray]
    db_name: str
    flush_coll_segIDs: _containers.MessageMap[str, _schema_pb2.LongArray]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., db_name: _Optional[str] = ..., coll_segIDs: _Optional[_Mapping[str, _schema_pb2.LongArray]] = ..., flush_coll_segIDs: _Optional[_Mapping[str, _schema_pb2.LongArray]] = ..., coll_seal_times: _Optional[_Mapping[str, int]] = ...) -> None: ...

class GetCollectionStatisticsRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class GetCollectionStatisticsResponse(_message.Message):
    __slots__ = ["stats", "status"]
    STATS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    stats: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., stats: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class GetCompactionPlansRequest(_message.Message):
    __slots__ = ["compactionID"]
    COMPACTIONID_FIELD_NUMBER: _ClassVar[int]
    compactionID: int
    def __init__(self, compactionID: _Optional[int] = ...) -> None: ...

class GetCompactionPlansResponse(_message.Message):
    __slots__ = ["mergeInfos", "state", "status"]
    MERGEINFOS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    mergeInfos: _containers.RepeatedCompositeFieldContainer[CompactionMergeInfo]
    state: _common_pb2.CompactionState
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., state: _Optional[_Union[_common_pb2.CompactionState, str]] = ..., mergeInfos: _Optional[_Iterable[_Union[CompactionMergeInfo, _Mapping]]] = ...) -> None: ...

class GetCompactionStateRequest(_message.Message):
    __slots__ = ["compactionID"]
    COMPACTIONID_FIELD_NUMBER: _ClassVar[int]
    compactionID: int
    def __init__(self, compactionID: _Optional[int] = ...) -> None: ...

class GetCompactionStateResponse(_message.Message):
    __slots__ = ["completedPlanNo", "executingPlanNo", "failedPlanNo", "state", "status", "timeoutPlanNo"]
    COMPLETEDPLANNO_FIELD_NUMBER: _ClassVar[int]
    EXECUTINGPLANNO_FIELD_NUMBER: _ClassVar[int]
    FAILEDPLANNO_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUTPLANNO_FIELD_NUMBER: _ClassVar[int]
    completedPlanNo: int
    executingPlanNo: int
    failedPlanNo: int
    state: _common_pb2.CompactionState
    status: _common_pb2.Status
    timeoutPlanNo: int
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., state: _Optional[_Union[_common_pb2.CompactionState, str]] = ..., executingPlanNo: _Optional[int] = ..., timeoutPlanNo: _Optional[int] = ..., completedPlanNo: _Optional[int] = ..., failedPlanNo: _Optional[int] = ...) -> None: ...

class GetComponentStatesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetFlushAllStateRequest(_message.Message):
    __slots__ = ["base", "flush_all_ts"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    FLUSH_ALL_TS_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    flush_all_ts: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., flush_all_ts: _Optional[int] = ...) -> None: ...

class GetFlushAllStateResponse(_message.Message):
    __slots__ = ["flushed", "status"]
    FLUSHED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    flushed: bool
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., flushed: bool = ...) -> None: ...

class GetFlushStateRequest(_message.Message):
    __slots__ = ["segmentIDs"]
    SEGMENTIDS_FIELD_NUMBER: _ClassVar[int]
    segmentIDs: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, segmentIDs: _Optional[_Iterable[int]] = ...) -> None: ...

class GetFlushStateResponse(_message.Message):
    __slots__ = ["flushed", "status"]
    FLUSHED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    flushed: bool
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., flushed: bool = ...) -> None: ...

class GetImportStateRequest(_message.Message):
    __slots__ = ["task"]
    TASK_FIELD_NUMBER: _ClassVar[int]
    task: int
    def __init__(self, task: _Optional[int] = ...) -> None: ...

class GetImportStateResponse(_message.Message):
    __slots__ = ["collection_id", "create_ts", "id", "id_list", "infos", "row_count", "segment_ids", "state", "status"]
    COLLECTION_ID_FIELD_NUMBER: _ClassVar[int]
    CREATE_TS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ID_LIST_FIELD_NUMBER: _ClassVar[int]
    INFOS_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_IDS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    collection_id: int
    create_ts: int
    id: int
    id_list: _containers.RepeatedScalarFieldContainer[int]
    infos: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    row_count: int
    segment_ids: _containers.RepeatedScalarFieldContainer[int]
    state: _common_pb2.ImportState
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., state: _Optional[_Union[_common_pb2.ImportState, str]] = ..., row_count: _Optional[int] = ..., id_list: _Optional[_Iterable[int]] = ..., infos: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., id: _Optional[int] = ..., collection_id: _Optional[int] = ..., segment_ids: _Optional[_Iterable[int]] = ..., create_ts: _Optional[int] = ...) -> None: ...

class GetIndexBuildProgressRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "field_name", "index_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    field_name: str
    index_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., field_name: _Optional[str] = ..., index_name: _Optional[str] = ...) -> None: ...

class GetIndexBuildProgressResponse(_message.Message):
    __slots__ = ["indexed_rows", "status", "total_rows"]
    INDEXED_ROWS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROWS_FIELD_NUMBER: _ClassVar[int]
    indexed_rows: int
    status: _common_pb2.Status
    total_rows: int
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., indexed_rows: _Optional[int] = ..., total_rows: _Optional[int] = ...) -> None: ...

class GetIndexStateRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "field_name", "index_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    field_name: str
    index_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., field_name: _Optional[str] = ..., index_name: _Optional[str] = ...) -> None: ...

class GetIndexStateResponse(_message.Message):
    __slots__ = ["fail_reason", "state", "status"]
    FAIL_REASON_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    fail_reason: str
    state: _common_pb2.IndexState
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., state: _Optional[_Union[_common_pb2.IndexState, str]] = ..., fail_reason: _Optional[str] = ...) -> None: ...

class GetLoadStateRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_names"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., collection_name: _Optional[str] = ..., partition_names: _Optional[_Iterable[str]] = ..., db_name: _Optional[str] = ...) -> None: ...

class GetLoadStateResponse(_message.Message):
    __slots__ = ["state", "status"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    state: _common_pb2.LoadState
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., state: _Optional[_Union[_common_pb2.LoadState, str]] = ...) -> None: ...

class GetLoadingProgressRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_names"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., collection_name: _Optional[str] = ..., partition_names: _Optional[_Iterable[str]] = ..., db_name: _Optional[str] = ...) -> None: ...

class GetLoadingProgressResponse(_message.Message):
    __slots__ = ["progress", "status"]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    progress: int
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., progress: _Optional[int] = ...) -> None: ...

class GetMetricsRequest(_message.Message):
    __slots__ = ["base", "request"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    request: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., request: _Optional[str] = ...) -> None: ...

class GetMetricsResponse(_message.Message):
    __slots__ = ["component_name", "response", "status"]
    COMPONENT_NAME_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    component_name: str
    response: str
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., response: _Optional[str] = ..., component_name: _Optional[str] = ...) -> None: ...

class GetPartitionStatisticsRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ...) -> None: ...

class GetPartitionStatisticsResponse(_message.Message):
    __slots__ = ["stats", "status"]
    STATS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    stats: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., stats: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class GetPersistentSegmentInfoRequest(_message.Message):
    __slots__ = ["base", "collectionName", "dbName"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONNAME_FIELD_NUMBER: _ClassVar[int]
    DBNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionName: str
    dbName: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., dbName: _Optional[str] = ..., collectionName: _Optional[str] = ...) -> None: ...

class GetPersistentSegmentInfoResponse(_message.Message):
    __slots__ = ["infos", "status"]
    INFOS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    infos: _containers.RepeatedCompositeFieldContainer[PersistentSegmentInfo]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., infos: _Optional[_Iterable[_Union[PersistentSegmentInfo, _Mapping]]] = ...) -> None: ...

class GetQuerySegmentInfoRequest(_message.Message):
    __slots__ = ["base", "collectionName", "dbName"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONNAME_FIELD_NUMBER: _ClassVar[int]
    DBNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionName: str
    dbName: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., dbName: _Optional[str] = ..., collectionName: _Optional[str] = ...) -> None: ...

class GetQuerySegmentInfoResponse(_message.Message):
    __slots__ = ["infos", "status"]
    INFOS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    infos: _containers.RepeatedCompositeFieldContainer[QuerySegmentInfo]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., infos: _Optional[_Iterable[_Union[QuerySegmentInfo, _Mapping]]] = ...) -> None: ...

class GetReplicasRequest(_message.Message):
    __slots__ = ["base", "collectionID", "collection_name", "db_name", "with_shard_nodes"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    WITH_SHARD_NODES_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionID: int
    collection_name: str
    db_name: str
    with_shard_nodes: bool
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., collectionID: _Optional[int] = ..., with_shard_nodes: bool = ..., collection_name: _Optional[str] = ..., db_name: _Optional[str] = ...) -> None: ...

class GetReplicasResponse(_message.Message):
    __slots__ = ["replicas", "status"]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    replicas: _containers.RepeatedCompositeFieldContainer[ReplicaInfo]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., replicas: _Optional[_Iterable[_Union[ReplicaInfo, _Mapping]]] = ...) -> None: ...

class GetStatisticsRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "guarantee_timestamp", "partition_names"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    GUARANTEE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    guarantee_timestamp: int
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_names: _Optional[_Iterable[str]] = ..., guarantee_timestamp: _Optional[int] = ...) -> None: ...

class GetStatisticsResponse(_message.Message):
    __slots__ = ["stats", "status"]
    STATS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    stats: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., stats: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class GetVersionRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetVersionResponse(_message.Message):
    __slots__ = ["status", "version"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    version: str
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., version: _Optional[str] = ...) -> None: ...

class GrantEntity(_message.Message):
    __slots__ = ["db_name", "grantor", "object", "object_name", "role"]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    GRANTOR_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    OBJECT_NAME_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    db_name: str
    grantor: GrantorEntity
    object: ObjectEntity
    object_name: str
    role: RoleEntity
    def __init__(self, role: _Optional[_Union[RoleEntity, _Mapping]] = ..., object: _Optional[_Union[ObjectEntity, _Mapping]] = ..., object_name: _Optional[str] = ..., grantor: _Optional[_Union[GrantorEntity, _Mapping]] = ..., db_name: _Optional[str] = ...) -> None: ...

class GrantPrivilegeEntity(_message.Message):
    __slots__ = ["entities"]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[GrantorEntity]
    def __init__(self, entities: _Optional[_Iterable[_Union[GrantorEntity, _Mapping]]] = ...) -> None: ...

class GrantorEntity(_message.Message):
    __slots__ = ["privilege", "user"]
    PRIVILEGE_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    privilege: PrivilegeEntity
    user: UserEntity
    def __init__(self, user: _Optional[_Union[UserEntity, _Mapping]] = ..., privilege: _Optional[_Union[PrivilegeEntity, _Mapping]] = ...) -> None: ...

class HasCollectionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "time_stamp"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    time_stamp: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., time_stamp: _Optional[int] = ...) -> None: ...

class HasPartitionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ...) -> None: ...

class Hits(_message.Message):
    __slots__ = ["IDs", "row_data", "scores"]
    IDS_FIELD_NUMBER: _ClassVar[int]
    IDs: _containers.RepeatedScalarFieldContainer[int]
    ROW_DATA_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    row_data: _containers.RepeatedScalarFieldContainer[bytes]
    scores: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, IDs: _Optional[_Iterable[int]] = ..., row_data: _Optional[_Iterable[bytes]] = ..., scores: _Optional[_Iterable[float]] = ...) -> None: ...

class ImportRequest(_message.Message):
    __slots__ = ["channel_names", "collection_name", "db_name", "files", "options", "partition_name", "row_based"]
    CHANNEL_NAMES_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FILES_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    ROW_BASED_FIELD_NUMBER: _ClassVar[int]
    channel_names: _containers.RepeatedScalarFieldContainer[str]
    collection_name: str
    db_name: str
    files: _containers.RepeatedScalarFieldContainer[str]
    options: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    partition_name: str
    row_based: bool
    def __init__(self, collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ..., channel_names: _Optional[_Iterable[str]] = ..., row_based: bool = ..., files: _Optional[_Iterable[str]] = ..., options: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., db_name: _Optional[str] = ...) -> None: ...

class ImportResponse(_message.Message):
    __slots__ = ["status", "tasks"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    tasks: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., tasks: _Optional[_Iterable[int]] = ...) -> None: ...

class IndexDescription(_message.Message):
    __slots__ = ["field_name", "indexID", "index_name", "index_state_fail_reason", "indexed_rows", "params", "state", "total_rows"]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEXED_ROWS_FIELD_NUMBER: _ClassVar[int]
    INDEXID_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_STATE_FAIL_REASON_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_ROWS_FIELD_NUMBER: _ClassVar[int]
    field_name: str
    indexID: int
    index_name: str
    index_state_fail_reason: str
    indexed_rows: int
    params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    state: _common_pb2.IndexState
    total_rows: int
    def __init__(self, index_name: _Optional[str] = ..., indexID: _Optional[int] = ..., params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., field_name: _Optional[str] = ..., indexed_rows: _Optional[int] = ..., total_rows: _Optional[int] = ..., state: _Optional[_Union[_common_pb2.IndexState, str]] = ..., index_state_fail_reason: _Optional[str] = ...) -> None: ...

class InsertRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "fields_data", "hash_keys", "num_rows", "partition_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELDS_DATA_FIELD_NUMBER: _ClassVar[int]
    HASH_KEYS_FIELD_NUMBER: _ClassVar[int]
    NUM_ROWS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    fields_data: _containers.RepeatedCompositeFieldContainer[_schema_pb2.FieldData]
    hash_keys: _containers.RepeatedScalarFieldContainer[int]
    num_rows: int
    partition_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_name: _Optional[str] = ..., fields_data: _Optional[_Iterable[_Union[_schema_pb2.FieldData, _Mapping]]] = ..., hash_keys: _Optional[_Iterable[int]] = ..., num_rows: _Optional[int] = ...) -> None: ...

class ListCredUsersRequest(_message.Message):
    __slots__ = ["base"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ...) -> None: ...

class ListCredUsersResponse(_message.Message):
    __slots__ = ["status", "usernames"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    USERNAMES_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    usernames: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., usernames: _Optional[_Iterable[str]] = ...) -> None: ...

class ListDatabasesRequest(_message.Message):
    __slots__ = ["base"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ...) -> None: ...

class ListDatabasesResponse(_message.Message):
    __slots__ = ["db_names", "status"]
    DB_NAMES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    db_names: _containers.RepeatedScalarFieldContainer[str]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., db_names: _Optional[_Iterable[str]] = ...) -> None: ...

class ListImportTasksRequest(_message.Message):
    __slots__ = ["collection_name", "db_name", "limit"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    db_name: str
    limit: int
    def __init__(self, collection_name: _Optional[str] = ..., limit: _Optional[int] = ..., db_name: _Optional[str] = ...) -> None: ...

class ListImportTasksResponse(_message.Message):
    __slots__ = ["status", "tasks"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    tasks: _containers.RepeatedCompositeFieldContainer[GetImportStateResponse]
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., tasks: _Optional[_Iterable[_Union[GetImportStateResponse, _Mapping]]] = ...) -> None: ...

class ListResourceGroupsRequest(_message.Message):
    __slots__ = ["base"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ...) -> None: ...

class ListResourceGroupsResponse(_message.Message):
    __slots__ = ["resource_groups", "status"]
    RESOURCE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    resource_groups: _containers.RepeatedScalarFieldContainer[str]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., resource_groups: _Optional[_Iterable[str]] = ...) -> None: ...

class LoadBalanceRequest(_message.Message):
    __slots__ = ["base", "collectionName", "db_name", "dst_nodeIDs", "sealed_segmentIDs", "src_nodeID"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONNAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    DST_NODEIDS_FIELD_NUMBER: _ClassVar[int]
    SEALED_SEGMENTIDS_FIELD_NUMBER: _ClassVar[int]
    SRC_NODEID_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionName: str
    db_name: str
    dst_nodeIDs: _containers.RepeatedScalarFieldContainer[int]
    sealed_segmentIDs: _containers.RepeatedScalarFieldContainer[int]
    src_nodeID: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., src_nodeID: _Optional[int] = ..., dst_nodeIDs: _Optional[_Iterable[int]] = ..., sealed_segmentIDs: _Optional[_Iterable[int]] = ..., collectionName: _Optional[str] = ..., db_name: _Optional[str] = ...) -> None: ...

class LoadCollectionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "refresh", "replica_number", "resource_groups"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    REFRESH_FIELD_NUMBER: _ClassVar[int]
    REPLICA_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    refresh: bool
    replica_number: int
    resource_groups: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., replica_number: _Optional[int] = ..., resource_groups: _Optional[_Iterable[str]] = ..., refresh: bool = ...) -> None: ...

class LoadPartitionsRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_names", "refresh", "replica_number", "resource_groups"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    REFRESH_FIELD_NUMBER: _ClassVar[int]
    REPLICA_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_GROUPS_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    refresh: bool
    replica_number: int
    resource_groups: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_names: _Optional[_Iterable[str]] = ..., replica_number: _Optional[int] = ..., resource_groups: _Optional[_Iterable[str]] = ..., refresh: bool = ...) -> None: ...

class ManualCompactionRequest(_message.Message):
    __slots__ = ["collectionID", "timetravel"]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    TIMETRAVEL_FIELD_NUMBER: _ClassVar[int]
    collectionID: int
    timetravel: int
    def __init__(self, collectionID: _Optional[int] = ..., timetravel: _Optional[int] = ...) -> None: ...

class ManualCompactionResponse(_message.Message):
    __slots__ = ["compactionID", "status"]
    COMPACTIONID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    compactionID: int
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., compactionID: _Optional[int] = ...) -> None: ...

class MilvusExt(_message.Message):
    __slots__ = ["version"]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    version: str
    def __init__(self, version: _Optional[str] = ...) -> None: ...

class MutationResult(_message.Message):
    __slots__ = ["IDs", "acknowledged", "delete_cnt", "err_index", "insert_cnt", "status", "succ_index", "timestamp", "upsert_cnt"]
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    DELETE_CNT_FIELD_NUMBER: _ClassVar[int]
    ERR_INDEX_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    IDs: _schema_pb2.IDs
    INSERT_CNT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SUCC_INDEX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    UPSERT_CNT_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    delete_cnt: int
    err_index: _containers.RepeatedScalarFieldContainer[int]
    insert_cnt: int
    status: _common_pb2.Status
    succ_index: _containers.RepeatedScalarFieldContainer[int]
    timestamp: int
    upsert_cnt: int
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., IDs: _Optional[_Union[_schema_pb2.IDs, _Mapping]] = ..., succ_index: _Optional[_Iterable[int]] = ..., err_index: _Optional[_Iterable[int]] = ..., acknowledged: bool = ..., insert_cnt: _Optional[int] = ..., delete_cnt: _Optional[int] = ..., upsert_cnt: _Optional[int] = ..., timestamp: _Optional[int] = ...) -> None: ...

class ObjectEntity(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class OperatePrivilegeRequest(_message.Message):
    __slots__ = ["base", "entity", "type"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    entity: GrantEntity
    type: OperatePrivilegeType
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., entity: _Optional[_Union[GrantEntity, _Mapping]] = ..., type: _Optional[_Union[OperatePrivilegeType, str]] = ...) -> None: ...

class OperateUserRoleRequest(_message.Message):
    __slots__ = ["base", "role_name", "type", "username"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    ROLE_NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    role_name: str
    type: OperateUserRoleType
    username: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., username: _Optional[str] = ..., role_name: _Optional[str] = ..., type: _Optional[_Union[OperateUserRoleType, str]] = ...) -> None: ...

class PersistentSegmentInfo(_message.Message):
    __slots__ = ["collectionID", "num_rows", "partitionID", "segmentID", "state"]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    NUM_ROWS_FIELD_NUMBER: _ClassVar[int]
    PARTITIONID_FIELD_NUMBER: _ClassVar[int]
    SEGMENTID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    collectionID: int
    num_rows: int
    partitionID: int
    segmentID: int
    state: _common_pb2.SegmentState
    def __init__(self, segmentID: _Optional[int] = ..., collectionID: _Optional[int] = ..., partitionID: _Optional[int] = ..., num_rows: _Optional[int] = ..., state: _Optional[_Union[_common_pb2.SegmentState, str]] = ...) -> None: ...

class PrivilegeEntity(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class QueryRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "expr", "guarantee_timestamp", "output_fields", "partition_names", "query_params", "travel_timestamp"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    EXPR_FIELD_NUMBER: _ClassVar[int]
    GUARANTEE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    QUERY_PARAMS_FIELD_NUMBER: _ClassVar[int]
    TRAVEL_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    expr: str
    guarantee_timestamp: int
    output_fields: _containers.RepeatedScalarFieldContainer[str]
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    query_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    travel_timestamp: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., expr: _Optional[str] = ..., output_fields: _Optional[_Iterable[str]] = ..., partition_names: _Optional[_Iterable[str]] = ..., travel_timestamp: _Optional[int] = ..., guarantee_timestamp: _Optional[int] = ..., query_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ...) -> None: ...

class QueryResults(_message.Message):
    __slots__ = ["collection_name", "fields_data", "status"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELDS_DATA_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    fields_data: _containers.RepeatedCompositeFieldContainer[_schema_pb2.FieldData]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., fields_data: _Optional[_Iterable[_Union[_schema_pb2.FieldData, _Mapping]]] = ..., collection_name: _Optional[str] = ...) -> None: ...

class QuerySegmentInfo(_message.Message):
    __slots__ = ["collectionID", "indexID", "index_name", "mem_size", "nodeID", "nodeIds", "num_rows", "partitionID", "segmentID", "state"]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    INDEXID_FIELD_NUMBER: _ClassVar[int]
    INDEX_NAME_FIELD_NUMBER: _ClassVar[int]
    MEM_SIZE_FIELD_NUMBER: _ClassVar[int]
    NODEIDS_FIELD_NUMBER: _ClassVar[int]
    NODEID_FIELD_NUMBER: _ClassVar[int]
    NUM_ROWS_FIELD_NUMBER: _ClassVar[int]
    PARTITIONID_FIELD_NUMBER: _ClassVar[int]
    SEGMENTID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    collectionID: int
    indexID: int
    index_name: str
    mem_size: int
    nodeID: int
    nodeIds: _containers.RepeatedScalarFieldContainer[int]
    num_rows: int
    partitionID: int
    segmentID: int
    state: _common_pb2.SegmentState
    def __init__(self, segmentID: _Optional[int] = ..., collectionID: _Optional[int] = ..., partitionID: _Optional[int] = ..., mem_size: _Optional[int] = ..., num_rows: _Optional[int] = ..., index_name: _Optional[str] = ..., indexID: _Optional[int] = ..., nodeID: _Optional[int] = ..., state: _Optional[_Union[_common_pb2.SegmentState, str]] = ..., nodeIds: _Optional[_Iterable[int]] = ...) -> None: ...

class RegisterLinkRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class RegisterLinkResponse(_message.Message):
    __slots__ = ["address", "status"]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    address: _common_pb2.Address
    status: _common_pb2.Status
    def __init__(self, address: _Optional[_Union[_common_pb2.Address, _Mapping]] = ..., status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ...) -> None: ...

class ReleaseCollectionRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class ReleasePartitionsRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "partition_names"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_names: _Optional[_Iterable[str]] = ...) -> None: ...

class RenameCollectionRequest(_message.Message):
    __slots__ = ["base", "db_name", "newName", "oldName"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    NEWNAME_FIELD_NUMBER: _ClassVar[int]
    OLDNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    db_name: str
    newName: str
    oldName: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., oldName: _Optional[str] = ..., newName: _Optional[str] = ...) -> None: ...

class ReplicaInfo(_message.Message):
    __slots__ = ["collectionID", "node_ids", "num_outbound_node", "partition_ids", "replicaID", "resource_group_name", "shard_replicas"]
    class NumOutboundNodeEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTBOUND_NODE_FIELD_NUMBER: _ClassVar[int]
    PARTITION_IDS_FIELD_NUMBER: _ClassVar[int]
    REPLICAID_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_GROUP_NAME_FIELD_NUMBER: _ClassVar[int]
    SHARD_REPLICAS_FIELD_NUMBER: _ClassVar[int]
    collectionID: int
    node_ids: _containers.RepeatedScalarFieldContainer[int]
    num_outbound_node: _containers.ScalarMap[str, int]
    partition_ids: _containers.RepeatedScalarFieldContainer[int]
    replicaID: int
    resource_group_name: str
    shard_replicas: _containers.RepeatedCompositeFieldContainer[ShardReplica]
    def __init__(self, replicaID: _Optional[int] = ..., collectionID: _Optional[int] = ..., partition_ids: _Optional[_Iterable[int]] = ..., shard_replicas: _Optional[_Iterable[_Union[ShardReplica, _Mapping]]] = ..., node_ids: _Optional[_Iterable[int]] = ..., resource_group_name: _Optional[str] = ..., num_outbound_node: _Optional[_Mapping[str, int]] = ...) -> None: ...

class ResourceGroup(_message.Message):
    __slots__ = ["capacity", "name", "num_available_node", "num_incoming_node", "num_loaded_replica", "num_outgoing_node"]
    class NumIncomingNodeEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class NumLoadedReplicaEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    class NumOutgoingNodeEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    CAPACITY_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_AVAILABLE_NODE_FIELD_NUMBER: _ClassVar[int]
    NUM_INCOMING_NODE_FIELD_NUMBER: _ClassVar[int]
    NUM_LOADED_REPLICA_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTGOING_NODE_FIELD_NUMBER: _ClassVar[int]
    capacity: int
    name: str
    num_available_node: int
    num_incoming_node: _containers.ScalarMap[str, int]
    num_loaded_replica: _containers.ScalarMap[str, int]
    num_outgoing_node: _containers.ScalarMap[str, int]
    def __init__(self, name: _Optional[str] = ..., capacity: _Optional[int] = ..., num_available_node: _Optional[int] = ..., num_loaded_replica: _Optional[_Mapping[str, int]] = ..., num_outgoing_node: _Optional[_Mapping[str, int]] = ..., num_incoming_node: _Optional[_Mapping[str, int]] = ...) -> None: ...

class RoleEntity(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class RoleResult(_message.Message):
    __slots__ = ["role", "users"]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    USERS_FIELD_NUMBER: _ClassVar[int]
    role: RoleEntity
    users: _containers.RepeatedCompositeFieldContainer[UserEntity]
    def __init__(self, role: _Optional[_Union[RoleEntity, _Mapping]] = ..., users: _Optional[_Iterable[_Union[UserEntity, _Mapping]]] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "dsl", "dsl_type", "guarantee_timestamp", "nq", "output_fields", "partition_names", "placeholder_group", "search_params", "travel_timestamp"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    DSL_FIELD_NUMBER: _ClassVar[int]
    DSL_TYPE_FIELD_NUMBER: _ClassVar[int]
    GUARANTEE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    NQ_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    PLACEHOLDER_GROUP_FIELD_NUMBER: _ClassVar[int]
    SEARCH_PARAMS_FIELD_NUMBER: _ClassVar[int]
    TRAVEL_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    dsl: str
    dsl_type: _common_pb2.DslType
    guarantee_timestamp: int
    nq: int
    output_fields: _containers.RepeatedScalarFieldContainer[str]
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    placeholder_group: bytes
    search_params: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValuePair]
    travel_timestamp: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., partition_names: _Optional[_Iterable[str]] = ..., dsl: _Optional[str] = ..., placeholder_group: _Optional[bytes] = ..., dsl_type: _Optional[_Union[_common_pb2.DslType, str]] = ..., output_fields: _Optional[_Iterable[str]] = ..., search_params: _Optional[_Iterable[_Union[_common_pb2.KeyValuePair, _Mapping]]] = ..., travel_timestamp: _Optional[int] = ..., guarantee_timestamp: _Optional[int] = ..., nq: _Optional[int] = ...) -> None: ...

class SearchResults(_message.Message):
    __slots__ = ["collection_name", "results", "status"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    results: _schema_pb2.SearchResultData
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., results: _Optional[_Union[_schema_pb2.SearchResultData, _Mapping]] = ..., collection_name: _Optional[str] = ...) -> None: ...

class SelectGrantRequest(_message.Message):
    __slots__ = ["base", "entity"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    entity: GrantEntity
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., entity: _Optional[_Union[GrantEntity, _Mapping]] = ...) -> None: ...

class SelectGrantResponse(_message.Message):
    __slots__ = ["entities", "status"]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[GrantEntity]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., entities: _Optional[_Iterable[_Union[GrantEntity, _Mapping]]] = ...) -> None: ...

class SelectRoleRequest(_message.Message):
    __slots__ = ["base", "include_user_info", "role"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_USER_INFO_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    include_user_info: bool
    role: RoleEntity
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., role: _Optional[_Union[RoleEntity, _Mapping]] = ..., include_user_info: bool = ...) -> None: ...

class SelectRoleResponse(_message.Message):
    __slots__ = ["results", "status"]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[RoleResult]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., results: _Optional[_Iterable[_Union[RoleResult, _Mapping]]] = ...) -> None: ...

class SelectUserRequest(_message.Message):
    __slots__ = ["base", "include_role_info", "user"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_ROLE_INFO_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    include_role_info: bool
    user: UserEntity
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., user: _Optional[_Union[UserEntity, _Mapping]] = ..., include_role_info: bool = ...) -> None: ...

class SelectUserResponse(_message.Message):
    __slots__ = ["results", "status"]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[UserResult]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., results: _Optional[_Iterable[_Union[UserResult, _Mapping]]] = ...) -> None: ...

class ShardReplica(_message.Message):
    __slots__ = ["dm_channel_name", "leaderID", "leader_addr", "node_ids"]
    DM_CHANNEL_NAME_FIELD_NUMBER: _ClassVar[int]
    LEADERID_FIELD_NUMBER: _ClassVar[int]
    LEADER_ADDR_FIELD_NUMBER: _ClassVar[int]
    NODE_IDS_FIELD_NUMBER: _ClassVar[int]
    dm_channel_name: str
    leaderID: int
    leader_addr: str
    node_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, leaderID: _Optional[int] = ..., leader_addr: _Optional[str] = ..., dm_channel_name: _Optional[str] = ..., node_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class ShowCollectionsRequest(_message.Message):
    __slots__ = ["base", "collection_names", "db_name", "time_stamp", "type"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAMES_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_names: _containers.RepeatedScalarFieldContainer[str]
    db_name: str
    time_stamp: int
    type: ShowType
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., time_stamp: _Optional[int] = ..., type: _Optional[_Union[ShowType, str]] = ..., collection_names: _Optional[_Iterable[str]] = ...) -> None: ...

class ShowCollectionsResponse(_message.Message):
    __slots__ = ["collection_ids", "collection_names", "created_timestamps", "created_utc_timestamps", "inMemory_percentages", "query_service_available", "status"]
    COLLECTION_IDS_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAMES_FIELD_NUMBER: _ClassVar[int]
    CREATED_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    CREATED_UTC_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    INMEMORY_PERCENTAGES_FIELD_NUMBER: _ClassVar[int]
    QUERY_SERVICE_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    collection_ids: _containers.RepeatedScalarFieldContainer[int]
    collection_names: _containers.RepeatedScalarFieldContainer[str]
    created_timestamps: _containers.RepeatedScalarFieldContainer[int]
    created_utc_timestamps: _containers.RepeatedScalarFieldContainer[int]
    inMemory_percentages: _containers.RepeatedScalarFieldContainer[int]
    query_service_available: _containers.RepeatedScalarFieldContainer[bool]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., collection_names: _Optional[_Iterable[str]] = ..., collection_ids: _Optional[_Iterable[int]] = ..., created_timestamps: _Optional[_Iterable[int]] = ..., created_utc_timestamps: _Optional[_Iterable[int]] = ..., inMemory_percentages: _Optional[_Iterable[int]] = ..., query_service_available: _Optional[_Iterable[bool]] = ...) -> None: ...

class ShowPartitionsRequest(_message.Message):
    __slots__ = ["base", "collectionID", "collection_name", "db_name", "partition_names", "type"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionID: int
    collection_name: str
    db_name: str
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    type: ShowType
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., db_name: _Optional[str] = ..., collection_name: _Optional[str] = ..., collectionID: _Optional[int] = ..., partition_names: _Optional[_Iterable[str]] = ..., type: _Optional[_Union[ShowType, str]] = ...) -> None: ...

class ShowPartitionsResponse(_message.Message):
    __slots__ = ["created_timestamps", "created_utc_timestamps", "inMemory_percentages", "partitionIDs", "partition_names", "status"]
    CREATED_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    CREATED_UTC_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    INMEMORY_PERCENTAGES_FIELD_NUMBER: _ClassVar[int]
    PARTITIONIDS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    created_timestamps: _containers.RepeatedScalarFieldContainer[int]
    created_utc_timestamps: _containers.RepeatedScalarFieldContainer[int]
    inMemory_percentages: _containers.RepeatedScalarFieldContainer[int]
    partitionIDs: _containers.RepeatedScalarFieldContainer[int]
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., partition_names: _Optional[_Iterable[str]] = ..., partitionIDs: _Optional[_Iterable[int]] = ..., created_timestamps: _Optional[_Iterable[int]] = ..., created_utc_timestamps: _Optional[_Iterable[int]] = ..., inMemory_percentages: _Optional[_Iterable[int]] = ...) -> None: ...

class ShowSegmentsRequest(_message.Message):
    __slots__ = ["base", "collectionID", "partitionID"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTIONID_FIELD_NUMBER: _ClassVar[int]
    PARTITIONID_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collectionID: int
    partitionID: int
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., collectionID: _Optional[int] = ..., partitionID: _Optional[int] = ...) -> None: ...

class ShowSegmentsResponse(_message.Message):
    __slots__ = ["segmentIDs", "status"]
    SEGMENTIDS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    segmentIDs: _containers.RepeatedScalarFieldContainer[int]
    status: _common_pb2.Status
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., segmentIDs: _Optional[_Iterable[int]] = ...) -> None: ...

class StringResponse(_message.Message):
    __slots__ = ["status", "value"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    value: str
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., value: _Optional[str] = ...) -> None: ...

class TransferNodeRequest(_message.Message):
    __slots__ = ["base", "num_node", "source_resource_group", "target_resource_group"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    NUM_NODE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    TARGET_RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    num_node: int
    source_resource_group: str
    target_resource_group: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., source_resource_group: _Optional[str] = ..., target_resource_group: _Optional[str] = ..., num_node: _Optional[int] = ...) -> None: ...

class TransferReplicaRequest(_message.Message):
    __slots__ = ["base", "collection_name", "db_name", "num_replica", "source_resource_group", "target_resource_group"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DB_NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_REPLICA_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    TARGET_RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    collection_name: str
    db_name: str
    num_replica: int
    source_resource_group: str
    target_resource_group: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., source_resource_group: _Optional[str] = ..., target_resource_group: _Optional[str] = ..., collection_name: _Optional[str] = ..., num_replica: _Optional[int] = ..., db_name: _Optional[str] = ...) -> None: ...

class UpdateCredentialRequest(_message.Message):
    __slots__ = ["base", "created_utc_timestamps", "modified_utc_timestamps", "newPassword", "oldPassword", "username"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    CREATED_UTC_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_UTC_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    NEWPASSWORD_FIELD_NUMBER: _ClassVar[int]
    OLDPASSWORD_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    base: _common_pb2.MsgBase
    created_utc_timestamps: int
    modified_utc_timestamps: int
    newPassword: str
    oldPassword: str
    username: str
    def __init__(self, base: _Optional[_Union[_common_pb2.MsgBase, _Mapping]] = ..., username: _Optional[str] = ..., oldPassword: _Optional[str] = ..., newPassword: _Optional[str] = ..., created_utc_timestamps: _Optional[int] = ..., modified_utc_timestamps: _Optional[int] = ...) -> None: ...

class UserEntity(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class UserResult(_message.Message):
    __slots__ = ["roles", "user"]
    ROLES_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    roles: _containers.RepeatedCompositeFieldContainer[RoleEntity]
    user: UserEntity
    def __init__(self, user: _Optional[_Union[UserEntity, _Mapping]] = ..., roles: _Optional[_Iterable[_Union[RoleEntity, _Mapping]]] = ...) -> None: ...

class VectorIDs(_message.Message):
    __slots__ = ["collection_name", "field_name", "id_array", "partition_names"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    ID_ARRAY_FIELD_NUMBER: _ClassVar[int]
    PARTITION_NAMES_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    field_name: str
    id_array: _schema_pb2.IDs
    partition_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, collection_name: _Optional[str] = ..., field_name: _Optional[str] = ..., id_array: _Optional[_Union[_schema_pb2.IDs, _Mapping]] = ..., partition_names: _Optional[_Iterable[str]] = ...) -> None: ...

class VectorsArray(_message.Message):
    __slots__ = ["data_array", "id_array"]
    DATA_ARRAY_FIELD_NUMBER: _ClassVar[int]
    ID_ARRAY_FIELD_NUMBER: _ClassVar[int]
    data_array: _schema_pb2.VectorField
    id_array: VectorIDs
    def __init__(self, id_array: _Optional[_Union[VectorIDs, _Mapping]] = ..., data_array: _Optional[_Union[_schema_pb2.VectorField, _Mapping]] = ...) -> None: ...

class ShowType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class OperateUserRoleType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class OperatePrivilegeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class QuotaState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
