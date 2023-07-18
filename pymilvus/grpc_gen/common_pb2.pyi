from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Success: _ClassVar[ErrorCode]
    UnexpectedError: _ClassVar[ErrorCode]
    ConnectFailed: _ClassVar[ErrorCode]
    PermissionDenied: _ClassVar[ErrorCode]
    CollectionNotExists: _ClassVar[ErrorCode]
    IllegalArgument: _ClassVar[ErrorCode]
    IllegalDimension: _ClassVar[ErrorCode]
    IllegalIndexType: _ClassVar[ErrorCode]
    IllegalCollectionName: _ClassVar[ErrorCode]
    IllegalTOPK: _ClassVar[ErrorCode]
    IllegalRowRecord: _ClassVar[ErrorCode]
    IllegalVectorID: _ClassVar[ErrorCode]
    IllegalSearchResult: _ClassVar[ErrorCode]
    FileNotFound: _ClassVar[ErrorCode]
    MetaFailed: _ClassVar[ErrorCode]
    CacheFailed: _ClassVar[ErrorCode]
    CannotCreateFolder: _ClassVar[ErrorCode]
    CannotCreateFile: _ClassVar[ErrorCode]
    CannotDeleteFolder: _ClassVar[ErrorCode]
    CannotDeleteFile: _ClassVar[ErrorCode]
    BuildIndexError: _ClassVar[ErrorCode]
    IllegalNLIST: _ClassVar[ErrorCode]
    IllegalMetricType: _ClassVar[ErrorCode]
    OutOfMemory: _ClassVar[ErrorCode]
    IndexNotExist: _ClassVar[ErrorCode]
    EmptyCollection: _ClassVar[ErrorCode]
    UpdateImportTaskFailure: _ClassVar[ErrorCode]
    CollectionNameNotFound: _ClassVar[ErrorCode]
    CreateCredentialFailure: _ClassVar[ErrorCode]
    UpdateCredentialFailure: _ClassVar[ErrorCode]
    DeleteCredentialFailure: _ClassVar[ErrorCode]
    GetCredentialFailure: _ClassVar[ErrorCode]
    ListCredUsersFailure: _ClassVar[ErrorCode]
    GetUserFailure: _ClassVar[ErrorCode]
    CreateRoleFailure: _ClassVar[ErrorCode]
    DropRoleFailure: _ClassVar[ErrorCode]
    OperateUserRoleFailure: _ClassVar[ErrorCode]
    SelectRoleFailure: _ClassVar[ErrorCode]
    SelectUserFailure: _ClassVar[ErrorCode]
    SelectResourceFailure: _ClassVar[ErrorCode]
    OperatePrivilegeFailure: _ClassVar[ErrorCode]
    SelectGrantFailure: _ClassVar[ErrorCode]
    RefreshPolicyInfoCacheFailure: _ClassVar[ErrorCode]
    ListPolicyFailure: _ClassVar[ErrorCode]
    NotShardLeader: _ClassVar[ErrorCode]
    NoReplicaAvailable: _ClassVar[ErrorCode]
    SegmentNotFound: _ClassVar[ErrorCode]
    ForceDeny: _ClassVar[ErrorCode]
    RateLimit: _ClassVar[ErrorCode]
    NodeIDNotMatch: _ClassVar[ErrorCode]
    UpsertAutoIDTrue: _ClassVar[ErrorCode]
    InsufficientMemoryToLoad: _ClassVar[ErrorCode]
    MemoryQuotaExhausted: _ClassVar[ErrorCode]
    DiskQuotaExhausted: _ClassVar[ErrorCode]
    TimeTickLongDelay: _ClassVar[ErrorCode]
    NotReadyServe: _ClassVar[ErrorCode]
    NotReadyCoordActivating: _ClassVar[ErrorCode]
    NotFoundTSafer: _ClassVar[ErrorCode]
    DataCoordNA: _ClassVar[ErrorCode]
    DDRequestRace: _ClassVar[ErrorCode]

class IndexState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    IndexStateNone: _ClassVar[IndexState]
    Unissued: _ClassVar[IndexState]
    InProgress: _ClassVar[IndexState]
    Finished: _ClassVar[IndexState]
    Failed: _ClassVar[IndexState]
    Retry: _ClassVar[IndexState]

class SegmentState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    SegmentStateNone: _ClassVar[SegmentState]
    NotExist: _ClassVar[SegmentState]
    Growing: _ClassVar[SegmentState]
    Sealed: _ClassVar[SegmentState]
    Flushed: _ClassVar[SegmentState]
    Flushing: _ClassVar[SegmentState]
    Dropped: _ClassVar[SegmentState]
    Importing: _ClassVar[SegmentState]

class PlaceholderType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    None: _ClassVar[PlaceholderType]
    BinaryVector: _ClassVar[PlaceholderType]
    FloatVector: _ClassVar[PlaceholderType]

class MsgType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Undefined: _ClassVar[MsgType]
    CreateCollection: _ClassVar[MsgType]
    DropCollection: _ClassVar[MsgType]
    HasCollection: _ClassVar[MsgType]
    DescribeCollection: _ClassVar[MsgType]
    ShowCollections: _ClassVar[MsgType]
    GetSystemConfigs: _ClassVar[MsgType]
    LoadCollection: _ClassVar[MsgType]
    ReleaseCollection: _ClassVar[MsgType]
    CreateAlias: _ClassVar[MsgType]
    DropAlias: _ClassVar[MsgType]
    AlterAlias: _ClassVar[MsgType]
    AlterCollection: _ClassVar[MsgType]
    RenameCollection: _ClassVar[MsgType]
    CreatePartition: _ClassVar[MsgType]
    DropPartition: _ClassVar[MsgType]
    HasPartition: _ClassVar[MsgType]
    DescribePartition: _ClassVar[MsgType]
    ShowPartitions: _ClassVar[MsgType]
    LoadPartitions: _ClassVar[MsgType]
    ReleasePartitions: _ClassVar[MsgType]
    ShowSegments: _ClassVar[MsgType]
    DescribeSegment: _ClassVar[MsgType]
    LoadSegments: _ClassVar[MsgType]
    ReleaseSegments: _ClassVar[MsgType]
    HandoffSegments: _ClassVar[MsgType]
    LoadBalanceSegments: _ClassVar[MsgType]
    DescribeSegments: _ClassVar[MsgType]
    CreateIndex: _ClassVar[MsgType]
    DescribeIndex: _ClassVar[MsgType]
    DropIndex: _ClassVar[MsgType]
    GetIndexStatistics: _ClassVar[MsgType]
    Insert: _ClassVar[MsgType]
    Delete: _ClassVar[MsgType]
    Flush: _ClassVar[MsgType]
    ResendSegmentStats: _ClassVar[MsgType]
    Search: _ClassVar[MsgType]
    SearchResult: _ClassVar[MsgType]
    GetIndexState: _ClassVar[MsgType]
    GetIndexBuildProgress: _ClassVar[MsgType]
    GetCollectionStatistics: _ClassVar[MsgType]
    GetPartitionStatistics: _ClassVar[MsgType]
    Retrieve: _ClassVar[MsgType]
    RetrieveResult: _ClassVar[MsgType]
    WatchDmChannels: _ClassVar[MsgType]
    RemoveDmChannels: _ClassVar[MsgType]
    WatchQueryChannels: _ClassVar[MsgType]
    RemoveQueryChannels: _ClassVar[MsgType]
    SealedSegmentsChangeInfo: _ClassVar[MsgType]
    WatchDeltaChannels: _ClassVar[MsgType]
    GetShardLeaders: _ClassVar[MsgType]
    GetReplicas: _ClassVar[MsgType]
    UnsubDmChannel: _ClassVar[MsgType]
    GetDistribution: _ClassVar[MsgType]
    SyncDistribution: _ClassVar[MsgType]
    SegmentInfo: _ClassVar[MsgType]
    SystemInfo: _ClassVar[MsgType]
    GetRecoveryInfo: _ClassVar[MsgType]
    GetSegmentState: _ClassVar[MsgType]
    TimeTick: _ClassVar[MsgType]
    QueryNodeStats: _ClassVar[MsgType]
    LoadIndex: _ClassVar[MsgType]
    RequestID: _ClassVar[MsgType]
    RequestTSO: _ClassVar[MsgType]
    AllocateSegment: _ClassVar[MsgType]
    SegmentStatistics: _ClassVar[MsgType]
    SegmentFlushDone: _ClassVar[MsgType]
    DataNodeTt: _ClassVar[MsgType]
    Connect: _ClassVar[MsgType]
    ListClientInfos: _ClassVar[MsgType]
    AllocTimestamp: _ClassVar[MsgType]
    CreateCredential: _ClassVar[MsgType]
    GetCredential: _ClassVar[MsgType]
    DeleteCredential: _ClassVar[MsgType]
    UpdateCredential: _ClassVar[MsgType]
    ListCredUsernames: _ClassVar[MsgType]
    CreateRole: _ClassVar[MsgType]
    DropRole: _ClassVar[MsgType]
    OperateUserRole: _ClassVar[MsgType]
    SelectRole: _ClassVar[MsgType]
    SelectUser: _ClassVar[MsgType]
    SelectResource: _ClassVar[MsgType]
    OperatePrivilege: _ClassVar[MsgType]
    SelectGrant: _ClassVar[MsgType]
    RefreshPolicyInfoCache: _ClassVar[MsgType]
    ListPolicy: _ClassVar[MsgType]
    CreateResourceGroup: _ClassVar[MsgType]
    DropResourceGroup: _ClassVar[MsgType]
    ListResourceGroups: _ClassVar[MsgType]
    DescribeResourceGroup: _ClassVar[MsgType]
    TransferNode: _ClassVar[MsgType]
    TransferReplica: _ClassVar[MsgType]
    CreateDatabase: _ClassVar[MsgType]
    DropDatabase: _ClassVar[MsgType]
    ListDatabases: _ClassVar[MsgType]

class DslType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Dsl: _ClassVar[DslType]
    BoolExprV1: _ClassVar[DslType]

class CompactionState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    UndefiedState: _ClassVar[CompactionState]
    Executing: _ClassVar[CompactionState]
    Completed: _ClassVar[CompactionState]

class ConsistencyLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Strong: _ClassVar[ConsistencyLevel]
    Session: _ClassVar[ConsistencyLevel]
    Bounded: _ClassVar[ConsistencyLevel]
    Eventually: _ClassVar[ConsistencyLevel]
    Customized: _ClassVar[ConsistencyLevel]

class ImportState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    ImportPending: _ClassVar[ImportState]
    ImportFailed: _ClassVar[ImportState]
    ImportStarted: _ClassVar[ImportState]
    ImportPersisted: _ClassVar[ImportState]
    ImportCompleted: _ClassVar[ImportState]
    ImportFailedAndCleaned: _ClassVar[ImportState]

class ObjectType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Collection: _ClassVar[ObjectType]
    Global: _ClassVar[ObjectType]
    User: _ClassVar[ObjectType]

class ObjectPrivilege(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    PrivilegeAll: _ClassVar[ObjectPrivilege]
    PrivilegeCreateCollection: _ClassVar[ObjectPrivilege]
    PrivilegeDropCollection: _ClassVar[ObjectPrivilege]
    PrivilegeDescribeCollection: _ClassVar[ObjectPrivilege]
    PrivilegeShowCollections: _ClassVar[ObjectPrivilege]
    PrivilegeLoad: _ClassVar[ObjectPrivilege]
    PrivilegeRelease: _ClassVar[ObjectPrivilege]
    PrivilegeCompaction: _ClassVar[ObjectPrivilege]
    PrivilegeInsert: _ClassVar[ObjectPrivilege]
    PrivilegeDelete: _ClassVar[ObjectPrivilege]
    PrivilegeGetStatistics: _ClassVar[ObjectPrivilege]
    PrivilegeCreateIndex: _ClassVar[ObjectPrivilege]
    PrivilegeIndexDetail: _ClassVar[ObjectPrivilege]
    PrivilegeDropIndex: _ClassVar[ObjectPrivilege]
    PrivilegeSearch: _ClassVar[ObjectPrivilege]
    PrivilegeFlush: _ClassVar[ObjectPrivilege]
    PrivilegeQuery: _ClassVar[ObjectPrivilege]
    PrivilegeLoadBalance: _ClassVar[ObjectPrivilege]
    PrivilegeImport: _ClassVar[ObjectPrivilege]
    PrivilegeCreateOwnership: _ClassVar[ObjectPrivilege]
    PrivilegeUpdateUser: _ClassVar[ObjectPrivilege]
    PrivilegeDropOwnership: _ClassVar[ObjectPrivilege]
    PrivilegeSelectOwnership: _ClassVar[ObjectPrivilege]
    PrivilegeManageOwnership: _ClassVar[ObjectPrivilege]
    PrivilegeSelectUser: _ClassVar[ObjectPrivilege]
    PrivilegeCreateResourceGroup: _ClassVar[ObjectPrivilege]
    PrivilegeDropResourceGroup: _ClassVar[ObjectPrivilege]
    PrivilegeDescribeResourceGroup: _ClassVar[ObjectPrivilege]
    PrivilegeListResourceGroups: _ClassVar[ObjectPrivilege]
    PrivilegeTransferNode: _ClassVar[ObjectPrivilege]
    PrivilegeTransferReplica: _ClassVar[ObjectPrivilege]
    PrivilegeGetLoadingProgress: _ClassVar[ObjectPrivilege]
    PrivilegeGetLoadState: _ClassVar[ObjectPrivilege]
    PrivilegeRenameCollection: _ClassVar[ObjectPrivilege]
    PrivilegeCreateDatabase: _ClassVar[ObjectPrivilege]
    PrivilegeDropDatabase: _ClassVar[ObjectPrivilege]
    PrivilegeListDatabases: _ClassVar[ObjectPrivilege]
    PrivilegeFlushAll: _ClassVar[ObjectPrivilege]

class StateCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    Initializing: _ClassVar[StateCode]
    Healthy: _ClassVar[StateCode]
    Abnormal: _ClassVar[StateCode]
    StandBy: _ClassVar[StateCode]
    Stopping: _ClassVar[StateCode]

class LoadState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    LoadStateNotExist: _ClassVar[LoadState]
    LoadStateNotLoad: _ClassVar[LoadState]
    LoadStateLoading: _ClassVar[LoadState]
    LoadStateLoaded: _ClassVar[LoadState]
Success: ErrorCode
UnexpectedError: ErrorCode
ConnectFailed: ErrorCode
PermissionDenied: ErrorCode
CollectionNotExists: ErrorCode
IllegalArgument: ErrorCode
IllegalDimension: ErrorCode
IllegalIndexType: ErrorCode
IllegalCollectionName: ErrorCode
IllegalTOPK: ErrorCode
IllegalRowRecord: ErrorCode
IllegalVectorID: ErrorCode
IllegalSearchResult: ErrorCode
FileNotFound: ErrorCode
MetaFailed: ErrorCode
CacheFailed: ErrorCode
CannotCreateFolder: ErrorCode
CannotCreateFile: ErrorCode
CannotDeleteFolder: ErrorCode
CannotDeleteFile: ErrorCode
BuildIndexError: ErrorCode
IllegalNLIST: ErrorCode
IllegalMetricType: ErrorCode
OutOfMemory: ErrorCode
IndexNotExist: ErrorCode
EmptyCollection: ErrorCode
UpdateImportTaskFailure: ErrorCode
CollectionNameNotFound: ErrorCode
CreateCredentialFailure: ErrorCode
UpdateCredentialFailure: ErrorCode
DeleteCredentialFailure: ErrorCode
GetCredentialFailure: ErrorCode
ListCredUsersFailure: ErrorCode
GetUserFailure: ErrorCode
CreateRoleFailure: ErrorCode
DropRoleFailure: ErrorCode
OperateUserRoleFailure: ErrorCode
SelectRoleFailure: ErrorCode
SelectUserFailure: ErrorCode
SelectResourceFailure: ErrorCode
OperatePrivilegeFailure: ErrorCode
SelectGrantFailure: ErrorCode
RefreshPolicyInfoCacheFailure: ErrorCode
ListPolicyFailure: ErrorCode
NotShardLeader: ErrorCode
NoReplicaAvailable: ErrorCode
SegmentNotFound: ErrorCode
ForceDeny: ErrorCode
RateLimit: ErrorCode
NodeIDNotMatch: ErrorCode
UpsertAutoIDTrue: ErrorCode
InsufficientMemoryToLoad: ErrorCode
MemoryQuotaExhausted: ErrorCode
DiskQuotaExhausted: ErrorCode
TimeTickLongDelay: ErrorCode
NotReadyServe: ErrorCode
NotReadyCoordActivating: ErrorCode
NotFoundTSafer: ErrorCode
DataCoordNA: ErrorCode
DDRequestRace: ErrorCode
IndexStateNone: IndexState
Unissued: IndexState
InProgress: IndexState
Finished: IndexState
Failed: IndexState
Retry: IndexState
SegmentStateNone: SegmentState
NotExist: SegmentState
Growing: SegmentState
Sealed: SegmentState
Flushed: SegmentState
Flushing: SegmentState
Dropped: SegmentState
Importing: SegmentState
None: PlaceholderType
BinaryVector: PlaceholderType
FloatVector: PlaceholderType
Undefined: MsgType
CreateCollection: MsgType
DropCollection: MsgType
HasCollection: MsgType
DescribeCollection: MsgType
ShowCollections: MsgType
GetSystemConfigs: MsgType
LoadCollection: MsgType
ReleaseCollection: MsgType
CreateAlias: MsgType
DropAlias: MsgType
AlterAlias: MsgType
AlterCollection: MsgType
RenameCollection: MsgType
CreatePartition: MsgType
DropPartition: MsgType
HasPartition: MsgType
DescribePartition: MsgType
ShowPartitions: MsgType
LoadPartitions: MsgType
ReleasePartitions: MsgType
ShowSegments: MsgType
DescribeSegment: MsgType
LoadSegments: MsgType
ReleaseSegments: MsgType
HandoffSegments: MsgType
LoadBalanceSegments: MsgType
DescribeSegments: MsgType
CreateIndex: MsgType
DescribeIndex: MsgType
DropIndex: MsgType
GetIndexStatistics: MsgType
Insert: MsgType
Delete: MsgType
Flush: MsgType
ResendSegmentStats: MsgType
Search: MsgType
SearchResult: MsgType
GetIndexState: MsgType
GetIndexBuildProgress: MsgType
GetCollectionStatistics: MsgType
GetPartitionStatistics: MsgType
Retrieve: MsgType
RetrieveResult: MsgType
WatchDmChannels: MsgType
RemoveDmChannels: MsgType
WatchQueryChannels: MsgType
RemoveQueryChannels: MsgType
SealedSegmentsChangeInfo: MsgType
WatchDeltaChannels: MsgType
GetShardLeaders: MsgType
GetReplicas: MsgType
UnsubDmChannel: MsgType
GetDistribution: MsgType
SyncDistribution: MsgType
SegmentInfo: MsgType
SystemInfo: MsgType
GetRecoveryInfo: MsgType
GetSegmentState: MsgType
TimeTick: MsgType
QueryNodeStats: MsgType
LoadIndex: MsgType
RequestID: MsgType
RequestTSO: MsgType
AllocateSegment: MsgType
SegmentStatistics: MsgType
SegmentFlushDone: MsgType
DataNodeTt: MsgType
Connect: MsgType
ListClientInfos: MsgType
AllocTimestamp: MsgType
CreateCredential: MsgType
GetCredential: MsgType
DeleteCredential: MsgType
UpdateCredential: MsgType
ListCredUsernames: MsgType
CreateRole: MsgType
DropRole: MsgType
OperateUserRole: MsgType
SelectRole: MsgType
SelectUser: MsgType
SelectResource: MsgType
OperatePrivilege: MsgType
SelectGrant: MsgType
RefreshPolicyInfoCache: MsgType
ListPolicy: MsgType
CreateResourceGroup: MsgType
DropResourceGroup: MsgType
ListResourceGroups: MsgType
DescribeResourceGroup: MsgType
TransferNode: MsgType
TransferReplica: MsgType
CreateDatabase: MsgType
DropDatabase: MsgType
ListDatabases: MsgType
Dsl: DslType
BoolExprV1: DslType
UndefiedState: CompactionState
Executing: CompactionState
Completed: CompactionState
Strong: ConsistencyLevel
Session: ConsistencyLevel
Bounded: ConsistencyLevel
Eventually: ConsistencyLevel
Customized: ConsistencyLevel
ImportPending: ImportState
ImportFailed: ImportState
ImportStarted: ImportState
ImportPersisted: ImportState
ImportCompleted: ImportState
ImportFailedAndCleaned: ImportState
Collection: ObjectType
Global: ObjectType
User: ObjectType
PrivilegeAll: ObjectPrivilege
PrivilegeCreateCollection: ObjectPrivilege
PrivilegeDropCollection: ObjectPrivilege
PrivilegeDescribeCollection: ObjectPrivilege
PrivilegeShowCollections: ObjectPrivilege
PrivilegeLoad: ObjectPrivilege
PrivilegeRelease: ObjectPrivilege
PrivilegeCompaction: ObjectPrivilege
PrivilegeInsert: ObjectPrivilege
PrivilegeDelete: ObjectPrivilege
PrivilegeGetStatistics: ObjectPrivilege
PrivilegeCreateIndex: ObjectPrivilege
PrivilegeIndexDetail: ObjectPrivilege
PrivilegeDropIndex: ObjectPrivilege
PrivilegeSearch: ObjectPrivilege
PrivilegeFlush: ObjectPrivilege
PrivilegeQuery: ObjectPrivilege
PrivilegeLoadBalance: ObjectPrivilege
PrivilegeImport: ObjectPrivilege
PrivilegeCreateOwnership: ObjectPrivilege
PrivilegeUpdateUser: ObjectPrivilege
PrivilegeDropOwnership: ObjectPrivilege
PrivilegeSelectOwnership: ObjectPrivilege
PrivilegeManageOwnership: ObjectPrivilege
PrivilegeSelectUser: ObjectPrivilege
PrivilegeCreateResourceGroup: ObjectPrivilege
PrivilegeDropResourceGroup: ObjectPrivilege
PrivilegeDescribeResourceGroup: ObjectPrivilege
PrivilegeListResourceGroups: ObjectPrivilege
PrivilegeTransferNode: ObjectPrivilege
PrivilegeTransferReplica: ObjectPrivilege
PrivilegeGetLoadingProgress: ObjectPrivilege
PrivilegeGetLoadState: ObjectPrivilege
PrivilegeRenameCollection: ObjectPrivilege
PrivilegeCreateDatabase: ObjectPrivilege
PrivilegeDropDatabase: ObjectPrivilege
PrivilegeListDatabases: ObjectPrivilege
PrivilegeFlushAll: ObjectPrivilege
Initializing: StateCode
Healthy: StateCode
Abnormal: StateCode
StandBy: StateCode
Stopping: StateCode
LoadStateNotExist: LoadState
LoadStateNotLoad: LoadState
LoadStateLoading: LoadState
LoadStateLoaded: LoadState
PRIVILEGE_EXT_OBJ_FIELD_NUMBER: _ClassVar[int]
privilege_ext_obj: _descriptor.FieldDescriptor

class Status(_message.Message):
    __slots__ = ["error_code", "reason"]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    error_code: ErrorCode
    reason: str
    def __init__(self, error_code: _Optional[_Union[ErrorCode, str]] = ..., reason: _Optional[str] = ...) -> None: ...

class KeyValuePair(_message.Message):
    __slots__ = ["key", "value"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class KeyDataPair(_message.Message):
    __slots__ = ["key", "data"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    key: str
    data: bytes
    def __init__(self, key: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class Blob(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: bytes
    def __init__(self, value: _Optional[bytes] = ...) -> None: ...

class PlaceholderValue(_message.Message):
    __slots__ = ["tag", "type", "values"]
    TAG_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    tag: str
    type: PlaceholderType
    values: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, tag: _Optional[str] = ..., type: _Optional[_Union[PlaceholderType, str]] = ..., values: _Optional[_Iterable[bytes]] = ...) -> None: ...

class PlaceholderGroup(_message.Message):
    __slots__ = ["placeholders"]
    PLACEHOLDERS_FIELD_NUMBER: _ClassVar[int]
    placeholders: _containers.RepeatedCompositeFieldContainer[PlaceholderValue]
    def __init__(self, placeholders: _Optional[_Iterable[_Union[PlaceholderValue, _Mapping]]] = ...) -> None: ...

class Address(_message.Message):
    __slots__ = ["ip", "port"]
    IP_FIELD_NUMBER: _ClassVar[int]
    PORT_FIELD_NUMBER: _ClassVar[int]
    ip: str
    port: int
    def __init__(self, ip: _Optional[str] = ..., port: _Optional[int] = ...) -> None: ...

class MsgBase(_message.Message):
    __slots__ = ["msg_type", "msgID", "timestamp", "sourceID", "targetID"]
    MSG_TYPE_FIELD_NUMBER: _ClassVar[int]
    MSGID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    SOURCEID_FIELD_NUMBER: _ClassVar[int]
    TARGETID_FIELD_NUMBER: _ClassVar[int]
    msg_type: MsgType
    msgID: int
    timestamp: int
    sourceID: int
    targetID: int
    def __init__(self, msg_type: _Optional[_Union[MsgType, str]] = ..., msgID: _Optional[int] = ..., timestamp: _Optional[int] = ..., sourceID: _Optional[int] = ..., targetID: _Optional[int] = ...) -> None: ...

class MsgHeader(_message.Message):
    __slots__ = ["base"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    base: MsgBase
    def __init__(self, base: _Optional[_Union[MsgBase, _Mapping]] = ...) -> None: ...

class DMLMsgHeader(_message.Message):
    __slots__ = ["base", "shardName"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    SHARDNAME_FIELD_NUMBER: _ClassVar[int]
    base: MsgBase
    shardName: str
    def __init__(self, base: _Optional[_Union[MsgBase, _Mapping]] = ..., shardName: _Optional[str] = ...) -> None: ...

class PrivilegeExt(_message.Message):
    __slots__ = ["object_type", "object_privilege", "object_name_index", "object_name_indexs"]
    OBJECT_TYPE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_PRIVILEGE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_NAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    OBJECT_NAME_INDEXS_FIELD_NUMBER: _ClassVar[int]
    object_type: ObjectType
    object_privilege: ObjectPrivilege
    object_name_index: int
    object_name_indexs: int
    def __init__(self, object_type: _Optional[_Union[ObjectType, str]] = ..., object_privilege: _Optional[_Union[ObjectPrivilege, str]] = ..., object_name_index: _Optional[int] = ..., object_name_indexs: _Optional[int] = ...) -> None: ...

class ClientInfo(_message.Message):
    __slots__ = ["sdk_type", "sdk_version", "local_time", "user", "host", "reserved"]
    class ReservedEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SDK_TYPE_FIELD_NUMBER: _ClassVar[int]
    SDK_VERSION_FIELD_NUMBER: _ClassVar[int]
    LOCAL_TIME_FIELD_NUMBER: _ClassVar[int]
    USER_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    RESERVED_FIELD_NUMBER: _ClassVar[int]
    sdk_type: str
    sdk_version: str
    local_time: str
    user: str
    host: str
    reserved: _containers.ScalarMap[str, str]
    def __init__(self, sdk_type: _Optional[str] = ..., sdk_version: _Optional[str] = ..., local_time: _Optional[str] = ..., user: _Optional[str] = ..., host: _Optional[str] = ..., reserved: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ["build_tags", "build_time", "git_commit", "go_version", "deploy_mode", "reserved"]
    class ReservedEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    BUILD_TAGS_FIELD_NUMBER: _ClassVar[int]
    BUILD_TIME_FIELD_NUMBER: _ClassVar[int]
    GIT_COMMIT_FIELD_NUMBER: _ClassVar[int]
    GO_VERSION_FIELD_NUMBER: _ClassVar[int]
    DEPLOY_MODE_FIELD_NUMBER: _ClassVar[int]
    RESERVED_FIELD_NUMBER: _ClassVar[int]
    build_tags: str
    build_time: str
    git_commit: str
    go_version: str
    deploy_mode: str
    reserved: _containers.ScalarMap[str, str]
    def __init__(self, build_tags: _Optional[str] = ..., build_time: _Optional[str] = ..., git_commit: _Optional[str] = ..., go_version: _Optional[str] = ..., deploy_mode: _Optional[str] = ..., reserved: _Optional[_Mapping[str, str]] = ...) -> None: ...
