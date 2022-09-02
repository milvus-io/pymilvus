# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63ommon.proto\x12\x13milvus.proto.common\x1a google/protobuf/descriptor.proto\"L\n\x06Status\x12\x32\n\nerror_code\x18\x01 \x01(\x0e\x32\x1e.milvus.proto.common.ErrorCode\x12\x0e\n\x06reason\x18\x02 \x01(\t\"*\n\x0cKeyValuePair\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"(\n\x0bKeyDataPair\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\x15\n\x04\x42lob\x12\r\n\x05value\x18\x01 \x01(\x0c\"c\n\x10PlaceholderValue\x12\x0b\n\x03tag\x18\x01 \x01(\t\x12\x32\n\x04type\x18\x02 \x01(\x0e\x32$.milvus.proto.common.PlaceholderType\x12\x0e\n\x06values\x18\x03 \x03(\x0c\"O\n\x10PlaceholderGroup\x12;\n\x0cplaceholders\x18\x01 \x03(\x0b\x32%.milvus.proto.common.PlaceholderValue\"#\n\x07\x41\x64\x64ress\x12\n\n\x02ip\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x03\"m\n\x07MsgBase\x12.\n\x08msg_type\x18\x01 \x01(\x0e\x32\x1c.milvus.proto.common.MsgType\x12\r\n\x05msgID\x18\x02 \x01(\x03\x12\x11\n\ttimestamp\x18\x03 \x01(\x04\x12\x10\n\x08sourceID\x18\x04 \x01(\x03\"7\n\tMsgHeader\x12*\n\x04\x62\x61se\x18\x01 \x01(\x0b\x32\x1c.milvus.proto.common.MsgBase\"M\n\x0c\x44MLMsgHeader\x12*\n\x04\x62\x61se\x18\x01 \x01(\x0b\x32\x1c.milvus.proto.common.MsgBase\x12\x11\n\tshardName\x18\x02 \x01(\t\"\xbb\x01\n\x0cPrivilegeExt\x12\x34\n\x0bobject_type\x18\x01 \x01(\x0e\x32\x1f.milvus.proto.common.ObjectType\x12>\n\x10object_privilege\x18\x02 \x01(\x0e\x32$.milvus.proto.common.ObjectPrivilege\x12\x19\n\x11object_name_index\x18\x03 \x01(\x05\x12\x1a\n\x12object_name_indexs\x18\x04 \x01(\x05*\xd3\x08\n\tErrorCode\x12\x0b\n\x07Success\x10\x00\x12\x13\n\x0fUnexpectedError\x10\x01\x12\x11\n\rConnectFailed\x10\x02\x12\x14\n\x10PermissionDenied\x10\x03\x12\x17\n\x13\x43ollectionNotExists\x10\x04\x12\x13\n\x0fIllegalArgument\x10\x05\x12\x14\n\x10IllegalDimension\x10\x07\x12\x14\n\x10IllegalIndexType\x10\x08\x12\x19\n\x15IllegalCollectionName\x10\t\x12\x0f\n\x0bIllegalTOPK\x10\n\x12\x14\n\x10IllegalRowRecord\x10\x0b\x12\x13\n\x0fIllegalVectorID\x10\x0c\x12\x17\n\x13IllegalSearchResult\x10\r\x12\x10\n\x0c\x46ileNotFound\x10\x0e\x12\x0e\n\nMetaFailed\x10\x0f\x12\x0f\n\x0b\x43\x61\x63heFailed\x10\x10\x12\x16\n\x12\x43\x61nnotCreateFolder\x10\x11\x12\x14\n\x10\x43\x61nnotCreateFile\x10\x12\x12\x16\n\x12\x43\x61nnotDeleteFolder\x10\x13\x12\x14\n\x10\x43\x61nnotDeleteFile\x10\x14\x12\x13\n\x0f\x42uildIndexError\x10\x15\x12\x10\n\x0cIllegalNLIST\x10\x16\x12\x15\n\x11IllegalMetricType\x10\x17\x12\x0f\n\x0bOutOfMemory\x10\x18\x12\x11\n\rIndexNotExist\x10\x19\x12\x13\n\x0f\x45mptyCollection\x10\x1a\x12\x1b\n\x17UpdateImportTaskFailure\x10\x1b\x12\x1a\n\x16\x43ollectionNameNotFound\x10\x1c\x12\x1b\n\x17\x43reateCredentialFailure\x10\x1d\x12\x1b\n\x17UpdateCredentialFailure\x10\x1e\x12\x1b\n\x17\x44\x65leteCredentialFailure\x10\x1f\x12\x18\n\x14GetCredentialFailure\x10 \x12\x18\n\x14ListCredUsersFailure\x10!\x12\x12\n\x0eGetUserFailure\x10\"\x12\x15\n\x11\x43reateRoleFailure\x10#\x12\x13\n\x0f\x44ropRoleFailure\x10$\x12\x1a\n\x16OperateUserRoleFailure\x10%\x12\x15\n\x11SelectRoleFailure\x10&\x12\x15\n\x11SelectUserFailure\x10\'\x12\x19\n\x15SelectResourceFailure\x10(\x12\x1b\n\x17OperatePrivilegeFailure\x10)\x12\x16\n\x12SelectGrantFailure\x10*\x12!\n\x1dRefreshPolicyInfoCacheFailure\x10+\x12\x15\n\x11ListPolicyFailure\x10,\x12\x12\n\x0eNotShardLeader\x10-\x12\x16\n\x12NoReplicaAvailable\x10.\x12\x13\n\x0fSegmentNotFound\x10/\x12\x12\n\rDDRequestRace\x10\xe8\x07*X\n\nIndexState\x12\x12\n\x0eIndexStateNone\x10\x00\x12\x0c\n\x08Unissued\x10\x01\x12\x0e\n\nInProgress\x10\x02\x12\x0c\n\x08\x46inished\x10\x03\x12\n\n\x06\x46\x61iled\x10\x04*\x82\x01\n\x0cSegmentState\x12\x14\n\x10SegmentStateNone\x10\x00\x12\x0c\n\x08NotExist\x10\x01\x12\x0b\n\x07Growing\x10\x02\x12\n\n\x06Sealed\x10\x03\x12\x0b\n\x07\x46lushed\x10\x04\x12\x0c\n\x08\x46lushing\x10\x05\x12\x0b\n\x07\x44ropped\x10\x06\x12\r\n\tImporting\x10\x07*>\n\x0fPlaceholderType\x12\x08\n\x04None\x10\x00\x12\x10\n\x0c\x42inaryVector\x10\x64\x12\x0f\n\x0b\x46loatVector\x10\x65*\xb6\x0c\n\x07MsgType\x12\r\n\tUndefined\x10\x00\x12\x14\n\x10\x43reateCollection\x10\x64\x12\x12\n\x0e\x44ropCollection\x10\x65\x12\x11\n\rHasCollection\x10\x66\x12\x16\n\x12\x44\x65scribeCollection\x10g\x12\x13\n\x0fShowCollections\x10h\x12\x14\n\x10GetSystemConfigs\x10i\x12\x12\n\x0eLoadCollection\x10j\x12\x15\n\x11ReleaseCollection\x10k\x12\x0f\n\x0b\x43reateAlias\x10l\x12\r\n\tDropAlias\x10m\x12\x0e\n\nAlterAlias\x10n\x12\x14\n\x0f\x43reatePartition\x10\xc8\x01\x12\x12\n\rDropPartition\x10\xc9\x01\x12\x11\n\x0cHasPartition\x10\xca\x01\x12\x16\n\x11\x44\x65scribePartition\x10\xcb\x01\x12\x13\n\x0eShowPartitions\x10\xcc\x01\x12\x13\n\x0eLoadPartitions\x10\xcd\x01\x12\x16\n\x11ReleasePartitions\x10\xce\x01\x12\x11\n\x0cShowSegments\x10\xfa\x01\x12\x14\n\x0f\x44\x65scribeSegment\x10\xfb\x01\x12\x11\n\x0cLoadSegments\x10\xfc\x01\x12\x14\n\x0fReleaseSegments\x10\xfd\x01\x12\x14\n\x0fHandoffSegments\x10\xfe\x01\x12\x18\n\x13LoadBalanceSegments\x10\xff\x01\x12\x15\n\x10\x44\x65scribeSegments\x10\x80\x02\x12\x10\n\x0b\x43reateIndex\x10\xac\x02\x12\x12\n\rDescribeIndex\x10\xad\x02\x12\x0e\n\tDropIndex\x10\xae\x02\x12\x0b\n\x06Insert\x10\x90\x03\x12\x0b\n\x06\x44\x65lete\x10\x91\x03\x12\n\n\x05\x46lush\x10\x92\x03\x12\x17\n\x12ResendSegmentStats\x10\x93\x03\x12\x0b\n\x06Search\x10\xf4\x03\x12\x11\n\x0cSearchResult\x10\xf5\x03\x12\x12\n\rGetIndexState\x10\xf6\x03\x12\x1a\n\x15GetIndexBuildProgress\x10\xf7\x03\x12\x1c\n\x17GetCollectionStatistics\x10\xf8\x03\x12\x1b\n\x16GetPartitionStatistics\x10\xf9\x03\x12\r\n\x08Retrieve\x10\xfa\x03\x12\x13\n\x0eRetrieveResult\x10\xfb\x03\x12\x14\n\x0fWatchDmChannels\x10\xfc\x03\x12\x15\n\x10RemoveDmChannels\x10\xfd\x03\x12\x17\n\x12WatchQueryChannels\x10\xfe\x03\x12\x18\n\x13RemoveQueryChannels\x10\xff\x03\x12\x1d\n\x18SealedSegmentsChangeInfo\x10\x80\x04\x12\x17\n\x12WatchDeltaChannels\x10\x81\x04\x12\x14\n\x0fGetShardLeaders\x10\x82\x04\x12\x10\n\x0bGetReplicas\x10\x83\x04\x12\x10\n\x0bSegmentInfo\x10\xd8\x04\x12\x0f\n\nSystemInfo\x10\xd9\x04\x12\x14\n\x0fGetRecoveryInfo\x10\xda\x04\x12\x14\n\x0fGetSegmentState\x10\xdb\x04\x12\r\n\x08TimeTick\x10\xb0\t\x12\x13\n\x0eQueryNodeStats\x10\xb1\t\x12\x0e\n\tLoadIndex\x10\xb2\t\x12\x0e\n\tRequestID\x10\xb3\t\x12\x0f\n\nRequestTSO\x10\xb4\t\x12\x14\n\x0f\x41llocateSegment\x10\xb5\t\x12\x16\n\x11SegmentStatistics\x10\xb6\t\x12\x15\n\x10SegmentFlushDone\x10\xb7\t\x12\x0f\n\nDataNodeTt\x10\xb8\t\x12\x15\n\x10\x43reateCredential\x10\xdc\x0b\x12\x12\n\rGetCredential\x10\xdd\x0b\x12\x15\n\x10\x44\x65leteCredential\x10\xde\x0b\x12\x15\n\x10UpdateCredential\x10\xdf\x0b\x12\x16\n\x11ListCredUsernames\x10\xe0\x0b\x12\x0f\n\nCreateRole\x10\xc0\x0c\x12\r\n\x08\x44ropRole\x10\xc1\x0c\x12\x14\n\x0fOperateUserRole\x10\xc2\x0c\x12\x0f\n\nSelectRole\x10\xc3\x0c\x12\x0f\n\nSelectUser\x10\xc4\x0c\x12\x13\n\x0eSelectResource\x10\xc5\x0c\x12\x15\n\x10OperatePrivilege\x10\xc6\x0c\x12\x10\n\x0bSelectGrant\x10\xc7\x0c\x12\x1b\n\x16RefreshPolicyInfoCache\x10\xc8\x0c\x12\x0f\n\nListPolicy\x10\xc9\x0c*\"\n\x07\x44slType\x12\x07\n\x03\x44sl\x10\x00\x12\x0e\n\nBoolExprV1\x10\x01*B\n\x0f\x43ompactionState\x12\x11\n\rUndefiedState\x10\x00\x12\r\n\tExecuting\x10\x01\x12\r\n\tCompleted\x10\x02*X\n\x10\x43onsistencyLevel\x12\n\n\x06Strong\x10\x00\x12\x0b\n\x07Session\x10\x01\x12\x0b\n\x07\x42ounded\x10\x02\x12\x0e\n\nEventually\x10\x03\x12\x0e\n\nCustomized\x10\x04*\xaf\x01\n\x0bImportState\x12\x11\n\rImportPending\x10\x00\x12\x10\n\x0cImportFailed\x10\x01\x12\x11\n\rImportStarted\x10\x02\x12\x14\n\x10ImportDownloaded\x10\x03\x12\x10\n\x0cImportParsed\x10\x04\x12\x13\n\x0fImportPersisted\x10\x05\x12\x13\n\x0fImportCompleted\x10\x06\x12\x16\n\x12ImportAllocSegment\x10\n*2\n\nObjectType\x12\x0e\n\nCollection\x10\x00\x12\n\n\x06Global\x10\x01\x12\x08\n\x04User\x10\x02*\xa6\x05\n\x0fObjectPrivilege\x12\x10\n\x0cPrivilegeAll\x10\x00\x12\x1d\n\x19PrivilegeCreateCollection\x10\x01\x12\x1b\n\x17PrivilegeDropCollection\x10\x02\x12\x1f\n\x1bPrivilegeDescribeCollection\x10\x03\x12\x1c\n\x18PrivilegeShowCollections\x10\x04\x12\x11\n\rPrivilegeLoad\x10\x05\x12\x14\n\x10PrivilegeRelease\x10\x06\x12\x17\n\x13PrivilegeCompaction\x10\x07\x12\x13\n\x0fPrivilegeInsert\x10\x08\x12\x13\n\x0fPrivilegeDelete\x10\t\x12\x1a\n\x16PrivilegeGetStatistics\x10\n\x12\x18\n\x14PrivilegeCreateIndex\x10\x0b\x12\x18\n\x14PrivilegeIndexDetail\x10\x0c\x12\x16\n\x12PrivilegeDropIndex\x10\r\x12\x13\n\x0fPrivilegeSearch\x10\x0e\x12\x12\n\x0ePrivilegeFlush\x10\x0f\x12\x12\n\x0ePrivilegeQuery\x10\x10\x12\x18\n\x14PrivilegeLoadBalance\x10\x11\x12\x13\n\x0fPrivilegeImport\x10\x12\x12\x1c\n\x18PrivilegeCreateOwnership\x10\x13\x12\x17\n\x13PrivilegeUpdateUser\x10\x14\x12\x1a\n\x16PrivilegeDropOwnership\x10\x15\x12\x1c\n\x18PrivilegeSelectOwnership\x10\x16\x12\x1c\n\x18PrivilegeManageOwnership\x10\x17\x12\x17\n\x13PrivilegeSelectUser\x10\x18\x12\x1e\n\x1aPrivilegeDescribePartition\x10\x19:^\n\x11privilege_ext_obj\x12\x1f.google.protobuf.MessageOptions\x18\xe9\x07 \x01(\x0b\x32!.milvus.proto.common.PrivilegeExtBW\n\x0eio.milvus.grpcB\x0b\x43ommonProtoP\x01Z3github.com/milvus-io/milvus/internal/proto/commonpb\xa0\x01\x01\x62\x06proto3')

_ERRORCODE = DESCRIPTOR.enum_types_by_name['ErrorCode']
ErrorCode = enum_type_wrapper.EnumTypeWrapper(_ERRORCODE)
_INDEXSTATE = DESCRIPTOR.enum_types_by_name['IndexState']
IndexState = enum_type_wrapper.EnumTypeWrapper(_INDEXSTATE)
_SEGMENTSTATE = DESCRIPTOR.enum_types_by_name['SegmentState']
SegmentState = enum_type_wrapper.EnumTypeWrapper(_SEGMENTSTATE)
_PLACEHOLDERTYPE = DESCRIPTOR.enum_types_by_name['PlaceholderType']
PlaceholderType = enum_type_wrapper.EnumTypeWrapper(_PLACEHOLDERTYPE)
_MSGTYPE = DESCRIPTOR.enum_types_by_name['MsgType']
MsgType = enum_type_wrapper.EnumTypeWrapper(_MSGTYPE)
_DSLTYPE = DESCRIPTOR.enum_types_by_name['DslType']
DslType = enum_type_wrapper.EnumTypeWrapper(_DSLTYPE)
_COMPACTIONSTATE = DESCRIPTOR.enum_types_by_name['CompactionState']
CompactionState = enum_type_wrapper.EnumTypeWrapper(_COMPACTIONSTATE)
_CONSISTENCYLEVEL = DESCRIPTOR.enum_types_by_name['ConsistencyLevel']
ConsistencyLevel = enum_type_wrapper.EnumTypeWrapper(_CONSISTENCYLEVEL)
_IMPORTSTATE = DESCRIPTOR.enum_types_by_name['ImportState']
ImportState = enum_type_wrapper.EnumTypeWrapper(_IMPORTSTATE)
_OBJECTTYPE = DESCRIPTOR.enum_types_by_name['ObjectType']
ObjectType = enum_type_wrapper.EnumTypeWrapper(_OBJECTTYPE)
_OBJECTPRIVILEGE = DESCRIPTOR.enum_types_by_name['ObjectPrivilege']
ObjectPrivilege = enum_type_wrapper.EnumTypeWrapper(_OBJECTPRIVILEGE)
Success = 0
UnexpectedError = 1
ConnectFailed = 2
PermissionDenied = 3
CollectionNotExists = 4
IllegalArgument = 5
IllegalDimension = 7
IllegalIndexType = 8
IllegalCollectionName = 9
IllegalTOPK = 10
IllegalRowRecord = 11
IllegalVectorID = 12
IllegalSearchResult = 13
FileNotFound = 14
MetaFailed = 15
CacheFailed = 16
CannotCreateFolder = 17
CannotCreateFile = 18
CannotDeleteFolder = 19
CannotDeleteFile = 20
BuildIndexError = 21
IllegalNLIST = 22
IllegalMetricType = 23
OutOfMemory = 24
IndexNotExist = 25
EmptyCollection = 26
UpdateImportTaskFailure = 27
CollectionNameNotFound = 28
CreateCredentialFailure = 29
UpdateCredentialFailure = 30
DeleteCredentialFailure = 31
GetCredentialFailure = 32
ListCredUsersFailure = 33
GetUserFailure = 34
CreateRoleFailure = 35
DropRoleFailure = 36
OperateUserRoleFailure = 37
SelectRoleFailure = 38
SelectUserFailure = 39
SelectResourceFailure = 40
OperatePrivilegeFailure = 41
SelectGrantFailure = 42
RefreshPolicyInfoCacheFailure = 43
ListPolicyFailure = 44
NotShardLeader = 45
NoReplicaAvailable = 46
SegmentNotFound = 47
DDRequestRace = 1000
IndexStateNone = 0
Unissued = 1
InProgress = 2
Finished = 3
Failed = 4
SegmentStateNone = 0
NotExist = 1
Growing = 2
Sealed = 3
Flushed = 4
Flushing = 5
Dropped = 6
Importing = 7
globals()['None'] = 0
BinaryVector = 100
FloatVector = 101
Undefined = 0
CreateCollection = 100
DropCollection = 101
HasCollection = 102
DescribeCollection = 103
ShowCollections = 104
GetSystemConfigs = 105
LoadCollection = 106
ReleaseCollection = 107
CreateAlias = 108
DropAlias = 109
AlterAlias = 110
CreatePartition = 200
DropPartition = 201
HasPartition = 202
DescribePartition = 203
ShowPartitions = 204
LoadPartitions = 205
ReleasePartitions = 206
ShowSegments = 250
DescribeSegment = 251
LoadSegments = 252
ReleaseSegments = 253
HandoffSegments = 254
LoadBalanceSegments = 255
DescribeSegments = 256
CreateIndex = 300
DescribeIndex = 301
DropIndex = 302
Insert = 400
Delete = 401
Flush = 402
ResendSegmentStats = 403
Search = 500
SearchResult = 501
GetIndexState = 502
GetIndexBuildProgress = 503
GetCollectionStatistics = 504
GetPartitionStatistics = 505
Retrieve = 506
RetrieveResult = 507
WatchDmChannels = 508
RemoveDmChannels = 509
WatchQueryChannels = 510
RemoveQueryChannels = 511
SealedSegmentsChangeInfo = 512
WatchDeltaChannels = 513
GetShardLeaders = 514
GetReplicas = 515
SegmentInfo = 600
SystemInfo = 601
GetRecoveryInfo = 602
GetSegmentState = 603
TimeTick = 1200
QueryNodeStats = 1201
LoadIndex = 1202
RequestID = 1203
RequestTSO = 1204
AllocateSegment = 1205
SegmentStatistics = 1206
SegmentFlushDone = 1207
DataNodeTt = 1208
CreateCredential = 1500
GetCredential = 1501
DeleteCredential = 1502
UpdateCredential = 1503
ListCredUsernames = 1504
CreateRole = 1600
DropRole = 1601
OperateUserRole = 1602
SelectRole = 1603
SelectUser = 1604
SelectResource = 1605
OperatePrivilege = 1606
SelectGrant = 1607
RefreshPolicyInfoCache = 1608
ListPolicy = 1609
Dsl = 0
BoolExprV1 = 1
UndefiedState = 0
Executing = 1
Completed = 2
Strong = 0
Session = 1
Bounded = 2
Eventually = 3
Customized = 4
ImportPending = 0
ImportFailed = 1
ImportStarted = 2
ImportDownloaded = 3
ImportParsed = 4
ImportPersisted = 5
ImportCompleted = 6
ImportAllocSegment = 10
Collection = 0
Global = 1
User = 2
PrivilegeAll = 0
PrivilegeCreateCollection = 1
PrivilegeDropCollection = 2
PrivilegeDescribeCollection = 3
PrivilegeShowCollections = 4
PrivilegeLoad = 5
PrivilegeRelease = 6
PrivilegeCompaction = 7
PrivilegeInsert = 8
PrivilegeDelete = 9
PrivilegeGetStatistics = 10
PrivilegeCreateIndex = 11
PrivilegeIndexDetail = 12
PrivilegeDropIndex = 13
PrivilegeSearch = 14
PrivilegeFlush = 15
PrivilegeQuery = 16
PrivilegeLoadBalance = 17
PrivilegeImport = 18
PrivilegeCreateOwnership = 19
PrivilegeUpdateUser = 20
PrivilegeDropOwnership = 21
PrivilegeSelectOwnership = 22
PrivilegeManageOwnership = 23
PrivilegeSelectUser = 24
PrivilegeDescribePartition = 25

PRIVILEGE_EXT_OBJ_FIELD_NUMBER = 1001
privilege_ext_obj = DESCRIPTOR.extensions_by_name['privilege_ext_obj']

_STATUS = DESCRIPTOR.message_types_by_name['Status']
_KEYVALUEPAIR = DESCRIPTOR.message_types_by_name['KeyValuePair']
_KEYDATAPAIR = DESCRIPTOR.message_types_by_name['KeyDataPair']
_BLOB = DESCRIPTOR.message_types_by_name['Blob']
_PLACEHOLDERVALUE = DESCRIPTOR.message_types_by_name['PlaceholderValue']
_PLACEHOLDERGROUP = DESCRIPTOR.message_types_by_name['PlaceholderGroup']
_ADDRESS = DESCRIPTOR.message_types_by_name['Address']
_MSGBASE = DESCRIPTOR.message_types_by_name['MsgBase']
_MSGHEADER = DESCRIPTOR.message_types_by_name['MsgHeader']
_DMLMSGHEADER = DESCRIPTOR.message_types_by_name['DMLMsgHeader']
_PRIVILEGEEXT = DESCRIPTOR.message_types_by_name['PrivilegeExt']
Status = _reflection.GeneratedProtocolMessageType('Status', (_message.Message,), {
  'DESCRIPTOR' : _STATUS,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.Status)
  })
_sym_db.RegisterMessage(Status)

KeyValuePair = _reflection.GeneratedProtocolMessageType('KeyValuePair', (_message.Message,), {
  'DESCRIPTOR' : _KEYVALUEPAIR,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.KeyValuePair)
  })
_sym_db.RegisterMessage(KeyValuePair)

KeyDataPair = _reflection.GeneratedProtocolMessageType('KeyDataPair', (_message.Message,), {
  'DESCRIPTOR' : _KEYDATAPAIR,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.KeyDataPair)
  })
_sym_db.RegisterMessage(KeyDataPair)

Blob = _reflection.GeneratedProtocolMessageType('Blob', (_message.Message,), {
  'DESCRIPTOR' : _BLOB,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.Blob)
  })
_sym_db.RegisterMessage(Blob)

PlaceholderValue = _reflection.GeneratedProtocolMessageType('PlaceholderValue', (_message.Message,), {
  'DESCRIPTOR' : _PLACEHOLDERVALUE,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.PlaceholderValue)
  })
_sym_db.RegisterMessage(PlaceholderValue)

PlaceholderGroup = _reflection.GeneratedProtocolMessageType('PlaceholderGroup', (_message.Message,), {
  'DESCRIPTOR' : _PLACEHOLDERGROUP,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.PlaceholderGroup)
  })
_sym_db.RegisterMessage(PlaceholderGroup)

Address = _reflection.GeneratedProtocolMessageType('Address', (_message.Message,), {
  'DESCRIPTOR' : _ADDRESS,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.Address)
  })
_sym_db.RegisterMessage(Address)

MsgBase = _reflection.GeneratedProtocolMessageType('MsgBase', (_message.Message,), {
  'DESCRIPTOR' : _MSGBASE,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.MsgBase)
  })
_sym_db.RegisterMessage(MsgBase)

MsgHeader = _reflection.GeneratedProtocolMessageType('MsgHeader', (_message.Message,), {
  'DESCRIPTOR' : _MSGHEADER,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.MsgHeader)
  })
_sym_db.RegisterMessage(MsgHeader)

DMLMsgHeader = _reflection.GeneratedProtocolMessageType('DMLMsgHeader', (_message.Message,), {
  'DESCRIPTOR' : _DMLMSGHEADER,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.DMLMsgHeader)
  })
_sym_db.RegisterMessage(DMLMsgHeader)

PrivilegeExt = _reflection.GeneratedProtocolMessageType('PrivilegeExt', (_message.Message,), {
  'DESCRIPTOR' : _PRIVILEGEEXT,
  '__module__' : 'common_pb2'
  # @@protoc_insertion_point(class_scope:milvus.proto.common.PrivilegeExt)
  })
_sym_db.RegisterMessage(PrivilegeExt)

if _descriptor._USE_C_DESCRIPTORS == False:
  google_dot_protobuf_dot_descriptor__pb2.MessageOptions.RegisterExtension(privilege_ext_obj)

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\016io.milvus.grpcB\013CommonProtoP\001Z3github.com/milvus-io/milvus/internal/proto/commonpb\240\001\001'
  _ERRORCODE._serialized_start=915
  _ERRORCODE._serialized_end=2022
  _INDEXSTATE._serialized_start=2024
  _INDEXSTATE._serialized_end=2112
  _SEGMENTSTATE._serialized_start=2115
  _SEGMENTSTATE._serialized_end=2245
  _PLACEHOLDERTYPE._serialized_start=2247
  _PLACEHOLDERTYPE._serialized_end=2309
  _MSGTYPE._serialized_start=2312
  _MSGTYPE._serialized_end=3902
  _DSLTYPE._serialized_start=3904
  _DSLTYPE._serialized_end=3938
  _COMPACTIONSTATE._serialized_start=3940
  _COMPACTIONSTATE._serialized_end=4006
  _CONSISTENCYLEVEL._serialized_start=4008
  _CONSISTENCYLEVEL._serialized_end=4096
  _IMPORTSTATE._serialized_start=4099
  _IMPORTSTATE._serialized_end=4274
  _OBJECTTYPE._serialized_start=4276
  _OBJECTTYPE._serialized_end=4326
  _OBJECTPRIVILEGE._serialized_start=4329
  _OBJECTPRIVILEGE._serialized_end=5007
  _STATUS._serialized_start=71
  _STATUS._serialized_end=147
  _KEYVALUEPAIR._serialized_start=149
  _KEYVALUEPAIR._serialized_end=191
  _KEYDATAPAIR._serialized_start=193
  _KEYDATAPAIR._serialized_end=233
  _BLOB._serialized_start=235
  _BLOB._serialized_end=256
  _PLACEHOLDERVALUE._serialized_start=258
  _PLACEHOLDERVALUE._serialized_end=357
  _PLACEHOLDERGROUP._serialized_start=359
  _PLACEHOLDERGROUP._serialized_end=438
  _ADDRESS._serialized_start=440
  _ADDRESS._serialized_end=475
  _MSGBASE._serialized_start=477
  _MSGBASE._serialized_end=586
  _MSGHEADER._serialized_start=588
  _MSGHEADER._serialized_end=643
  _DMLMSGHEADER._serialized_start=645
  _DMLMSGHEADER._serialized_end=722
  _PRIVILEGEEXT._serialized_start=725
  _PRIVILEGEEXT._serialized_end=912
# @@protoc_insertion_point(module_scope)
