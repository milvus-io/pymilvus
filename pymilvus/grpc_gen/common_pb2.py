# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import descriptor_pb2 as google_dot_protobuf_dot_descriptor__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63ommon.proto\x12\x13milvus.proto.common\x1a google/protobuf/descriptor.proto\"\xf3\x01\n\x06Status\x12\x36\n\nerror_code\x18\x01 \x01(\x0e\x32\x1e.milvus.proto.common.ErrorCodeB\x02\x18\x01\x12\x0e\n\x06reason\x18\x02 \x01(\t\x12\x0c\n\x04\x63ode\x18\x03 \x01(\x05\x12\x11\n\tretriable\x18\x04 \x01(\x08\x12\x0e\n\x06\x64\x65tail\x18\x05 \x01(\t\x12>\n\nextra_info\x18\x06 \x03(\x0b\x32*.milvus.proto.common.Status.ExtraInfoEntry\x1a\x30\n\x0e\x45xtraInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"*\n\x0cKeyValuePair\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"(\n\x0bKeyDataPair\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\"\x15\n\x04\x42lob\x12\r\n\x05value\x18\x01 \x01(\x0c\"c\n\x10PlaceholderValue\x12\x0b\n\x03tag\x18\x01 \x01(\t\x12\x32\n\x04type\x18\x02 \x01(\x0e\x32$.milvus.proto.common.PlaceholderType\x12\x0e\n\x06values\x18\x03 \x03(\x0c\"O\n\x10PlaceholderGroup\x12;\n\x0cplaceholders\x18\x01 \x03(\x0b\x32%.milvus.proto.common.PlaceholderValue\"#\n\x07\x41\x64\x64ress\x12\n\n\x02ip\x18\x01 \x01(\t\x12\x0c\n\x04port\x18\x02 \x01(\x03\"\xaf\x02\n\x07MsgBase\x12.\n\x08msg_type\x18\x01 \x01(\x0e\x32\x1c.milvus.proto.common.MsgType\x12\r\n\x05msgID\x18\x02 \x01(\x03\x12\x11\n\ttimestamp\x18\x03 \x01(\x04\x12\x10\n\x08sourceID\x18\x04 \x01(\x03\x12\x10\n\x08targetID\x18\x05 \x01(\x03\x12@\n\nproperties\x18\x06 \x03(\x0b\x32,.milvus.proto.common.MsgBase.PropertiesEntry\x12\x39\n\rreplicateInfo\x18\x07 \x01(\x0b\x32\".milvus.proto.common.ReplicateInfo\x1a\x31\n\x0fPropertiesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\":\n\rReplicateInfo\x12\x13\n\x0bisReplicate\x18\x01 \x01(\x08\x12\x14\n\x0cmsgTimestamp\x18\x02 \x01(\x04\"7\n\tMsgHeader\x12*\n\x04\x62\x61se\x18\x01 \x01(\x0b\x32\x1c.milvus.proto.common.MsgBase\"M\n\x0c\x44MLMsgHeader\x12*\n\x04\x62\x61se\x18\x01 \x01(\x0b\x32\x1c.milvus.proto.common.MsgBase\x12\x11\n\tshardName\x18\x02 \x01(\t\"\xbb\x01\n\x0cPrivilegeExt\x12\x34\n\x0bobject_type\x18\x01 \x01(\x0e\x32\x1f.milvus.proto.common.ObjectType\x12>\n\x10object_privilege\x18\x02 \x01(\x0e\x32$.milvus.proto.common.ObjectPrivilege\x12\x19\n\x11object_name_index\x18\x03 \x01(\x05\x12\x1a\n\x12object_name_indexs\x18\x04 \x01(\x05\"2\n\x0cSegmentStats\x12\x11\n\tSegmentID\x18\x01 \x01(\x03\x12\x0f\n\x07NumRows\x18\x02 \x01(\x03\"\xd5\x01\n\nClientInfo\x12\x10\n\x08sdk_type\x18\x01 \x01(\t\x12\x13\n\x0bsdk_version\x18\x02 \x01(\t\x12\x12\n\nlocal_time\x18\x03 \x01(\t\x12\x0c\n\x04user\x18\x04 \x01(\t\x12\x0c\n\x04host\x18\x05 \x01(\t\x12?\n\x08reserved\x18\x06 \x03(\x0b\x32-.milvus.proto.common.ClientInfo.ReservedEntry\x1a/\n\rReservedEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xe3\x01\n\nServerInfo\x12\x12\n\nbuild_tags\x18\x01 \x01(\t\x12\x12\n\nbuild_time\x18\x02 \x01(\t\x12\x12\n\ngit_commit\x18\x03 \x01(\t\x12\x12\n\ngo_version\x18\x04 \x01(\t\x12\x13\n\x0b\x64\x65ploy_mode\x18\x05 \x01(\t\x12?\n\x08reserved\x18\x06 \x03(\x0b\x32-.milvus.proto.common.ServerInfo.ReservedEntry\x1a/\n\rReservedEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\">\n\x08NodeInfo\x12\x0f\n\x07node_id\x18\x01 \x01(\x03\x12\x0f\n\x07\x61\x64\x64ress\x18\x02 \x01(\t\x12\x10\n\x08hostname\x18\x03 \x01(\t*\xc7\n\n\tErrorCode\x12\x0b\n\x07Success\x10\x00\x12\x13\n\x0fUnexpectedError\x10\x01\x12\x11\n\rConnectFailed\x10\x02\x12\x14\n\x10PermissionDenied\x10\x03\x12\x17\n\x13\x43ollectionNotExists\x10\x04\x12\x13\n\x0fIllegalArgument\x10\x05\x12\x14\n\x10IllegalDimension\x10\x07\x12\x14\n\x10IllegalIndexType\x10\x08\x12\x19\n\x15IllegalCollectionName\x10\t\x12\x0f\n\x0bIllegalTOPK\x10\n\x12\x14\n\x10IllegalRowRecord\x10\x0b\x12\x13\n\x0fIllegalVectorID\x10\x0c\x12\x17\n\x13IllegalSearchResult\x10\r\x12\x10\n\x0c\x46ileNotFound\x10\x0e\x12\x0e\n\nMetaFailed\x10\x0f\x12\x0f\n\x0b\x43\x61\x63heFailed\x10\x10\x12\x16\n\x12\x43\x61nnotCreateFolder\x10\x11\x12\x14\n\x10\x43\x61nnotCreateFile\x10\x12\x12\x16\n\x12\x43\x61nnotDeleteFolder\x10\x13\x12\x14\n\x10\x43\x61nnotDeleteFile\x10\x14\x12\x13\n\x0f\x42uildIndexError\x10\x15\x12\x10\n\x0cIllegalNLIST\x10\x16\x12\x15\n\x11IllegalMetricType\x10\x17\x12\x0f\n\x0bOutOfMemory\x10\x18\x12\x11\n\rIndexNotExist\x10\x19\x12\x13\n\x0f\x45mptyCollection\x10\x1a\x12\x1b\n\x17UpdateImportTaskFailure\x10\x1b\x12\x1a\n\x16\x43ollectionNameNotFound\x10\x1c\x12\x1b\n\x17\x43reateCredentialFailure\x10\x1d\x12\x1b\n\x17UpdateCredentialFailure\x10\x1e\x12\x1b\n\x17\x44\x65leteCredentialFailure\x10\x1f\x12\x18\n\x14GetCredentialFailure\x10 \x12\x18\n\x14ListCredUsersFailure\x10!\x12\x12\n\x0eGetUserFailure\x10\"\x12\x15\n\x11\x43reateRoleFailure\x10#\x12\x13\n\x0f\x44ropRoleFailure\x10$\x12\x1a\n\x16OperateUserRoleFailure\x10%\x12\x15\n\x11SelectRoleFailure\x10&\x12\x15\n\x11SelectUserFailure\x10\'\x12\x19\n\x15SelectResourceFailure\x10(\x12\x1b\n\x17OperatePrivilegeFailure\x10)\x12\x16\n\x12SelectGrantFailure\x10*\x12!\n\x1dRefreshPolicyInfoCacheFailure\x10+\x12\x15\n\x11ListPolicyFailure\x10,\x12\x12\n\x0eNotShardLeader\x10-\x12\x16\n\x12NoReplicaAvailable\x10.\x12\x13\n\x0fSegmentNotFound\x10/\x12\r\n\tForceDeny\x10\x30\x12\r\n\tRateLimit\x10\x31\x12\x12\n\x0eNodeIDNotMatch\x10\x32\x12\x14\n\x10UpsertAutoIDTrue\x10\x33\x12\x1c\n\x18InsufficientMemoryToLoad\x10\x34\x12\x18\n\x14MemoryQuotaExhausted\x10\x35\x12\x16\n\x12\x44iskQuotaExhausted\x10\x36\x12\x15\n\x11TimeTickLongDelay\x10\x37\x12\x11\n\rNotReadyServe\x10\x38\x12\x1b\n\x17NotReadyCoordActivating\x10\x39\x12\x0f\n\x0b\x44\x61taCoordNA\x10\x64\x12\x12\n\rDDRequestRace\x10\xe8\x07\x1a\x02\x18\x01*c\n\nIndexState\x12\x12\n\x0eIndexStateNone\x10\x00\x12\x0c\n\x08Unissued\x10\x01\x12\x0e\n\nInProgress\x10\x02\x12\x0c\n\x08\x46inished\x10\x03\x12\n\n\x06\x46\x61iled\x10\x04\x12\t\n\x05Retry\x10\x05*\x82\x01\n\x0cSegmentState\x12\x14\n\x10SegmentStateNone\x10\x00\x12\x0c\n\x08NotExist\x10\x01\x12\x0b\n\x07Growing\x10\x02\x12\n\n\x06Sealed\x10\x03\x12\x0b\n\x07\x46lushed\x10\x04\x12\x0c\n\x08\x46lushing\x10\x05\x12\x0b\n\x07\x44ropped\x10\x06\x12\r\n\tImporting\x10\x07*\x94\x01\n\x0fPlaceholderType\x12\x08\n\x04None\x10\x00\x12\x10\n\x0c\x42inaryVector\x10\x64\x12\x0f\n\x0b\x46loatVector\x10\x65\x12\x11\n\rFloat16Vector\x10\x66\x12\x12\n\x0e\x42\x46loat16Vector\x10g\x12\x15\n\x11SparseFloatVector\x10h\x12\t\n\x05Int64\x10\x05\x12\x0b\n\x07VarChar\x10\x15*\xe0\x10\n\x07MsgType\x12\r\n\tUndefined\x10\x00\x12\x14\n\x10\x43reateCollection\x10\x64\x12\x12\n\x0e\x44ropCollection\x10\x65\x12\x11\n\rHasCollection\x10\x66\x12\x16\n\x12\x44\x65scribeCollection\x10g\x12\x13\n\x0fShowCollections\x10h\x12\x14\n\x10GetSystemConfigs\x10i\x12\x12\n\x0eLoadCollection\x10j\x12\x15\n\x11ReleaseCollection\x10k\x12\x0f\n\x0b\x43reateAlias\x10l\x12\r\n\tDropAlias\x10m\x12\x0e\n\nAlterAlias\x10n\x12\x13\n\x0f\x41lterCollection\x10o\x12\x14\n\x10RenameCollection\x10p\x12\x11\n\rDescribeAlias\x10q\x12\x0f\n\x0bListAliases\x10r\x12\x14\n\x0f\x43reatePartition\x10\xc8\x01\x12\x12\n\rDropPartition\x10\xc9\x01\x12\x11\n\x0cHasPartition\x10\xca\x01\x12\x16\n\x11\x44\x65scribePartition\x10\xcb\x01\x12\x13\n\x0eShowPartitions\x10\xcc\x01\x12\x13\n\x0eLoadPartitions\x10\xcd\x01\x12\x16\n\x11ReleasePartitions\x10\xce\x01\x12\x11\n\x0cShowSegments\x10\xfa\x01\x12\x14\n\x0f\x44\x65scribeSegment\x10\xfb\x01\x12\x11\n\x0cLoadSegments\x10\xfc\x01\x12\x14\n\x0fReleaseSegments\x10\xfd\x01\x12\x14\n\x0fHandoffSegments\x10\xfe\x01\x12\x18\n\x13LoadBalanceSegments\x10\xff\x01\x12\x15\n\x10\x44\x65scribeSegments\x10\x80\x02\x12\x1c\n\x17\x46\x65\x64\x65rListIndexedSegment\x10\x81\x02\x12\"\n\x1d\x46\x65\x64\x65rDescribeSegmentIndexData\x10\x82\x02\x12\x10\n\x0b\x43reateIndex\x10\xac\x02\x12\x12\n\rDescribeIndex\x10\xad\x02\x12\x0e\n\tDropIndex\x10\xae\x02\x12\x17\n\x12GetIndexStatistics\x10\xaf\x02\x12\x0f\n\nAlterIndex\x10\xb0\x02\x12\x0b\n\x06Insert\x10\x90\x03\x12\x0b\n\x06\x44\x65lete\x10\x91\x03\x12\n\n\x05\x46lush\x10\x92\x03\x12\x17\n\x12ResendSegmentStats\x10\x93\x03\x12\x0b\n\x06Upsert\x10\x94\x03\x12\x0b\n\x06Search\x10\xf4\x03\x12\x11\n\x0cSearchResult\x10\xf5\x03\x12\x12\n\rGetIndexState\x10\xf6\x03\x12\x1a\n\x15GetIndexBuildProgress\x10\xf7\x03\x12\x1c\n\x17GetCollectionStatistics\x10\xf8\x03\x12\x1b\n\x16GetPartitionStatistics\x10\xf9\x03\x12\r\n\x08Retrieve\x10\xfa\x03\x12\x13\n\x0eRetrieveResult\x10\xfb\x03\x12\x14\n\x0fWatchDmChannels\x10\xfc\x03\x12\x15\n\x10RemoveDmChannels\x10\xfd\x03\x12\x17\n\x12WatchQueryChannels\x10\xfe\x03\x12\x18\n\x13RemoveQueryChannels\x10\xff\x03\x12\x1d\n\x18SealedSegmentsChangeInfo\x10\x80\x04\x12\x17\n\x12WatchDeltaChannels\x10\x81\x04\x12\x14\n\x0fGetShardLeaders\x10\x82\x04\x12\x10\n\x0bGetReplicas\x10\x83\x04\x12\x13\n\x0eUnsubDmChannel\x10\x84\x04\x12\x14\n\x0fGetDistribution\x10\x85\x04\x12\x15\n\x10SyncDistribution\x10\x86\x04\x12\x10\n\x0bSegmentInfo\x10\xd8\x04\x12\x0f\n\nSystemInfo\x10\xd9\x04\x12\x14\n\x0fGetRecoveryInfo\x10\xda\x04\x12\x14\n\x0fGetSegmentState\x10\xdb\x04\x12\r\n\x08TimeTick\x10\xb0\t\x12\x13\n\x0eQueryNodeStats\x10\xb1\t\x12\x0e\n\tLoadIndex\x10\xb2\t\x12\x0e\n\tRequestID\x10\xb3\t\x12\x0f\n\nRequestTSO\x10\xb4\t\x12\x14\n\x0f\x41llocateSegment\x10\xb5\t\x12\x16\n\x11SegmentStatistics\x10\xb6\t\x12\x15\n\x10SegmentFlushDone\x10\xb7\t\x12\x0f\n\nDataNodeTt\x10\xb8\t\x12\x0c\n\x07\x43onnect\x10\xb9\t\x12\x14\n\x0fListClientInfos\x10\xba\t\x12\x13\n\x0e\x41llocTimestamp\x10\xbb\t\x12\x15\n\x10\x43reateCredential\x10\xdc\x0b\x12\x12\n\rGetCredential\x10\xdd\x0b\x12\x15\n\x10\x44\x65leteCredential\x10\xde\x0b\x12\x15\n\x10UpdateCredential\x10\xdf\x0b\x12\x16\n\x11ListCredUsernames\x10\xe0\x0b\x12\x0f\n\nCreateRole\x10\xc0\x0c\x12\r\n\x08\x44ropRole\x10\xc1\x0c\x12\x14\n\x0fOperateUserRole\x10\xc2\x0c\x12\x0f\n\nSelectRole\x10\xc3\x0c\x12\x0f\n\nSelectUser\x10\xc4\x0c\x12\x13\n\x0eSelectResource\x10\xc5\x0c\x12\x15\n\x10OperatePrivilege\x10\xc6\x0c\x12\x10\n\x0bSelectGrant\x10\xc7\x0c\x12\x1b\n\x16RefreshPolicyInfoCache\x10\xc8\x0c\x12\x0f\n\nListPolicy\x10\xc9\x0c\x12\x18\n\x13\x43reateResourceGroup\x10\xa4\r\x12\x16\n\x11\x44ropResourceGroup\x10\xa5\r\x12\x17\n\x12ListResourceGroups\x10\xa6\r\x12\x1a\n\x15\x44\x65scribeResourceGroup\x10\xa7\r\x12\x11\n\x0cTransferNode\x10\xa8\r\x12\x14\n\x0fTransferReplica\x10\xa9\r\x12\x19\n\x14UpdateResourceGroups\x10\xaa\r\x12\x13\n\x0e\x43reateDatabase\x10\x89\x0e\x12\x11\n\x0c\x44ropDatabase\x10\x8a\x0e\x12\x12\n\rListDatabases\x10\x8b\x0e*\"\n\x07\x44slType\x12\x07\n\x03\x44sl\x10\x00\x12\x0e\n\nBoolExprV1\x10\x01*B\n\x0f\x43ompactionState\x12\x11\n\rUndefiedState\x10\x00\x12\r\n\tExecuting\x10\x01\x12\r\n\tCompleted\x10\x02*X\n\x10\x43onsistencyLevel\x12\n\n\x06Strong\x10\x00\x12\x0b\n\x07Session\x10\x01\x12\x0b\n\x07\x42ounded\x10\x02\x12\x0e\n\nEventually\x10\x03\x12\x0e\n\nCustomized\x10\x04*\x9e\x01\n\x0bImportState\x12\x11\n\rImportPending\x10\x00\x12\x10\n\x0cImportFailed\x10\x01\x12\x11\n\rImportStarted\x10\x02\x12\x13\n\x0fImportPersisted\x10\x05\x12\x11\n\rImportFlushed\x10\x08\x12\x13\n\x0fImportCompleted\x10\x06\x12\x1a\n\x16ImportFailedAndCleaned\x10\x07*2\n\nObjectType\x12\x0e\n\nCollection\x10\x00\x12\n\n\x06Global\x10\x01\x12\x08\n\x04User\x10\x02*\xd6\n\n\x0fObjectPrivilege\x12\x10\n\x0cPrivilegeAll\x10\x00\x12\x1d\n\x19PrivilegeCreateCollection\x10\x01\x12\x1b\n\x17PrivilegeDropCollection\x10\x02\x12\x1f\n\x1bPrivilegeDescribeCollection\x10\x03\x12\x1c\n\x18PrivilegeShowCollections\x10\x04\x12\x11\n\rPrivilegeLoad\x10\x05\x12\x14\n\x10PrivilegeRelease\x10\x06\x12\x17\n\x13PrivilegeCompaction\x10\x07\x12\x13\n\x0fPrivilegeInsert\x10\x08\x12\x13\n\x0fPrivilegeDelete\x10\t\x12\x1a\n\x16PrivilegeGetStatistics\x10\n\x12\x18\n\x14PrivilegeCreateIndex\x10\x0b\x12\x18\n\x14PrivilegeIndexDetail\x10\x0c\x12\x16\n\x12PrivilegeDropIndex\x10\r\x12\x13\n\x0fPrivilegeSearch\x10\x0e\x12\x12\n\x0ePrivilegeFlush\x10\x0f\x12\x12\n\x0ePrivilegeQuery\x10\x10\x12\x18\n\x14PrivilegeLoadBalance\x10\x11\x12\x13\n\x0fPrivilegeImport\x10\x12\x12\x1c\n\x18PrivilegeCreateOwnership\x10\x13\x12\x17\n\x13PrivilegeUpdateUser\x10\x14\x12\x1a\n\x16PrivilegeDropOwnership\x10\x15\x12\x1c\n\x18PrivilegeSelectOwnership\x10\x16\x12\x1c\n\x18PrivilegeManageOwnership\x10\x17\x12\x17\n\x13PrivilegeSelectUser\x10\x18\x12\x13\n\x0fPrivilegeUpsert\x10\x19\x12 \n\x1cPrivilegeCreateResourceGroup\x10\x1a\x12\x1e\n\x1aPrivilegeDropResourceGroup\x10\x1b\x12\"\n\x1ePrivilegeDescribeResourceGroup\x10\x1c\x12\x1f\n\x1bPrivilegeListResourceGroups\x10\x1d\x12\x19\n\x15PrivilegeTransferNode\x10\x1e\x12\x1c\n\x18PrivilegeTransferReplica\x10\x1f\x12\x1f\n\x1bPrivilegeGetLoadingProgress\x10 \x12\x19\n\x15PrivilegeGetLoadState\x10!\x12\x1d\n\x19PrivilegeRenameCollection\x10\"\x12\x1b\n\x17PrivilegeCreateDatabase\x10#\x12\x19\n\x15PrivilegeDropDatabase\x10$\x12\x1a\n\x16PrivilegeListDatabases\x10%\x12\x15\n\x11PrivilegeFlushAll\x10&\x12\x1c\n\x18PrivilegeCreatePartition\x10\'\x12\x1a\n\x16PrivilegeDropPartition\x10(\x12\x1b\n\x17PrivilegeShowPartitions\x10)\x12\x19\n\x15PrivilegeHasPartition\x10*\x12\x1a\n\x16PrivilegeGetFlushState\x10+\x12\x18\n\x14PrivilegeCreateAlias\x10,\x12\x16\n\x12PrivilegeDropAlias\x10-\x12\x1a\n\x16PrivilegeDescribeAlias\x10.\x12\x18\n\x14PrivilegeListAliases\x10/\x12!\n\x1dPrivilegeUpdateResourceGroups\x10\x30\x12\x1a\n\x16PrivilegeAlterDatabase\x10\x31*S\n\tStateCode\x12\x10\n\x0cInitializing\x10\x00\x12\x0b\n\x07Healthy\x10\x01\x12\x0c\n\x08\x41\x62normal\x10\x02\x12\x0b\n\x07StandBy\x10\x03\x12\x0c\n\x08Stopping\x10\x04*c\n\tLoadState\x12\x15\n\x11LoadStateNotExist\x10\x00\x12\x14\n\x10LoadStateNotLoad\x10\x01\x12\x14\n\x10LoadStateLoading\x10\x02\x12\x13\n\x0fLoadStateLoaded\x10\x03:^\n\x11privilege_ext_obj\x12\x1f.google.protobuf.MessageOptions\x18\xe9\x07 \x01(\x0b\x32!.milvus.proto.common.PrivilegeExtBm\n\x0eio.milvus.grpcB\x0b\x43ommonProtoP\x01Z4github.com/milvus-io/milvus-proto/go-api/v2/commonpb\xa0\x01\x01\xaa\x02\x12Milvus.Client.Grpcb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'common_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\016io.milvus.grpcB\013CommonProtoP\001Z4github.com/milvus-io/milvus-proto/go-api/v2/commonpb\240\001\001\252\002\022Milvus.Client.Grpc'
  _globals['_ERRORCODE']._options = None
  _globals['_ERRORCODE']._serialized_options = b'\030\001'
  _globals['_STATUS_EXTRAINFOENTRY']._options = None
  _globals['_STATUS_EXTRAINFOENTRY']._serialized_options = b'8\001'
  _globals['_STATUS'].fields_by_name['error_code']._options = None
  _globals['_STATUS'].fields_by_name['error_code']._serialized_options = b'\030\001'
  _globals['_MSGBASE_PROPERTIESENTRY']._options = None
  _globals['_MSGBASE_PROPERTIESENTRY']._serialized_options = b'8\001'
  _globals['_CLIENTINFO_RESERVEDENTRY']._options = None
  _globals['_CLIENTINFO_RESERVEDENTRY']._serialized_options = b'8\001'
  _globals['_SERVERINFO_RESERVEDENTRY']._options = None
  _globals['_SERVERINFO_RESERVEDENTRY']._serialized_options = b'8\001'
  _globals['_ERRORCODE']._serialized_start=1900
  _globals['_ERRORCODE']._serialized_end=3251
  _globals['_INDEXSTATE']._serialized_start=3253
  _globals['_INDEXSTATE']._serialized_end=3352
  _globals['_SEGMENTSTATE']._serialized_start=3355
  _globals['_SEGMENTSTATE']._serialized_end=3485
  _globals['_PLACEHOLDERTYPE']._serialized_start=3488
  _globals['_PLACEHOLDERTYPE']._serialized_end=3636
  _globals['_MSGTYPE']._serialized_start=3639
  _globals['_MSGTYPE']._serialized_end=5783
  _globals['_DSLTYPE']._serialized_start=5785
  _globals['_DSLTYPE']._serialized_end=5819
  _globals['_COMPACTIONSTATE']._serialized_start=5821
  _globals['_COMPACTIONSTATE']._serialized_end=5887
  _globals['_CONSISTENCYLEVEL']._serialized_start=5889
  _globals['_CONSISTENCYLEVEL']._serialized_end=5977
  _globals['_IMPORTSTATE']._serialized_start=5980
  _globals['_IMPORTSTATE']._serialized_end=6138
  _globals['_OBJECTTYPE']._serialized_start=6140
  _globals['_OBJECTTYPE']._serialized_end=6190
  _globals['_OBJECTPRIVILEGE']._serialized_start=6193
  _globals['_OBJECTPRIVILEGE']._serialized_end=7559
  _globals['_STATECODE']._serialized_start=7561
  _globals['_STATECODE']._serialized_end=7644
  _globals['_LOADSTATE']._serialized_start=7646
  _globals['_LOADSTATE']._serialized_end=7745
  _globals['_STATUS']._serialized_start=72
  _globals['_STATUS']._serialized_end=315
  _globals['_STATUS_EXTRAINFOENTRY']._serialized_start=267
  _globals['_STATUS_EXTRAINFOENTRY']._serialized_end=315
  _globals['_KEYVALUEPAIR']._serialized_start=317
  _globals['_KEYVALUEPAIR']._serialized_end=359
  _globals['_KEYDATAPAIR']._serialized_start=361
  _globals['_KEYDATAPAIR']._serialized_end=401
  _globals['_BLOB']._serialized_start=403
  _globals['_BLOB']._serialized_end=424
  _globals['_PLACEHOLDERVALUE']._serialized_start=426
  _globals['_PLACEHOLDERVALUE']._serialized_end=525
  _globals['_PLACEHOLDERGROUP']._serialized_start=527
  _globals['_PLACEHOLDERGROUP']._serialized_end=606
  _globals['_ADDRESS']._serialized_start=608
  _globals['_ADDRESS']._serialized_end=643
  _globals['_MSGBASE']._serialized_start=646
  _globals['_MSGBASE']._serialized_end=949
  _globals['_MSGBASE_PROPERTIESENTRY']._serialized_start=900
  _globals['_MSGBASE_PROPERTIESENTRY']._serialized_end=949
  _globals['_REPLICATEINFO']._serialized_start=951
  _globals['_REPLICATEINFO']._serialized_end=1009
  _globals['_MSGHEADER']._serialized_start=1011
  _globals['_MSGHEADER']._serialized_end=1066
  _globals['_DMLMSGHEADER']._serialized_start=1068
  _globals['_DMLMSGHEADER']._serialized_end=1145
  _globals['_PRIVILEGEEXT']._serialized_start=1148
  _globals['_PRIVILEGEEXT']._serialized_end=1335
  _globals['_SEGMENTSTATS']._serialized_start=1337
  _globals['_SEGMENTSTATS']._serialized_end=1387
  _globals['_CLIENTINFO']._serialized_start=1390
  _globals['_CLIENTINFO']._serialized_end=1603
  _globals['_CLIENTINFO_RESERVEDENTRY']._serialized_start=1556
  _globals['_CLIENTINFO_RESERVEDENTRY']._serialized_end=1603
  _globals['_SERVERINFO']._serialized_start=1606
  _globals['_SERVERINFO']._serialized_end=1833
  _globals['_SERVERINFO_RESERVEDENTRY']._serialized_start=1556
  _globals['_SERVERINFO_RESERVEDENTRY']._serialized_end=1603
  _globals['_NODEINFO']._serialized_start=1835
  _globals['_NODEINFO']._serialized_end=1897
# @@protoc_insertion_point(module_scope)
