import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pymilvus import MilvusClient
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import common_pb2, schema_pb2

logging.getLogger("faker").setLevel(logging.WARNING)
sys.path.append(Path(__file__).absolute().parent.parent)


# ============================================================
# Response Builder Utilities
# ============================================================


def make_status(code=0, reason="", error_code=0):
    """Create a mock status response."""
    status = MagicMock()
    status.code = code
    status.error_code = error_code
    status.reason = reason
    return status


def make_response(status_code=0, reason="", **kwargs):
    """Create a mock gRPC response with status and additional fields."""
    response = MagicMock()
    response.status = make_status(status_code, reason)
    for key, value in kwargs.items():
        setattr(response, key, value)
    return response


def make_collection_schema_response(
    collection_name="test_collection",
    collection_id=1,
    fields=None,
    enable_dynamic_field=False,
):
    """Create a mock DescribeCollection response."""
    if fields is None:
        fields = [
            {"name": "id", "type": DataType.INT64, "is_primary": True},
            {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": 128},
        ]

    response = MagicMock()
    response.status = make_status()
    response.collection_name = collection_name
    response.collectionID = collection_id
    response.schema = MagicMock()
    response.schema.name = collection_name
    response.schema.enable_dynamic_field = enable_dynamic_field

    mock_fields = []
    for f in fields:
        field = MagicMock()
        field.name = f["name"]
        field.data_type = f["type"]
        field.is_primary_key = f.get("is_primary", False)
        field.autoID = f.get("auto_id", False)
        field.type_params = []
        if "dim" in f:
            dim_param = MagicMock()
            dim_param.key = "dim"
            dim_param.value = str(f["dim"])
            field.type_params.append(dim_param)
        mock_fields.append(field)

    response.schema.fields = mock_fields
    response.properties = []
    response.shards_num = 1
    response.consistency_level = 0
    return response


def make_mutation_response(insert_count=1, ids=None, timestamp=1234567890):
    """Create a mock mutation (insert/upsert/delete) response."""
    if ids is None:
        ids = [1]
    response = MagicMock()
    response.status = make_status()
    response.insert_count = insert_count
    response.delete_count = insert_count
    response.upsert_count = insert_count

    # Mock IDs
    response.IDs = MagicMock()
    response.IDs.int_id = MagicMock()
    response.IDs.int_id.data = ids
    response.IDs.str_id = MagicMock()
    response.IDs.str_id.data = []

    response.timestamp = timestamp
    response.succ_index = list(range(len(ids)))
    response.err_index = []
    return response


def make_search_response(num_queries=1, top_k=10, ids=None, scores=None):
    """Create a mock search response."""
    if ids is None:
        ids = list(range(top_k))
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(top_k)]

    response = MagicMock()
    response.status = make_status()
    response.results = schema_pb2.SearchResultData(
        num_queries=num_queries,
        top_k=top_k,
        scores=scores * num_queries,
        ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=ids * num_queries)),
        topks=[top_k] * num_queries,
    )
    response.session_ts = 0
    return response


def make_query_response(fields_data=None):
    """Create a mock query response."""
    response = MagicMock()
    response.status = make_status()
    response.fields_data = fields_data or []
    response.collection_name = "test_collection"
    response.session_ts = 0
    return response


# ============================================================
# GrpcHandler Fixtures
# ============================================================


@pytest.fixture
def mock_grpc_channel():
    """Create a mock gRPC channel."""
    channel = MagicMock()
    channel.subscribe = MagicMock()
    channel.unsubscribe = MagicMock()
    channel.close = MagicMock()
    return channel


@pytest.fixture
def mock_grpc_stub():
    """Create a mock gRPC stub with all common methods."""
    stub = MagicMock()

    # Collection operations
    stub.CreateCollection = MagicMock(return_value=make_status())
    stub.DropCollection = MagicMock(return_value=make_status())
    stub.HasCollection = MagicMock(return_value=make_response(value=True))
    stub.DescribeCollection = MagicMock(return_value=make_collection_schema_response())
    stub.ShowCollections = MagicMock(
        return_value=make_response(collection_names=["test_collection"])
    )
    stub.LoadCollection = MagicMock(return_value=make_status())
    stub.ReleaseCollection = MagicMock(return_value=make_status())
    stub.RenameCollection = MagicMock(return_value=make_status())
    stub.GetCollectionStatistics = MagicMock(return_value=make_response(stats=[]))
    stub.TruncateCollection = MagicMock(return_value=make_response())
    stub.AlterCollection = MagicMock(return_value=make_status())
    stub.AlterCollectionField = MagicMock(return_value=make_status())
    stub.AddCollectionField = MagicMock(return_value=make_status())

    # Partition operations
    stub.CreatePartition = MagicMock(return_value=make_status())
    stub.DropPartition = MagicMock(return_value=make_status())
    stub.HasPartition = MagicMock(return_value=make_response(value=True))
    stub.ShowPartitions = MagicMock(return_value=make_response(partition_names=["_default"]))
    stub.LoadPartitions = MagicMock(return_value=make_status())
    stub.ReleasePartitions = MagicMock(return_value=make_status())
    stub.GetPartitionStatistics = MagicMock(return_value=make_response(stats=[]))

    # Data operations
    stub.Insert = MagicMock(return_value=make_mutation_response())
    stub.Delete = MagicMock(return_value=make_mutation_response())
    stub.Upsert = MagicMock(return_value=make_mutation_response())
    stub.Search = MagicMock(return_value=make_search_response())
    stub.HybridSearch = MagicMock(return_value=make_search_response())
    stub.Query = MagicMock(return_value=make_query_response())
    stub.Flush = MagicMock(return_value=make_response(coll_segIDs={}))
    stub.FlushAll = MagicMock(return_value=make_response(flush_all_ts=123456))

    # Index operations
    stub.CreateIndex = MagicMock(return_value=make_status())
    stub.DropIndex = MagicMock(return_value=make_status())
    stub.DescribeIndex = MagicMock(return_value=make_response(index_descriptions=[]))
    stub.GetIndexState = MagicMock(return_value=make_response(state=3))
    stub.GetIndexBuildProgress = MagicMock(
        return_value=make_response(total_rows=100, indexed_rows=100)
    )

    # Alias operations
    stub.CreateAlias = MagicMock(return_value=make_status())
    stub.DropAlias = MagicMock(return_value=make_status())
    stub.AlterAlias = MagicMock(return_value=make_status())
    stub.ListAliases = MagicMock(return_value=make_response(aliases=[]))
    stub.DescribeAlias = MagicMock(return_value=make_response(alias="", collection=""))

    # User operations
    stub.CreateCredential = MagicMock(return_value=make_status())
    stub.UpdateCredential = MagicMock(return_value=make_status())
    stub.DeleteCredential = MagicMock(return_value=make_status())
    stub.ListCredUsers = MagicMock(return_value=make_response(usernames=[]))
    stub.SelectUser = MagicMock(return_value=make_response(results=[]))

    # Role operations
    stub.CreateRole = MagicMock(return_value=make_status())
    stub.DropRole = MagicMock(return_value=make_status())
    stub.SelectRole = MagicMock(return_value=make_response(results=[]))
    stub.OperateUserRole = MagicMock(return_value=make_status())
    stub.SelectGrant = MagicMock(return_value=make_response(entities=[]))
    stub.OperatePrivilege = MagicMock(return_value=make_status())

    # Resource group operations
    stub.CreateResourceGroup = MagicMock(return_value=make_status())
    stub.DropResourceGroup = MagicMock(return_value=make_status())
    stub.ListResourceGroups = MagicMock(return_value=make_response(resource_groups=[]))
    stub.DescribeResourceGroup = MagicMock(return_value=make_response(resource_group=None))
    stub.UpdateResourceGroups = MagicMock(return_value=make_status())
    stub.TransferNode = MagicMock(return_value=make_status())
    stub.TransferReplica = MagicMock(return_value=make_status())

    # Database operations
    stub.CreateDatabase = MagicMock(return_value=make_status())
    stub.DropDatabase = MagicMock(return_value=make_status())
    stub.ListDatabases = MagicMock(return_value=make_response(db_names=["default"]))
    stub.DescribeDatabase = MagicMock(return_value=make_response(db_name="default"))
    stub.AlterDatabase = MagicMock(return_value=make_status())

    # Utility operations
    stub.GetVersion = MagicMock(return_value=make_response(version="v2.4.0"))
    stub.GetLoadState = MagicMock(return_value=make_response(state=3))
    stub.GetLoadingProgress = MagicMock(return_value=make_response(progress=100))
    stub.ManualCompaction = MagicMock(return_value=make_response(compactionID=1))
    stub.GetCompactionState = MagicMock(return_value=make_response(state=2))
    stub.GetCompactionStateWithPlans = MagicMock(return_value=make_response(state=2, mergeInfos=[]))
    stub.GetFlushState = MagicMock(return_value=make_response(flushed=True))
    stub.GetFlushAllState = MagicMock(return_value=make_response(flushed=True))
    stub.GetReplicas = MagicMock(return_value=make_response(replicas=[]))
    stub.Connect = MagicMock(return_value=make_response(identifier=12345))
    stub.AllocTimestamp = MagicMock(return_value=make_response(timestamp=123456789))

    # Import/Export operations
    stub.Import = MagicMock(return_value=make_response(tasks=[]))
    stub.GetImportState = MagicMock(return_value=make_response(state=2))
    stub.ListImportTasks = MagicMock(return_value=make_response(tasks=[]))
    stub.ListBulkInsertTasks = MagicMock(return_value=make_response(tasks=[]))

    # Add future variants for async operations
    for method_name in [
        "CreateCollection",
        "Insert",
        "Delete",
        "Upsert",
        "Search",
        "Query",
        "Flush",
    ]:
        method = getattr(stub, method_name)
        future_mock = MagicMock()
        future_mock.result = MagicMock(return_value=method.return_value)
        method.future = MagicMock(return_value=future_mock)

    return stub


@pytest.fixture
def mock_grpc_handler(mock_grpc_channel, mock_grpc_stub):
    """Create a GrpcHandler with mocked gRPC channel and stub."""
    with patch(
        "pymilvus.client.grpc_handler.grpc.insecure_channel", return_value=mock_grpc_channel
    ):
        with patch("pymilvus.client.grpc_handler.grpc.channel_ready_future"):
            with patch(
                "pymilvus.client.grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
                return_value=mock_grpc_stub,
            ):

                handler = GrpcHandler(uri="localhost:19530")
                handler._stub = mock_grpc_stub
                handler._channel = mock_grpc_channel
                yield handler


# ============================================================
# AsyncGrpcHandler Fixtures
# ============================================================


@pytest.fixture
def mock_async_channel():
    """Create a mock async gRPC channel."""
    channel = AsyncMock()
    channel.channel_ready = AsyncMock()
    channel.close = AsyncMock()
    channel._unary_unary_interceptors = []
    return channel


@pytest.fixture
def mock_async_stub():
    """Create a mock async gRPC stub with all common methods."""
    stub = AsyncMock()

    # Collection operations
    stub.CreateCollection = AsyncMock(return_value=make_status())
    stub.DropCollection = AsyncMock(return_value=make_status())
    stub.HasCollection = AsyncMock(return_value=make_response(value=True))
    stub.DescribeCollection = AsyncMock(return_value=make_collection_schema_response())
    stub.ShowCollections = AsyncMock(
        return_value=make_response(collection_names=["test_collection"])
    )
    stub.LoadCollection = AsyncMock(return_value=make_status())
    stub.ReleaseCollection = AsyncMock(return_value=make_status())
    stub.RenameCollection = AsyncMock(return_value=make_status())
    stub.GetCollectionStatistics = AsyncMock(return_value=make_response(stats=[]))
    stub.TruncateCollection = AsyncMock(return_value=make_response())
    stub.AlterCollection = AsyncMock(return_value=make_status())

    # Partition operations
    stub.CreatePartition = AsyncMock(return_value=make_status())
    stub.DropPartition = AsyncMock(return_value=make_status())
    stub.HasPartition = AsyncMock(return_value=make_response(value=True))
    stub.ShowPartitions = AsyncMock(return_value=make_response(partition_names=["_default"]))
    stub.LoadPartitions = AsyncMock(return_value=make_status())
    stub.ReleasePartitions = AsyncMock(return_value=make_status())
    stub.GetPartitionStatistics = AsyncMock(return_value=make_response(stats=[]))

    # Data operations
    stub.Insert = AsyncMock(return_value=make_mutation_response())
    stub.Delete = AsyncMock(return_value=make_mutation_response())
    stub.Upsert = AsyncMock(return_value=make_mutation_response())
    stub.Search = AsyncMock(return_value=make_search_response())
    stub.HybridSearch = AsyncMock(return_value=make_search_response())
    stub.Query = AsyncMock(return_value=make_query_response())
    stub.Flush = AsyncMock(return_value=make_response(coll_segIDs={}))
    stub.FlushAll = AsyncMock(return_value=make_response(flush_all_ts=123456))

    # Index operations
    stub.CreateIndex = AsyncMock(return_value=make_status())
    stub.DropIndex = AsyncMock(return_value=make_status())
    stub.DescribeIndex = AsyncMock(return_value=make_response(index_descriptions=[]))
    stub.GetIndexState = AsyncMock(return_value=make_response(state=3))
    stub.GetIndexBuildProgress = AsyncMock(
        return_value=make_response(total_rows=100, indexed_rows=100)
    )

    # Alias operations
    stub.CreateAlias = AsyncMock(return_value=make_status())
    stub.DropAlias = AsyncMock(return_value=make_status())
    stub.AlterAlias = AsyncMock(return_value=make_status())
    stub.ListAliases = AsyncMock(return_value=make_response(aliases=[]))
    stub.DescribeAlias = AsyncMock(return_value=make_response(alias="", collection=""))

    # User operations
    stub.CreateCredential = AsyncMock(return_value=make_status())
    stub.UpdateCredential = AsyncMock(return_value=make_status())
    stub.DeleteCredential = AsyncMock(return_value=make_status())
    stub.ListCredUsers = AsyncMock(return_value=make_response(usernames=[]))
    stub.SelectUser = AsyncMock(return_value=make_response(results=[]))

    # Role operations
    stub.CreateRole = AsyncMock(return_value=make_status())
    stub.DropRole = AsyncMock(return_value=make_status())
    stub.SelectRole = AsyncMock(return_value=make_response(results=[]))
    stub.OperateUserRole = AsyncMock(return_value=make_status())
    stub.SelectGrant = AsyncMock(return_value=make_response(entities=[]))
    stub.OperatePrivilege = AsyncMock(return_value=make_status())

    # Resource group operations
    stub.CreateResourceGroup = AsyncMock(return_value=make_status())
    stub.DropResourceGroup = AsyncMock(return_value=make_status())
    stub.ListResourceGroups = AsyncMock(return_value=make_response(resource_groups=[]))
    stub.DescribeResourceGroup = AsyncMock(return_value=make_response(resource_group=None))
    stub.UpdateResourceGroups = AsyncMock(return_value=make_status())

    # Database operations
    stub.CreateDatabase = AsyncMock(return_value=make_status())
    stub.DropDatabase = AsyncMock(return_value=make_status())
    stub.ListDatabases = AsyncMock(return_value=make_response(db_names=["default"]))
    stub.DescribeDatabase = AsyncMock(return_value=make_response(db_name="default"))
    stub.AlterDatabase = AsyncMock(return_value=make_status())

    # Utility operations
    stub.GetVersion = AsyncMock(return_value=make_response(version="v2.4.0"))
    stub.GetLoadState = AsyncMock(return_value=make_response(state=3))
    stub.GetLoadingProgress = AsyncMock(return_value=make_response(progress=100))
    stub.ManualCompaction = AsyncMock(return_value=make_response(compactionID=1))
    stub.GetCompactionState = AsyncMock(return_value=make_response(state=2))
    stub.GetFlushState = AsyncMock(return_value=make_response(flushed=True))
    stub.GetFlushAllState = AsyncMock(return_value=make_response(flushed=True))
    stub.GetReplicas = AsyncMock(return_value=make_response(replicas=[]))
    stub.Connect = AsyncMock(return_value=make_response(identifier=12345))
    stub.AllocTimestamp = AsyncMock(return_value=make_response(timestamp=123456789))

    # Snapshot operations
    stub.CreateSnapshot = AsyncMock(return_value=make_status())
    stub.DropSnapshot = AsyncMock(return_value=make_status())
    stub.ListSnapshots = AsyncMock(return_value=make_response(snapshots=[]))
    stub.DescribeSnapshot = AsyncMock(return_value=make_response())
    stub.RestoreSnapshot = AsyncMock(return_value=make_response(job_id=1))
    stub.GetRestoreSnapshotState = AsyncMock(return_value=make_response())
    stub.ListRestoreSnapshotJobs = AsyncMock(return_value=make_response(jobs=[]))

    return stub


@pytest.fixture
def mock_async_grpc_handler(mock_async_channel, mock_async_stub):
    """Create an AsyncGrpcHandler with mocked async channel and stub."""

    handler = AsyncGrpcHandler(channel=mock_async_channel)
    handler._is_channel_ready = True
    handler._async_stub = mock_async_stub
    return handler


# ============================================================
# MilvusClient Fixtures
# ============================================================


@pytest.fixture
def mock_milvus_client_handler():
    """Create a mock handler for MilvusClient."""
    handler = MagicMock()
    handler.get_server_type.return_value = "milvus"

    # Collection operations
    handler.create_collection = MagicMock()
    handler.drop_collection = MagicMock()
    handler.has_collection = MagicMock(return_value=True)
    handler.describe_collection = MagicMock(
        return_value={
            "collection_name": "test_collection",
            "auto_id": False,
            "fields": [
                {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": False},
                {"name": "vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 128}},
            ],
            "enable_dynamic_field": True,
        }
    )
    handler.list_collections = MagicMock(return_value=["test_collection"])
    handler.get_collection_stats = MagicMock(return_value={"row_count": 100})
    handler.load_collection = MagicMock()
    handler.release_collection = MagicMock()
    handler.rename_collections = MagicMock()

    # Partition operations
    handler.create_partition = MagicMock()
    handler.drop_partition = MagicMock()
    handler.has_partition = MagicMock(return_value=True)
    handler.list_partitions = MagicMock(return_value=["_default"])
    handler.load_partitions = MagicMock()
    handler.release_partitions = MagicMock()

    # Data operations
    result = MagicMock()
    result.insert_count = 1
    result.primary_keys = [1]
    result.cost = 0
    handler.insert_rows = MagicMock(return_value=result)
    handler.upsert_rows = MagicMock(return_value=result)
    handler.delete = MagicMock(return_value=result)
    handler.search = MagicMock(return_value=MagicMock())
    handler.hybrid_search = MagicMock(return_value=MagicMock())
    handler.query = MagicMock(return_value=[])
    handler.get = MagicMock(return_value=[])

    # Index operations
    handler.create_index = MagicMock()
    handler.drop_index = MagicMock()
    handler.describe_index = MagicMock(return_value=[])
    handler.list_indexes = MagicMock(return_value=[])

    # Alias operations
    handler.create_alias = MagicMock()
    handler.drop_alias = MagicMock()
    handler.alter_alias = MagicMock()
    handler.list_aliases = MagicMock(return_value=[])

    # User operations
    handler.create_user = MagicMock()
    handler.drop_user = MagicMock()
    handler.list_users = MagicMock(return_value=[])
    handler.describe_user = MagicMock(return_value={})

    # Role operations
    handler.create_role = MagicMock()
    handler.drop_role = MagicMock()
    handler.list_roles = MagicMock(return_value=[])
    handler.describe_role = MagicMock(return_value={})
    handler.grant_role = MagicMock()
    handler.revoke_role = MagicMock()

    # Database operations
    handler.create_database = MagicMock()
    handler.drop_database = MagicMock()
    handler.list_database = MagicMock(return_value=["default"])

    # Utility operations
    handler.flush = MagicMock()
    handler.compact = MagicMock(return_value=1)
    handler.get_load_state = MagicMock(return_value={"state": "Loaded"})
    handler.get_server_version = MagicMock(return_value="v2.4.0")

    return handler


@pytest.fixture
def mock_milvus_client(mock_milvus_client_handler):
    """Create a MilvusClient with mocked connection."""
    with patch("pymilvus.milvus_client.milvus_client.create_connection", return_value="test"):
        with patch(
            "pymilvus.orm.connections.Connections._fetch_handler",
            return_value=mock_milvus_client_handler,
        ):

            client = MilvusClient.__new__(MilvusClient)
            client._using = "test"
            client._db_name = "default"
            client.is_self_hosted = True
            yield client, mock_milvus_client_handler


# ============================================================
# Original Fixtures (kept for backward compatibility)
# ============================================================


@pytest.fixture
def mock_field_data_int64():
    """Create a mock FieldData with INT64 type."""
    field_data = schema_pb2.FieldData()
    field_data.type = DataType.INT64
    field_data.field_name = "test_field"
    return field_data


@pytest.fixture
def mock_field_data_array():
    """Create a mock FieldData with ARRAY type."""
    field_data = schema_pb2.FieldData()
    field_data.type = DataType.ARRAY
    return field_data


@pytest.fixture
def mock_collection_schema():
    """Create a mock collection schema for testing."""
    schema = schema_pb2.CollectionSchema()
    # Add primary key field
    pk_field = schema.fields.add()
    pk_field.name = "id"
    pk_field.data_type = DataType.INT64
    pk_field.is_primary_key = True
    # Add vector field
    vec_field = schema.fields.add()
    vec_field.name = "vector"
    vec_field.data_type = DataType.FLOAT_VECTOR
    return schema


@pytest.fixture
def mock_field_data_float_vector():
    """Create a mock FieldData with FLOAT_VECTOR type."""
    field_data = schema_pb2.FieldData()
    field_data.type = DataType.FLOAT_VECTOR
    field_data.field_name = "vector_field"
    return field_data
