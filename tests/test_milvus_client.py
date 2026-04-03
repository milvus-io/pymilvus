import logging
from unittest.mock import ANY, MagicMock, patch

import pytest
from pymilvus import DataType
from pymilvus.client.connection_manager import ConnectionManager
from pymilvus.client.types import (
    LoadState,
    RefreshExternalCollectionJobInfo,
    RestoreSnapshotJobInfo,
    SnapshotInfo,
)
from pymilvus.exceptions import (
    DataTypeNotMatchException,
    MilvusException,
    ParamError,
    PrimaryKeyException,
)
from pymilvus.milvus_client.index import IndexParams
from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus.milvus_client.optimize_task import OptimizeResult, OptimizeTask

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_connection_manager():
    """Reset ConnectionManager singleton before and after each test."""
    ConnectionManager._reset_instance()
    yield
    ConnectionManager._reset_instance()


class TestMilvusClient:
    @pytest.mark.parametrize("index_params", [None, {}, "str", MilvusClient.prepare_index_params()])
    def test_create_index_invalid_params(self, index_params):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            if isinstance(index_params, IndexParams):
                with pytest.raises(
                    ParamError, match="IndexParams is empty, no index can be created"
                ):
                    client.create_index("test_collection", index_params)
            elif index_params is None:
                with pytest.raises(ParamError, match=r"missing required argument:.*"):
                    client.create_index("test_collection", index_params)
            else:
                with pytest.raises(ParamError, match=r"wrong type of argument .*"):
                    client.create_index("test_collection", index_params)

    def test_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        assert len(index_params) == 0

        index_params.add_index("vector", index_type="FLAT", metric_type="L2")
        assert len(index_params) == 1

        index_params.add_index("vector2", index_type="HNSW", efConstruction=100, metric_type="L2")

        log.info(index_params)
        assert len(index_params) == 2

        for index in index_params:
            log.info(index)

    def test_connection_reuse(self):
        """Test that connections with same config share handler, different configs get different handlers."""
        mock_handler1 = MagicMock()
        mock_handler1.get_server_type.return_value = "milvus"
        mock_handler1._wait_for_channel_ready = MagicMock()

        mock_handler2 = MagicMock()
        mock_handler2.get_server_type.return_value = "milvus"
        mock_handler2._wait_for_channel_ready = MagicMock()

        mock_handler3 = MagicMock()
        mock_handler3.get_server_type.return_value = "milvus"
        mock_handler3._wait_for_channel_ready = MagicMock()

        with patch(
            "pymilvus.client.grpc_handler.GrpcHandler",
            side_effect=[mock_handler1, mock_handler2, mock_handler3],
        ):
            # Same URI and no token - should share same handler
            client1 = MilvusClient()
            client1_handler = client1._handler

            client2 = MilvusClient()
            assert client2._handler is client1_handler, "Same config should share handler"

            # Different token - should get different handler
            client3 = MilvusClient(user="test", password="foobar")
            assert (
                client3._handler is not client1_handler
            ), "Different token should get different handler"

            # Different token again - should get yet another handler
            client4 = MilvusClient(token="foobar")
            assert (
                client4._handler is not client1_handler
            ), "Different token should get different handler"
            assert (
                client4._handler is not client3._handler
            ), "Different token should get different handler"

            client1.close()
            client2.close()
            client3.close()
            client4.close()

    @pytest.mark.parametrize(
        "data_type",
        [
            "FLOAT_VECTOR",
            "BINARY_VECTOR",
            "FLOAT16_VECTOR",
            "BFLOAT16_VECTOR",
            "SPARSE_FLOAT_VECTOR",
            "INT8_VECTOR",
        ],
    )
    def test_add_collection_field_vector_requires_nullable(self, data_type):
        """Test that adding vector field to collection requires nullable=True"""

        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            dtype = getattr(DataType, data_type)

            # Should raise ParamError when nullable is not set or False
            with pytest.raises(
                ParamError,
                match="Adding vector field to existing collection requires nullable=True",
            ):
                client.add_collection_field(
                    collection_name="test_collection",
                    field_name="vector_field",
                    data_type=dtype,
                    dim=128,
                )

            # Should raise ParamError when nullable is explicitly False
            with pytest.raises(
                ParamError,
                match="Adding vector field to existing collection requires nullable=True",
            ):
                client.add_collection_field(
                    collection_name="test_collection",
                    field_name="vector_field",
                    data_type=dtype,
                    dim=128,
                    nullable=False,
                )

    def test_add_collection_field_vector_with_nullable_true(self):
        """Test that adding vector field with nullable=True passes validation"""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_conn = MagicMock()

        with patch(
            "pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler
        ), patch.object(MilvusClient, "_get_connection", return_value=mock_conn):
            client = MilvusClient()

            # Should not raise when nullable=True
            client.add_collection_field(
                collection_name="test_collection",
                field_name="vector_field",
                data_type=DataType.FLOAT_VECTOR,
                dim=128,
                nullable=True,
            )
            mock_conn.add_collection_field.assert_called_once()

    def test_add_collection_field_non_vector_no_nullable_required(self):
        """Test that non-vector fields don't require nullable=True"""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_conn = MagicMock()

        with patch(
            "pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler
        ), patch.object(MilvusClient, "_get_connection", return_value=mock_conn):
            client = MilvusClient()

            # Non-vector types should not require nullable
            client.add_collection_field(
                collection_name="test_collection",
                field_name="int_field",
                data_type=DataType.INT64,
            )
            mock_conn.add_collection_field.assert_called_once()


class TestMilvusClientSnapshot:
    """Test snapshot-related APIs in MilvusClient."""

    def test_create_snapshot(self):
        """Test create_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.create_snapshot.return_value = None

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            client.create_snapshot(
                collection_name="test_collection",
                snapshot_name="test_snapshot",
                description="Test description",
            )

            mock_handler.create_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot",
                collection_name="test_collection",
                description="Test description",
                timeout=None,
                context=ANY,
            )

    def test_create_snapshot_minimal(self):
        """Test create_snapshot with minimal parameters."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.create_snapshot.return_value = None

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            client.create_snapshot(collection_name="test_collection", snapshot_name="test_snapshot")

            mock_handler.create_snapshot.assert_called_once()
            call_kwargs = mock_handler.create_snapshot.call_args[1]
            assert call_kwargs["snapshot_name"] == "test_snapshot"
            assert call_kwargs["collection_name"] == "test_collection"
            assert call_kwargs["description"] == ""

    def test_drop_snapshot(self):
        """Test drop_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.drop_snapshot.return_value = None

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            client.drop_snapshot(snapshot_name="test_snapshot")

            mock_handler.drop_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot", timeout=None, context=ANY
            )

    def test_list_snapshots(self):
        """Test list_snapshots method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_snapshots.return_value = ["snapshot1", "snapshot2"]

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            snapshots = client.list_snapshots(collection_name="test_collection")

            assert snapshots == ["snapshot1", "snapshot2"]
            mock_handler.list_snapshots.assert_called_once_with(
                collection_name="test_collection", timeout=None, context=ANY
            )

    def test_list_snapshots_all(self):
        """Test list_snapshots for all collections."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_snapshots.return_value = ["snapshot1", "snapshot2", "snapshot3"]

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            snapshots = client.list_snapshots()

            assert len(snapshots) == 3
            mock_handler.list_snapshots.assert_called_once()
            call_kwargs = mock_handler.list_snapshots.call_args[1]
            assert call_kwargs["collection_name"] == ""

    def test_describe_snapshot(self):
        """Test describe_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.describe_snapshot.return_value = SnapshotInfo(
            name="test_snapshot",
            description="Test description",
            collection_name="test_collection",
            partition_names=["_default"],
            create_ts=1234567890,
            s3_location="s3://bucket/path",
        )

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            info = client.describe_snapshot(snapshot_name="test_snapshot")

            assert info.name == "test_snapshot"
            assert info.description == "Test description"
            assert info.collection_name == "test_collection"
            assert info.partition_names == ["_default"]
            assert info.create_ts == 1234567890
            assert info.s3_location == "s3://bucket/path"

            mock_handler.describe_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot", timeout=None, context=ANY
            )

    def test_restore_snapshot(self):
        """Test restore_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.restore_snapshot.return_value = 12345

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            job_id = client.restore_snapshot(
                snapshot_name="test_snapshot", collection_name="restored_collection"
            )

            assert job_id == 12345
            mock_handler.restore_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot",
                collection_name="restored_collection",
                rewrite_data=False,
                timeout=None,
                context=ANY,
            )

    def test_get_restore_snapshot_state(self):
        """Test get_restore_snapshot_state method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.get_restore_snapshot_state.return_value = RestoreSnapshotJobInfo(
            job_id=12345,
            snapshot_name="test_snapshot",
            db_name="default",
            collection_name="test_collection",
            state="RestoreSnapshotCompleted",
            progress=100,
            reason="",
            start_time=1234567890,
            time_cost=5000,
        )

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            state = client.get_restore_snapshot_state(job_id=12345)

            assert state.job_id == 12345
            assert state.snapshot_name == "test_snapshot"
            assert state.collection_name == "test_collection"
            assert state.state == "RestoreSnapshotCompleted"
            assert state.progress == 100
            assert state.time_cost == 5000

            mock_handler.get_restore_snapshot_state.assert_called_once_with(
                job_id=12345, timeout=None, context=ANY
            )

    def test_list_restore_snapshot_jobs(self):
        """Test list_restore_snapshot_jobs method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_restore_snapshot_jobs.return_value = [
            RestoreSnapshotJobInfo(
                job_id=12345,
                snapshot_name="snapshot1",
                db_name="default",
                collection_name="collection1",
                state="RestoreSnapshotCompleted",
                progress=100,
                reason="",
                start_time=1234567890,
                time_cost=5000,
            ),
            RestoreSnapshotJobInfo(
                job_id=12346,
                snapshot_name="snapshot2",
                db_name="default",
                collection_name="collection2",
                state="RestoreSnapshotExecuting",
                progress=50,
                reason="",
                start_time=1234567890,
                time_cost=2500,
            ),
        ]

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            jobs = client.list_restore_snapshot_jobs(collection_name="test_collection")

            assert len(jobs) == 2
            assert jobs[0].job_id == 12345
            assert jobs[0].progress == 100
            assert jobs[1].job_id == 12346
            assert jobs[1].progress == 50

            mock_handler.list_restore_snapshot_jobs.assert_called_once_with(
                collection_name="test_collection", timeout=None, context=ANY
            )

    def test_list_restore_snapshot_jobs_all(self):
        """Test list_restore_snapshot_jobs for all collections."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_restore_snapshot_jobs.return_value = []

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()

            jobs = client.list_restore_snapshot_jobs()

            assert len(jobs) == 0
            mock_handler.list_restore_snapshot_jobs.assert_called_once()
            call_kwargs = mock_handler.list_restore_snapshot_jobs.call_args[1]
            assert call_kwargs["collection_name"] == ""

    def test_client_db_isolation(self):
        """
        Test that two clients sharing the same connection but using different databases
        remain isolated when one switches database.
        """
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client_a = MilvusClient(uri="http://localhost:19530", db_name="default")
            client_b = MilvusClient(uri="http://localhost:19530", db_name="testdb")

            assert client_a._config.db_name == "default"
            assert client_b._config.db_name == "testdb"

            # Mock describe_database to simulate that 'db1' exists
            # use_database now validates database existence by calling describe_database
            with patch.object(client_a, "describe_database", return_value={}):
                client_a.use_database("db1")

            assert client_a._config.db_name == "db1"
            assert client_b._config.db_name == "testdb"

            client_b.list_collections()

            assert mock_handler.list_collections.called
            _, kwargs = mock_handler.list_collections.call_args
            context = kwargs.get("context")

            assert context is not None
            assert context.get_db_name() == "testdb"

    @pytest.mark.parametrize(
        "uri, db_name, expected_db_name",
        [
            # Issue #3236: db_name passed in URI path should be used when no explicit db_name
            ("http://localhost:19530/test_db", "", "test_db"),
            ("http://localhost:19530/production_db", "", "production_db"),
            ("https://localhost:19530/test_db", "", "test_db"),
            ("http://localhost:19530/mydb", "", "mydb"),
            # URI ending with slash should still extract db_name correctly
            ("http://localhost:19530/mydb/", "", "mydb"),
            ("https://localhost:19530/test_db/", "", "test_db"),
            # Mixed scenarios: explicit db_name takes precedence over URI path
            ("http://localhost:19530/uri_db", "explicit_db", "explicit_db"),
            ("http://localhost:19530/uri_db/", "explicit_db", "explicit_db"),
            # URI without path, no explicit db_name (should remain empty)
            ("http://localhost:19530", "", ""),
            ("https://localhost:19530", "", ""),
            # Multiple path segments - only first should be used as db_name
            ("http://localhost:19530/db1/collection1", "", "db1"),
            ("http://localhost:19530/db1/collection1/", "", "db1"),
            # Empty path segments should be handled correctly
            ("http://localhost:19530//", "", ""),
            ("http://localhost:19530///", "", ""),
        ],
    )
    def test_milvus_client_extract_db_name_from_uri(
        self, uri: str, db_name: str, expected_db_name: str
    ):
        """
        Test that MilvusClient extracts db_name from URI path when db_name is not explicitly provided.
        This fixes issue #3236: v2.6.7 db name do not work
        """
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient(uri=uri, db_name=db_name)
            assert client._config.db_name == expected_db_name, (
                f"Expected db_name to be '{expected_db_name}', "
                f"but got '{client._config.db_name}' for uri='{uri}' and db_name='{db_name}'"
            )

            # Verify that the extracted db_name is used in requests (only if db_name was extracted)
            if expected_db_name:
                client.list_collections()
                assert mock_handler.list_collections.called
                _, kwargs = mock_handler.list_collections.call_args
                context = kwargs.get("context")
                assert context is not None
                assert context.get_db_name() == expected_db_name


# (First-wave test classes removed — unique tests migrated to second-wave classes below)


def _make_handler(**overrides):
    handler = MagicMock()
    handler.get_server_type.return_value = "milvus"
    handler._wait_for_channel_ready = MagicMock()
    for k, v in overrides.items():
        setattr(handler, k, v)
    return handler


@pytest.fixture
def mc():
    """Yield (client, handler) with a mocked GrpcHandler."""
    handler = _make_handler()
    with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
        yield MilvusClient(), handler


class TestMilvusClientCreateCollection:
    def test_create_collection_no_schema_routes_to_fast(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch.object(client, "_fast_create_collection") as mock_fast:
                client.create_collection("col", dimension=128)
                mock_fast.assert_called_once()

    def test_create_collection_with_schema_routes_to_schema_method(self):

        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            schema = MilvusClient.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
            with patch.object(client, "_create_collection_with_schema") as mock_schema:
                client.create_collection("col", schema=schema)
                mock_schema.assert_called_once()

    @pytest.mark.parametrize(
        "id_type",
        ["int", "string", "str", DataType.VARCHAR, DataType.INT64],
    )
    def test_fast_create_collection_valid_id_types(self, id_type):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch.object(client, "create_index"), patch.object(client, "load_collection"):
                client._fast_create_collection("col", dimension=128, id_type=id_type)
            handler.create_collection.assert_called_once()

    def test_fast_create_collection_invalid_id_type(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(PrimaryKeyException):
                client._fast_create_collection("col", dimension=128, id_type="invalid")

    def test_fast_create_collection_varchar_with_max_length(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch.object(client, "create_index"), patch.object(client, "load_collection"):
                client._fast_create_collection(
                    "col", dimension=128, id_type="string", max_length=256
                )
            handler.create_collection.assert_called_once()

    def test_create_collection_with_schema_no_index_params(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            schema = MilvusClient.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
            client._create_collection_with_schema("col", schema, index_params=None)
            handler.create_collection.assert_called_once()

    def test_create_collection_with_schema_with_index_params(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            schema = MilvusClient.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
            index_params = MilvusClient.prepare_index_params(
                "vec", index_type="FLAT", metric_type="L2"
            )
            with patch.object(client, "create_index"), patch.object(client, "load_collection"):
                client._create_collection_with_schema("col", schema, index_params=index_params)
            handler.create_collection.assert_called_once()

    def test_create_index_iterates_params(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index("vec1", index_type="FLAT", metric_type="L2")
            index_params.add_index("vec2", index_type="FLAT", metric_type="L2")
            client.create_index("col", index_params)
            assert handler.create_index.call_count == 2


class TestMilvusClientCRUD:
    def test_insert_dict_converts_to_list(self):
        result = MagicMock()
        result.insert_count = 1
        result.primary_keys = [1]
        result.cost = 0
        handler = _make_handler(insert_rows=MagicMock(return_value=result))
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            ret = client.insert("col", data={"id": 1, "vec": [0.1, 0.2]})
            assert ret["insert_count"] == 1

    def test_insert_empty_list_returns_early(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            ret = client.insert("col", data=[])
            assert ret == {"insert_count": 0, "ids": []}
            handler.insert_rows.assert_not_called()

    def test_insert_invalid_type_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(TypeError):
                client.insert("col", data="invalid")

    def test_insert_exception_propagates(self):
        handler = _make_handler()
        handler.insert_rows.side_effect = RuntimeError("insert failed")
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(RuntimeError, match="insert failed"):
                client.insert("col", data=[{"id": 1}])

    def test_upsert_dict_converts_to_list(self):
        result = MagicMock()
        result.upsert_count = 1
        result.primary_keys = [1]
        result.cost = 0
        handler = _make_handler(upsert_rows=MagicMock(return_value=result))
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            ret = client.upsert("col", data={"id": 1, "vec": [0.1, 0.2]})
            assert ret["upsert_count"] == 1

    def test_upsert_empty_list_returns_early(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            ret = client.upsert("col", data=[])
            assert ret == {"upsert_count": 0, "ids": []}

    def test_upsert_invalid_type_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(TypeError):
                client.upsert("col", data=42)

    def test_upsert_exception_propagates(self):
        handler = _make_handler()
        handler.upsert_rows.side_effect = RuntimeError("upsert failed")
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(RuntimeError, match="upsert failed"):
                client.upsert("col", data=[{"id": 1}])

    def test_query_filter_and_ids_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(ParamError):
                client.query("col", filter="id > 0", ids=[1, 2])

    def test_query_non_string_filter_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(DataTypeNotMatchException):
                client.query("col", filter=123)

    def test_query_ids_as_int_converts_to_list(self):
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler = _make_handler()
        handler.query.return_value = [{"id": 42}]
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.query("col", ids=42)
            assert result == [{"id": 42}]

    def test_query_ids_as_str_converts_to_list(self):
        schema = {"fields": [{"name": "pk", "is_primary": True, "type": DataType.VARCHAR}]}
        handler = _make_handler()
        handler.query.return_value = [{"pk": "abc"}]
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.query("col", ids="abc")
            assert result == [{"pk": "abc"}]

    def test_delete_ids_as_int(self):
        result = MagicMock()
        result.primary_keys = []
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler = _make_handler()
        handler.delete.return_value = result
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.delete("col", ids=42)
            handler.delete.assert_called_once()

    def test_delete_ids_as_list(self):
        result = MagicMock()
        result.primary_keys = []
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler = _make_handler()
        handler.delete.return_value = result
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.delete("col", ids=[1, 2, 3])
            handler.delete.assert_called_once()

    def test_delete_invalid_ids_type_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(TypeError):
                client.delete("col", ids=3.14)

    def test_delete_invalid_id_in_list_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(TypeError):
                client.delete("col", ids=[1, 3.14])

    def test_delete_filter_and_pks_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(ParamError):
                client.delete("col", ids=[1], filter="id > 0")

    def test_delete_with_filter_string(self):
        result = MagicMock()
        result.primary_keys = []
        handler = _make_handler()
        handler.delete.return_value = result
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.delete("col", filter="id > 0")
            handler.delete.assert_called_once()


class TestMilvusClientCollectionDetails:
    def test_describe_collection_struct_array_fields_converted(self):
        handler = _make_handler()
        handler.describe_collection.return_value = {
            "fields": [{"name": "id"}],
            "struct_array_fields": [{"name": "struct", "type": "STRUCT", "fields": []}],
        }
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.describe_collection("col")
            assert "struct_array_fields" not in result

    def test_describe_collection_no_struct_array_fields(self):
        handler = _make_handler()
        handler.describe_collection.return_value = {"fields": [{"name": "id"}]}
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.describe_collection("col")
            assert result == {"fields": [{"name": "id"}]}

    def test_truncate_collection(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.truncate_collection("col")
            handler.truncate_collection.assert_called_once()

    def test_rename_collection(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.rename_collection("old", "new")
            handler.rename_collections.assert_called_once()

    def test_get_load_state_loading_returns_progress(self):
        handler = _make_handler()
        handler.get_load_state.return_value = LoadState.Loading
        handler.get_loading_progress.return_value = 75
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.get_load_state("col")
            assert result["state"] == LoadState.Loading
            assert result["progress"] == 75

    def test_get_load_state_loaded_no_progress(self):
        handler = _make_handler()
        handler.get_load_state.return_value = LoadState.Loaded
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.get_load_state("col")
            assert result["state"] == LoadState.Loaded
            assert "progress" not in result

    def test_get_load_state_with_partition_name(self):
        handler = _make_handler()
        handler.get_load_state.return_value = LoadState.Loaded
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.get_load_state("col", partition_name="part1")
            args, _ = handler.get_load_state.call_args
            # partition_names is the 2nd positional argument
            assert args[1] == ["part1"]

    def test_refresh_load(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.refresh_load("col")
            _, kwargs = handler.load_collection.call_args
            assert kwargs.get("_refresh") is True

    def test_load_partitions_string_converts_to_list(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.load_partitions("col", "part1")
            args, _ = handler.load_partitions.call_args
            # partition_names is the 2nd positional arg
            assert args[1] == ["part1"]

    def test_load_partitions_list_stays_list(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.load_partitions("col", ["part1", "part2"])
            handler.load_partitions.assert_called_once()

    def test_release_partitions_string_converts_to_list(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.release_partitions("col", "part1")
            handler.release_partitions.assert_called_once()


class TestMilvusClientOptimize:
    def test_optimize_wait_true_returns_result(self):
        expected = OptimizeResult(
            status="success",
            collection_name="col",
            compaction_id=1,
            target_size="1GB",
            progress=[],
        )
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch.object(client, "_execute_optimize", return_value=expected):
                result = client.optimize("col", target_size="1GB", wait=True, timeout=5.0)
                assert result == expected

    def test_optimize_wait_false_returns_task(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch.object(client, "_execute_optimize", side_effect=lambda **kw: None):
                task = client.optimize("col", wait=False)
                assert isinstance(task, OptimizeTask)
                task.join(timeout=1.0)


class TestMilvusClientSearchOps:
    def test_hybrid_search_delegates(self):
        handler = _make_handler()
        handler.hybrid_search.return_value = []
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.hybrid_search("col", reqs=[], ranker=MagicMock())
            handler.hybrid_search.assert_called_once()

    def test_search_basic_delegates(self):
        handler = _make_handler()
        handler.search.return_value = []
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.search("col", data=[[0.1, 0.2]])
            handler.search.assert_called_once()

    def test_get_empty_list_ids_returns_early(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.get("col", ids=[])
            assert result == []
            handler.query.assert_not_called()

    def test_get_scalar_id_converts_to_list(self):
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler = _make_handler()
        handler.query.return_value = [{"id": 1}]
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.get("col", ids=1)
            handler.query.assert_called_once()

    def test_query_iterator_delegates(self):
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        handler = _make_handler()
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch("pymilvus.milvus_client.milvus_client.QueryIterator") as mock_qi:
                mock_qi.return_value = MagicMock()
                client.query_iterator("col", filter="id > 0")
                mock_qi.assert_called_once()

    def test_query_iterator_invalid_filter_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(DataTypeNotMatchException):
                client.query_iterator("col", filter=123)

    def test_search_iterator_delegates_v2(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch("pymilvus.milvus_client.milvus_client.SearchIteratorV2") as mock_v2:
                mock_v2.return_value = MagicMock()
                client.search_iterator("col", data=[[0.1, 0.2]])
                mock_v2.assert_called_once()


class TestMilvusClientDeleteBranches:
    def test_delete_with_pks_kwarg_int(self):
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        result = MagicMock()
        result.primary_keys = []
        result.delete_count = 1
        result.cost = 0
        handler = _make_handler()
        handler.delete.return_value = result
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.delete("col", pks=1)
            handler.delete.assert_called_once()

    def test_delete_pks_kwarg_invalid_type_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(TypeError):
                client.delete("col", pks=3.14)

    def test_delete_returns_primary_keys_when_present(self):
        schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        result = MagicMock()
        result.primary_keys = [1, 2, 3]
        handler = _make_handler()
        handler.delete.return_value = result
        handler._get_schema.return_value = (schema, 100)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            ret = client.delete("col", ids=[1, 2, 3])
            assert ret == [1, 2, 3]

    def test_delete_filter_non_str_raises(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(DataTypeNotMatchException):
                client.delete("col", filter=123)


# ============================================================
# Parametrized simple delegation tests
# ============================================================

# Each tuple: (client_method, args, kwargs, handler_method)
_SIMPLE_DELEGATION_CASES = [
    # Collection ops (migrated from first-wave)
    ("drop_collection", ("col",), {}, "drop_collection"),
    ("has_collection", ("col",), {}, "has_collection"),
    ("truncate_collection", ("col",), {}, "truncate_collection"),
    ("rename_collection", ("old", "new"), {}, "rename_collections"),
    # Partition ops (migrated from first-wave)
    ("create_partition", ("col", "part"), {}, "create_partition"),
    ("drop_partition", ("col", "part"), {}, "drop_partition"),
    ("has_partition", ("col", "part"), {}, "has_partition"),
    # Index ops (migrated from first-wave)
    ("drop_index", ("col", "idx"), {}, "drop_index"),
    # Alias ops
    ("create_alias", ("col", "alias1"), {}, "create_alias"),
    ("drop_alias", ("alias1",), {}, "drop_alias"),
    ("alter_alias", ("col", "alias1"), {}, "alter_alias"),
    # User/role ops (migrated from first-wave)
    ("create_user", ("user", "pass"), {}, "create_user"),
    ("drop_user", ("user",), {}, "delete_user"),
    ("create_role", ("role",), {}, "create_role"),
    ("drop_role", ("role",), {}, "drop_role"),
    ("grant_role", ("user", "role"), {}, "add_user_to_role"),
    ("revoke_role", ("user", "role"), {}, "remove_user_from_role"),
    # Privilege ops
    ("grant_privilege", ("admin", "Collection", "Insert", "col"), {}, "grant_privilege"),
    ("revoke_privilege", ("admin", "Collection", "Insert", "col"), {}, "revoke_privilege"),
    ("grant_privilege_v2", ("admin", "Insert", "col"), {}, "grant_privilege_v2"),
    ("revoke_privilege_v2", ("admin", "Insert", "col"), {}, "revoke_privilege_v2"),
    # Privilege groups
    ("create_privilege_group", ("grp",), {}, "create_privilege_group"),
    ("drop_privilege_group", ("grp",), {}, "drop_privilege_group"),
    ("add_privileges_to_group", ("grp", ["Insert"]), {}, "add_privileges_to_group"),
    ("remove_privileges_from_group", ("grp", ["Insert"]), {}, "remove_privileges_from_group"),
    # Resource groups
    ("create_resource_group", ("rg1",), {}, "create_resource_group"),
    ("drop_resource_group", ("rg1",), {}, "drop_resource_group"),
    ("update_resource_groups", ({},), {}, "update_resource_groups"),
    ("transfer_replica", ("rg1", "rg2", "col", 1), {}, "transfer_replica"),
    # Database ops
    ("create_database", ("mydb",), {}, "create_database"),
    ("drop_database", ("mydb",), {}, "drop_database"),
    ("drop_database_properties", ("mydb", ["key"]), {}, "drop_database_properties"),
    # Index management
    (
        "alter_index_properties",
        ("col", "idx", {"mmap.enabled": True}),
        {},
        "alter_index_properties",
    ),
    ("drop_index_properties", ("col", "idx", ["mmap.enabled"]), {}, "drop_index_properties"),
    # Collection properties
    ("alter_collection_properties", ("col", {"key": "val"}), {}, "alter_collection_properties"),
    ("drop_collection_properties", ("col", ["key"]), {}, "drop_collection_properties"),
    # Misc ops
    ("flush", ("col",), {}, "flush"),
    ("flush_all", (), {}, "flush_all"),
    ("load_collection", ("col",), {}, "load_collection"),
    ("release_collection", ("col",), {}, "release_collection"),
    # File resources
    ("add_file_resource", ("res1", "/path"), {}, "add_file_resource"),
    ("remove_file_resource", ("res1",), {}, "remove_file_resource"),
    ("list_file_resources", (), {}, "list_file_resources"),
    # Misc delegation (migrated from MiscOps)
    ("get_compaction_plans", (42,), {}, "get_compaction_plans"),
    ("run_analyzer", ("hello world",), {}, "run_analyzer"),
    ("update_replicate_configuration", (), {"clusters": []}, "update_replicate_configuration"),
    ("alter_database_properties", ("mydb", {"key": "val"}), {}, "alter_database"),
    ("describe_alias", ("alias1",), {}, "describe_alias"),
    ("describe_resource_group", ("rg1",), {}, "describe_resource_group"),
    ("describe_replica", ("col",), {}, "describe_replica"),
]


class TestMilvusClientSimpleDelegation:
    @pytest.mark.parametrize(
        "method,args,kwargs,handler_method",
        _SIMPLE_DELEGATION_CASES,
        ids=[c[0] for c in _SIMPLE_DELEGATION_CASES],
    )
    def test_delegation(self, method, args, kwargs, handler_method):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            getattr(client, method)(*args, **kwargs)
            getattr(handler, handler_method).assert_called_once()


# ============================================================
# Non-trivial tests that need custom setup/assertions
# ============================================================


class TestMilvusClientCollectionMgmt:
    def test_alter_collection_field(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.alter_collection_field("col", "field", {"mmap.enabled": True})
            handler.alter_collection_field_properties.assert_called_once()

    def test_get_load_state_exception_reraises(self):
        handler = _make_handler()
        handler.get_load_state.side_effect = MilvusException(message="load error")
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with pytest.raises(MilvusException):
                client.get_load_state("col")

    def test_list_indexes_with_field_filter(self):
        idx = MagicMock()
        idx.field_name = "vec"
        idx.index_name = "vec_idx"
        handler = _make_handler()
        handler.list_indexes.return_value = [idx]
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.list_indexes("col", field_name="vec")
            assert "vec_idx" in result

    def test_list_indexes_skips_none_index(self):
        handler = _make_handler()
        handler.list_indexes.return_value = [None]
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            result = client.list_indexes("col")
            assert result == []

    def test_list_collections_returns_list(self):
        handler = _make_handler()
        handler.list_collections.return_value = ["coll1", "coll2"]
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            assert client.list_collections() == ["coll1", "coll2"]

    def test_list_partitions_returns_list(self):
        handler = _make_handler()
        handler.list_partitions.return_value = ["_default", "part1"]
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            assert client.list_partitions("col") == ["_default", "part1"]

    def test_list_users_returns_list(self):
        handler = _make_handler()
        handler.list_usernames.return_value = ["root", "user1"]
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            assert client.list_users() == ["root", "user1"]

    def test_get_collection_stats(self):
        stat = MagicMock()
        stat.key = "row_count"
        stat.value = "1000"
        handler = _make_handler()
        handler.get_collection_stats.return_value = [stat]
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            assert client.get_collection_stats("col") == {"row_count": 1000}

    def test_list_vector_indexes(self):
        schema = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "vector", "type": DataType.FLOAT_VECTOR},
            ]
        }
        handler = _make_handler()
        handler._get_schema.return_value = (schema, 12345)
        idx = MagicMock()
        idx.field_name = "vector"
        idx.index_name = "vec_index"
        handler.list_indexes.return_value = [idx]
        handler.describe_index.return_value = {"field_name": "vector", "index_name": "vec_index"}
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            assert client._list_vector_indexes("col") == ["vec_index"]

    def test_list_vector_indexes_no_vector_fields(self):
        schema = {
            "fields": [
                {"name": "id", "type": DataType.INT64},
                {"name": "text", "type": DataType.VARCHAR},
            ]
        }
        handler = _make_handler()
        handler._get_schema.return_value = (schema, 12345)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            assert client._list_vector_indexes("col") == []

    def test_close_releases_connection(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            manager = client._manager
            with patch.object(manager, "release") as mock_release:
                client.close()
                mock_release.assert_called_once_with(handler, client=client)

    def test_add_collection_function_delegates(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.add_collection_function("col", MagicMock())
            handler.add_collection_function.assert_called_once()

    def test_alter_collection_function_delegates(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.alter_collection_function("col", "fn", MagicMock())
            handler.alter_collection_function.assert_called_once()

    def test_drop_collection_function_delegates(self):
        handler = _make_handler()
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            client.drop_collection_function("col", "fn")
            handler.drop_collection_function.assert_called_once()


class TestMilvusClientMiscOps:
    def test_compact_returns_id(self, mc):
        client, handler = mc
        handler.compact.return_value = 42
        assert client.compact("col") == 42

    def test_get_compaction_state(self, mc):
        client, handler = mc
        state = MagicMock()
        state.state_name = "Completed"
        handler.get_compaction_state.return_value = state
        assert client.get_compaction_state(42) == "Completed"

    def test_get_server_version(self, mc):
        client, handler = mc
        handler.get_server_version.return_value = "2.6.0"
        assert client.get_server_version() == "2.6.0"

    def test_get_flush_all_state(self, mc):
        client, handler = mc
        handler.get_flush_all_state.return_value = True
        assert client.get_flush_all_state() is True

    def test_list_loaded_segments(self, mc):
        client, handler = mc
        handler.get_query_segment_info.return_value = []
        assert client.list_loaded_segments("col") == []

    def test_list_persistent_segments(self, mc):
        client, handler = mc
        handler.get_persistent_segment_infos.return_value = []
        assert client.list_persistent_segments("col") == []

    def test_using_database(self, mc):
        client, handler = mc
        handler.describe_database.return_value = {"db_name": "mydb"}
        client.using_database("mydb")
        assert client._config.db_name == "mydb"

    def test_list_databases(self, mc):
        client, handler = mc
        handler.list_database.return_value = ["default", "mydb"]
        assert "mydb" in client.list_databases()

    def test_list_aliases(self, mc):
        client, handler = mc
        handler.list_aliases.return_value = ["alias1"]
        assert client.list_aliases("col") == ["alias1"]

    def test_list_resource_groups(self, mc):
        client, handler = mc
        handler.list_resource_groups.return_value = ["rg1"]
        assert "rg1" in client.list_resource_groups()


class TestMilvusClientRBACOps:
    def test_describe_user_with_groups(self, mc):
        client, handler = mc
        group = MagicMock()
        group.roles = ["admin"]
        res = MagicMock()
        res.groups = [group]
        handler.select_one_user.return_value = res
        result = client.describe_user("alice")
        assert result["user_name"] == "alice"
        assert result["roles"] == ["admin"]

    def test_describe_user_no_groups(self, mc):
        client, handler = mc
        res = MagicMock()
        res.groups = []
        handler.select_one_user.return_value = res
        assert client.describe_user("alice") == {}

    def test_describe_user_exception_reraises(self, mc):
        client, handler = mc
        handler.select_one_user.side_effect = MilvusException(message="not found")
        with pytest.raises(MilvusException):
            client.describe_user("alice")

    def test_describe_role_returns_privileges(self, mc):
        client, handler = mc
        res = MagicMock()
        res.groups = []
        handler.select_grant_for_one_role.return_value = res
        result = client.describe_role("admin")
        assert result["role"] == "admin"
        assert result["privileges"] == []

    def test_describe_role_exception_reraises(self, mc):
        client, handler = mc
        handler.select_grant_for_one_role.side_effect = MilvusException(message="err")
        with pytest.raises(MilvusException):
            client.describe_role("admin")

    def test_list_roles(self, mc):
        client, handler = mc
        g1 = MagicMock()
        g1.role_name = "admin"
        res = MagicMock()
        res.groups = [g1]
        handler.select_all_role.return_value = res
        assert "admin" in client.list_roles()

    def test_list_roles_exception_reraises(self, mc):
        client, handler = mc
        handler.select_all_role.side_effect = MilvusException(message="err")
        with pytest.raises(MilvusException):
            client.list_roles()

    def test_update_password_with_reset_connection(self, mc):
        client, handler = mc
        client.update_password("user", "old", "new", reset_connection=True)
        handler.update_password.assert_called_once()
        handler._setup_authorization_interceptor.assert_called_once()
        handler._setup_grpc_channel.assert_called_once()

    def test_list_privilege_groups(self, mc):
        client, handler = mc
        grp = MagicMock()
        grp.privilege_group = "grp1"
        grp.privileges = ["Insert"]
        res = MagicMock()
        res.groups = [grp]
        handler.list_privilege_groups.return_value = res
        result = client.list_privilege_groups()
        assert len(result) == 1
        assert result[0]["privilege_group"] == "grp1"


class TestMilvusClientGetPartitionStats:
    def test_get_partition_stats(self, mc):
        client, handler = mc
        stat = MagicMock()
        stat.key = "row_count"
        stat.value = "100"
        handler.get_partition_stats.return_value = [stat]
        assert client.get_partition_stats("col", "part") == {"row_count": 100}

    def test_get_partition_stats_invalid_type_raises(self, mc):
        client, _handler = mc
        with pytest.raises(TypeError):
            client.get_partition_stats("col", 123)


class TestMilvusClientInternalOps:
    def test_wait_for_indexes_empty_returns_early(self, mc):
        client, handler = mc
        task = MagicMock()
        client._wait_for_indexes(task, "col", [])
        handler.wait_for_creating_index.assert_not_called()

    def test_wait_for_indexes_calls_wait(self, mc):
        client, handler = mc
        task = MagicMock()
        task.check_cancelled = MagicMock()
        client._wait_for_indexes(task, "col", ["vec_idx"])
        handler.wait_for_creating_index.assert_called_once()

    def test_wait_for_compaction_completes(self, mc):
        client, handler = mc
        state = MagicMock()
        state.state = 2
        handler.get_compaction_state.return_value = state
        task = MagicMock()
        task.check_cancelled = MagicMock()
        client._wait_for_compaction_with_cancel(task, 42)
        handler.get_compaction_state.assert_called_once()

    def test_wait_for_compaction_failed_raises(self, mc):
        client, handler = mc
        state = MagicMock()
        state.state = 3
        handler.get_compaction_state.return_value = state
        task = MagicMock()
        task.check_cancelled = MagicMock()
        with pytest.raises(MilvusException, match="Compaction 42 failed"):
            client._wait_for_compaction_with_cancel(task, 42)

    def test_is_collection_loaded_true(self, mc):
        client, handler = mc
        handler.get_load_state.return_value = LoadState.Loaded
        assert client._is_collection_loaded("col") is True

    def test_execute_optimize_success(self):
        handler = _make_handler()
        handler.compact.return_value = 99
        state = MagicMock()
        state.state = 2
        handler.get_compaction_state.return_value = state
        handler.get_load_state.return_value = LoadState.NotLoad
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            client = MilvusClient()
            with patch.object(client, "_list_vector_indexes", return_value=[]), patch.object(
                client, "_wait_for_indexes"
            ):
                task = MagicMock()
                task.check_cancelled = MagicMock()
                task.set_progress = MagicMock()
                task.progress_history = MagicMock(return_value=[])
                task._target_size = None
                result = client._execute_optimize(task, "col", None, None)
                assert result.collection_name == "col"


class TestFileResourceMethods(TestMilvusClient):
    """Tests for file_resource API methods with context passing."""

    def test_add_file_resource(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.add_file_resource.return_value = None

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            client.add_file_resource(name="test_file", path="/data/test.csv")
            mock_handler.add_file_resource.assert_called_once()
            call_kwargs = mock_handler.add_file_resource.call_args
            assert call_kwargs.kwargs["name"] == "test_file"
            assert call_kwargs.kwargs["path"] == "/data/test.csv"
            assert "context" in call_kwargs.kwargs

    def test_add_file_resource_with_timeout(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.add_file_resource.return_value = None

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            client.add_file_resource(name="f", path="/p", timeout=30)
            call_kwargs = mock_handler.add_file_resource.call_args
            assert call_kwargs.kwargs["timeout"] == 30

    def test_remove_file_resource(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.remove_file_resource.return_value = None

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            client.remove_file_resource(name="test_file")
            mock_handler.remove_file_resource.assert_called_once()
            call_kwargs = mock_handler.remove_file_resource.call_args
            assert call_kwargs.kwargs["name"] == "test_file"
            assert "context" in call_kwargs.kwargs

    def test_list_file_resources(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_file_resources.return_value = ["file1", "file2"]

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_file_resources()
            mock_handler.list_file_resources.assert_called_once()
            assert result == ["file1", "file2"]
            assert "context" in mock_handler.list_file_resources.call_args.kwargs


class TestMilvusClientExternalCollection:
    """Test external collection refresh APIs in MilvusClient."""

    def test_refresh_external_collection(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.refresh_external_collection.return_value = 42

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            result = client.refresh_external_collection(collection_name="ext_coll")

            assert result == 42
            mock_handler.refresh_external_collection.assert_called_once_with(
                collection_name="ext_coll",
                timeout=None,
                context=ANY,
                external_source="",
                external_spec="",
            )

    def test_refresh_with_new_source(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.refresh_external_collection.return_value = 43

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            result = client.refresh_external_collection(
                collection_name="ext_coll",
                external_source="s3://new-path",
                external_spec='{"format": "iceberg"}',
            )

            assert result == 43
            call_kwargs = mock_handler.refresh_external_collection.call_args[1]
            assert call_kwargs["external_source"] == "s3://new-path"
            assert call_kwargs["external_spec"] == '{"format": "iceberg"}'

    def test_get_refresh_progress(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.get_refresh_external_collection_progress.return_value = (
            RefreshExternalCollectionJobInfo(
                job_id=42,
                collection_name="ext_coll",
                state="RefreshCompleted",
                progress=100,
                reason="",
                external_source="s3://bucket",
                start_time=1000,
                end_time=2000,
            )
        )

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            result = client.get_refresh_external_collection_progress(job_id=42)

            assert result.job_id == 42
            assert result.state == "RefreshCompleted"
            assert result.progress == 100
            mock_handler.get_refresh_external_collection_progress.assert_called_once_with(
                job_id=42, timeout=None, context=ANY
            )

    def test_list_refresh_jobs(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_refresh_external_collection_jobs.return_value = [
            RefreshExternalCollectionJobInfo(
                job_id=1,
                collection_name="ext_coll",
                state="RefreshCompleted",
                progress=100,
                reason="",
                external_source="s3://a",
                start_time=1000,
                end_time=2000,
            ),
        ]

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_refresh_external_collection_jobs(collection_name="ext_coll")

            assert len(result) == 1
            assert result[0].job_id == 1
            mock_handler.list_refresh_external_collection_jobs.assert_called_once_with(
                collection_name="ext_coll", timeout=None, context=ANY
            )

    def test_list_refresh_jobs_all(self):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler._wait_for_channel_ready = MagicMock()
        mock_handler.list_refresh_external_collection_jobs.return_value = []

        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_refresh_external_collection_jobs()

            assert result == []
            call_kwargs = mock_handler.list_refresh_external_collection_jobs.call_args[1]
            assert call_kwargs["collection_name"] == ""
