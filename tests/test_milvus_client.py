import logging
from unittest.mock import MagicMock, patch

import pytest
from pymilvus import DataType
from pymilvus.client.types import RestoreSnapshotJobInfo, SnapshotInfo
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParams
from pymilvus.milvus_client.milvus_client import MilvusClient

log = logging.getLogger(__name__)


class TestMilvusClient:
    @pytest.mark.parametrize("index_params", [None, {}, "str", MilvusClient.prepare_index_params()])
    def test_create_index_invalid_params(self, index_params):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch("pymilvus.orm.connections.Connections.connect", return_value=None), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ):
            client = MilvusClient()
            assert client._using == "http://localhost:19530"
            client = MilvusClient(user="test", password="foobar")
            assert client._using == "http://localhost:19530-test"
            client = MilvusClient(token="foobar")
            assert client._using == "http://localhost:19530-3858f62230ac3c915f300c664312c63f"

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

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
        mock_conn = MagicMock()

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch.object(
            MilvusClient, "_get_connection", return_value=mock_conn
        ):
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
        mock_conn = MagicMock()

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch.object(
            MilvusClient, "_get_connection", return_value=mock_conn
        ):
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
        mock_handler.create_snapshot.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
            )

    def test_create_snapshot_minimal(self):
        """Test create_snapshot with minimal parameters."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_snapshot.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
        mock_handler.drop_snapshot.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            client.drop_snapshot(snapshot_name="test_snapshot")

            mock_handler.drop_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot", timeout=None
            )

    def test_list_snapshots(self):
        """Test list_snapshots method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_snapshots.return_value = ["snapshot1", "snapshot2"]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            snapshots = client.list_snapshots(collection_name="test_collection")

            assert snapshots == ["snapshot1", "snapshot2"]
            mock_handler.list_snapshots.assert_called_once_with(
                collection_name="test_collection", timeout=None
            )

    def test_list_snapshots_all(self):
        """Test list_snapshots for all collections."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_snapshots.return_value = ["snapshot1", "snapshot2", "snapshot3"]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
        mock_handler.describe_snapshot.return_value = SnapshotInfo(
            name="test_snapshot",
            description="Test description",
            collection_name="test_collection",
            partition_names=["_default"],
            create_ts=1234567890,
            s3_location="s3://bucket/path",
        )

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            info = client.describe_snapshot(snapshot_name="test_snapshot")

            assert info.name == "test_snapshot"
            assert info.description == "Test description"
            assert info.collection_name == "test_collection"
            assert info.partition_names == ["_default"]
            assert info.create_ts == 1234567890
            assert info.s3_location == "s3://bucket/path"

            mock_handler.describe_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot", timeout=None
            )

    def test_restore_snapshot(self):
        """Test restore_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.restore_snapshot.return_value = 12345

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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
            )

    def test_get_restore_snapshot_state(self):
        """Test get_restore_snapshot_state method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
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

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            state = client.get_restore_snapshot_state(job_id=12345)

            assert state.job_id == 12345
            assert state.snapshot_name == "test_snapshot"
            assert state.collection_name == "test_collection"
            assert state.state == "RestoreSnapshotCompleted"
            assert state.progress == 100
            assert state.time_cost == 5000

            mock_handler.get_restore_snapshot_state.assert_called_once_with(
                job_id=12345, timeout=None
            )

    def test_list_restore_snapshot_jobs(self):
        """Test list_restore_snapshot_jobs method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
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

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            jobs = client.list_restore_snapshot_jobs(collection_name="test_collection")

            assert len(jobs) == 2
            assert jobs[0].job_id == 12345
            assert jobs[0].progress == 100
            assert jobs[1].job_id == 12346
            assert jobs[1].progress == 50

            mock_handler.list_restore_snapshot_jobs.assert_called_once_with(
                collection_name="test_collection", timeout=None
            )

    def test_list_restore_snapshot_jobs_all(self):
        """Test list_restore_snapshot_jobs for all collections."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_restore_snapshot_jobs.return_value = []

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
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

        with patch(
            "pymilvus.milvus_client._utils.create_connection", return_value="shared_alias"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch(
            "pymilvus.orm.connections.Connections.has_connection", return_value=True
        ):
            client_a = MilvusClient(uri="http://localhost:19530", db_name="default")
            client_b = MilvusClient(uri="http://localhost:19530", db_name="testdb")

            assert client_a._db_name == "default"
            assert client_b._db_name == "testdb"

            # Mock describe_database to simulate that 'db1' exists
            # use_database now validates database existence by calling describe_database
            with patch.object(client_a, "describe_database", return_value={}):
                client_a.use_database("db1")

            assert client_a._db_name == "db1"
            assert client_b._db_name == "testdb"

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

        with patch(
            "pymilvus.milvus_client._utils.create_connection", return_value="test_alias"
        ), patch(
            "pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler
        ), patch(
            "pymilvus.orm.connections.Connections.has_connection", return_value=False
        ), patch(
            "pymilvus.orm.connections.Connections.connect"
        ), patch.object(
            MilvusClient, "get_server_type", return_value="milvus"
        ):
            client = MilvusClient(uri=uri, db_name=db_name)
            assert client._db_name == expected_db_name, (
                f"Expected db_name to be '{expected_db_name}', "
                f"but got '{client._db_name}' for uri='{uri}' and db_name='{db_name}'"
            )

            # Verify that the extracted db_name is used in requests (only if db_name was extracted)
            if expected_db_name:
                client.list_collections()
                assert mock_handler.list_collections.called
                _, kwargs = mock_handler.list_collections.call_args
                context = kwargs.get("context")
                assert context is not None
                assert context.get_db_name() == expected_db_name


# ============================================================
# MilvusClient Collection Operations Tests
# ============================================================


class TestMilvusClientCollectionOps:
    """Tests for MilvusClient collection operations."""

    def test_drop_collection(self):
        """Test drop_collection method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.drop_collection.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.drop_collection("test_collection")

            mock_handler.drop_collection.assert_called_once()

    def test_has_collection(self):
        """Test has_collection method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.has_collection.return_value = True

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.has_collection("test_collection")

            assert result is True
            mock_handler.has_collection.assert_called_once()

    def test_list_collections(self):
        """Test list_collections method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_collections.return_value = ["coll1", "coll2", "coll3"]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_collections()

            assert result == ["coll1", "coll2", "coll3"]

    def test_get_collection_stats(self):
        """Test get_collection_stats method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        # Method expects list of objects with .key and .value attributes
        mock_stat = MagicMock()
        mock_stat.key = "row_count"
        mock_stat.value = "1000"
        mock_handler.get_collection_stats.return_value = [mock_stat]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.get_collection_stats("test_collection")

            assert result == {"row_count": 1000}


# ============================================================
# MilvusClient Partition Operations Tests
# ============================================================


class TestMilvusClientPartitionOps:
    """Tests for MilvusClient partition operations."""

    def test_create_partition(self):
        """Test create_partition method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_partition.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.create_partition("test_collection", "test_partition")

            mock_handler.create_partition.assert_called_once()

    def test_drop_partition(self):
        """Test drop_partition method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.drop_partition.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.drop_partition("test_collection", "test_partition")

            mock_handler.drop_partition.assert_called_once()

    def test_has_partition(self):
        """Test has_partition method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.has_partition.return_value = True

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.has_partition("test_collection", "test_partition")

            assert result is True

    def test_list_partitions(self):
        """Test list_partitions method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_partitions.return_value = ["_default", "partition1"]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_partitions("test_collection")

            assert result == ["_default", "partition1"]


# ============================================================
# MilvusClient Index Operations Tests
# ============================================================


class TestMilvusClientIndexOps:
    """Tests for MilvusClient index operations."""

    def test_drop_index(self):
        """Test drop_index method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.drop_index.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.drop_index("test_collection", "test_index")

            mock_handler.drop_index.assert_called_once()

    def test_list_indexes(self):
        """Test list_indexes method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        # Method expects list of objects with .field_name and .index_name attributes
        mock_index1 = MagicMock()
        mock_index1.field_name = "vector"
        mock_index1.index_name = "index1"
        mock_index2 = MagicMock()
        mock_index2.field_name = "vector"
        mock_index2.index_name = "index2"
        mock_handler.list_indexes.return_value = [mock_index1, mock_index2]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_indexes("test_collection")

            assert result == ["index1", "index2"]


# ============================================================
# MilvusClient Alias Operations Tests
# ============================================================


class TestMilvusClientAliasOps:
    """Tests for MilvusClient alias operations."""

    def test_create_alias(self):
        """Test create_alias method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_alias.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.create_alias("test_collection", "test_alias")

            mock_handler.create_alias.assert_called_once()

    def test_drop_alias(self):
        """Test drop_alias method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.drop_alias.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.drop_alias("test_alias")

            mock_handler.drop_alias.assert_called_once()

    def test_alter_alias(self):
        """Test alter_alias method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.alter_alias.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.alter_alias("test_collection", "test_alias")

            mock_handler.alter_alias.assert_called_once()


# ============================================================
# MilvusClient User/Role Operations Tests
# ============================================================


class TestMilvusClientUserOps:
    """Tests for MilvusClient user and role operations."""

    def test_create_user(self):
        """Test create_user method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_user.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.create_user("test_user", "password123")

            mock_handler.create_user.assert_called_once()

    def test_drop_user(self):
        """Test drop_user method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.delete_user.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.drop_user("test_user")

            mock_handler.delete_user.assert_called_once()

    def test_list_users(self):
        """Test list_users method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_usernames.return_value = ["root", "user1"]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.list_users()

            assert result == ["root", "user1"]

    def test_create_role(self):
        """Test create_role method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_role.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.create_role("test_role")

            mock_handler.create_role.assert_called_once()

    def test_drop_role(self):
        """Test drop_role method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.drop_role.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.drop_role("test_role")

            mock_handler.drop_role.assert_called_once()

    def test_grant_role(self):
        """Test grant_role method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.add_user_to_role.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.grant_role("test_user", "test_role")

            mock_handler.add_user_to_role.assert_called_once()

    def test_revoke_role(self):
        """Test revoke_role method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.remove_user_from_role.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.revoke_role("test_user", "test_role")

            mock_handler.remove_user_from_role.assert_called_once()


# ============================================================
# MilvusClient Utility Operations Tests
# ============================================================


# ============================================================
# MilvusClient Data Operations Tests
# ============================================================


class TestMilvusClientDataOps:
    """Tests for MilvusClient data operations."""

    def test_insert_single_dict(self):
        """Test insert with single dict."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_result = MagicMock()
        mock_result.insert_count = 1
        mock_result.primary_keys = [1]
        mock_result.cost = 0
        mock_handler.insert_rows.return_value = mock_result

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.insert("test_collection", {"id": 1, "vector": [0.1, 0.2]})

            assert result["insert_count"] == 1
            assert result["ids"] == [1]

    def test_insert_list_of_dicts(self):
        """Test insert with list of dicts."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_result = MagicMock()
        mock_result.insert_count = 2
        mock_result.primary_keys = [1, 2]
        mock_result.cost = 0
        mock_handler.insert_rows.return_value = mock_result

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            data = [{"id": 1, "vector": [0.1, 0.2]}, {"id": 2, "vector": [0.3, 0.4]}]
            result = client.insert("test_collection", data)

            assert result["insert_count"] == 2

    def test_insert_empty_data(self):
        """Test insert with empty data returns zero count."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.insert("test_collection", [])

            assert result["insert_count"] == 0
            assert result["ids"] == []

    def test_insert_invalid_type(self):
        """Test insert with invalid type raises TypeError."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            with pytest.raises(TypeError, match="wrong type of argument"):
                client.insert("test_collection", "invalid")

    def test_upsert_single_dict(self):
        """Test upsert with single dict."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_result = MagicMock()
        mock_result.upsert_count = 1
        mock_result.primary_keys = [1]
        mock_result.cost = 0
        mock_handler.upsert_rows.return_value = mock_result

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.upsert("test_collection", {"id": 1, "vector": [0.1, 0.2]})

            assert result["upsert_count"] == 1

    def test_upsert_empty_data(self):
        """Test upsert with empty data returns zero count."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.upsert("test_collection", [])

            assert result["upsert_count"] == 0
            assert result["ids"] == []

    def test_upsert_invalid_type(self):
        """Test upsert with invalid type raises TypeError."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()

            with pytest.raises(TypeError, match="wrong type of argument"):
                client.upsert("test_collection", "invalid")

    def test_delete_with_ids(self):
        """Test delete with IDs."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_result = MagicMock()
        mock_result.delete_count = 2
        mock_result.cost = 0
        mock_result.primary_keys = []
        mock_handler.delete.return_value = mock_result
        # Mock _get_schema to return schema with primary key field
        mock_schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        mock_handler._get_schema.return_value = (mock_schema, 100)

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.delete("test_collection", ids=[1, 2])

            assert result["delete_count"] == 2

    def test_delete_with_filter(self):
        """Test delete with filter expression."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_result = MagicMock()
        mock_result.delete_count = 5
        mock_result.cost = 0
        mock_result.primary_keys = []
        mock_handler.delete.return_value = mock_result

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.delete("test_collection", filter="age > 20")

            assert result["delete_count"] == 5

    def test_search(self):
        """Test search method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.search.return_value = [[{"id": 1, "distance": 0.1}]]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.search("test_collection", data=[[0.1, 0.2]])

            assert result == [[{"id": 1, "distance": 0.1}]]
            mock_handler.search.assert_called_once()

    def test_query(self):
        """Test query method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.query.return_value = [{"id": 1, "name": "test"}]

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.query("test_collection", filter="id > 0")

            assert result == [{"id": 1, "name": "test"}]
            mock_handler.query.assert_called_once()

    def test_get(self):
        """Test get method (query by ids)."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.query.return_value = [{"id": 1, "name": "test"}]
        # Mock _get_schema to return schema with primary key field
        mock_schema = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}
        mock_handler._get_schema.return_value = (mock_schema, 100)

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.get("test_collection", ids=[1])

            assert result == [{"id": 1, "name": "test"}]


class TestMilvusClientUtilityOps:
    """Tests for MilvusClient utility operations."""

    def test_flush(self):
        """Test flush method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.flush.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.flush("test_collection")

            mock_handler.flush.assert_called_once()

    def test_get_load_state(self):
        """Test get_load_state method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.get_load_state.return_value = {"state": "Loaded"}

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            result = client.get_load_state("test_collection")

            assert result is not None

    def test_load_collection(self):
        """Test load_collection method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.load_collection.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.load_collection("test_collection")

            mock_handler.load_collection.assert_called_once()

    def test_release_collection(self):
        """Test release_collection method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.release_collection.return_value = None

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            client.release_collection("test_collection")

            mock_handler.release_collection.assert_called_once()

    def test_close(self):
        """Test close method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"

        with patch(
            "pymilvus.milvus_client.milvus_client.create_connection", return_value="test"
        ), patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            with patch("pymilvus.orm.connections.Connections.disconnect") as mock_disconnect:
                client = MilvusClient()
                client.close()

                mock_disconnect.assert_called_once()
