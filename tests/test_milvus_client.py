import logging
from unittest.mock import MagicMock, patch

import pytest
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParams
from pymilvus.milvus_client.milvus_client import MilvusClient

log = logging.getLogger(__name__)


class TestMilvusClient:
    @pytest.mark.parametrize("index_params", [None, {}, "str", MilvusClient.prepare_index_params()])
    def test_create_index_invalid_params(self, index_params):
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        
        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            if isinstance(index_params, IndexParams):
                with pytest.raises(ParamError, match="IndexParams is empty, no index can be created"):
                    client.create_index("test_collection", index_params)
            elif index_params is None:
                with pytest.raises(ParamError, match="missing required argument:.*"):
                    client.create_index("test_collection", index_params)
            else:
                with pytest.raises(ParamError, match="wrong type of argument .*"):
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
        
        with patch("pymilvus.orm.connections.Connections.connect", return_value=None), \
             patch("pymilvus.orm.connections.Connections._fetch_handler", return_value=mock_handler):
            client = MilvusClient()
            assert client._using == "http://localhost:19530"
            client = MilvusClient(user="test", password="foobar")
            assert client._using == "http://localhost:19530-test"
            client = MilvusClient(token="foobar")
            assert client._using == "http://localhost:19530-3858f62230ac3c915f300c664312c63f"


class TestMilvusClientSnapshot:
    """Test snapshot-related APIs in MilvusClient."""

    def test_create_snapshot(self):
        """Test create_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_snapshot.return_value = None

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            client.create_snapshot(
                collection_name="test_collection",
                snapshot_name="test_snapshot",
                description="Test description"
            )

            mock_handler.create_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot",
                db_name="",
                collection_name="test_collection",
                description="Test description",
                timeout=None
            )

    def test_create_snapshot_minimal(self):
        """Test create_snapshot with minimal parameters."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.create_snapshot.return_value = None

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            client.create_snapshot(
                collection_name="test_collection",
                snapshot_name="test_snapshot"
            )

            mock_handler.create_snapshot.assert_called_once()
            call_kwargs = mock_handler.create_snapshot.call_args[1]
            assert call_kwargs['snapshot_name'] == "test_snapshot"
            assert call_kwargs['collection_name'] == "test_collection"
            assert call_kwargs['description'] == ""

    def test_drop_snapshot(self):
        """Test drop_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.drop_snapshot.return_value = None

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            client.drop_snapshot(snapshot_name="test_snapshot")

            mock_handler.drop_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot",
                timeout=None
            )

    def test_list_snapshots(self):
        """Test list_snapshots method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_snapshots.return_value = ["snapshot1", "snapshot2"]

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            snapshots = client.list_snapshots(collection_name="test_collection")

            assert snapshots == ["snapshot1", "snapshot2"]
            mock_handler.list_snapshots.assert_called_once_with(
                db_name="",
                collection_name="test_collection",
                timeout=None
            )

    def test_list_snapshots_all(self):
        """Test list_snapshots for all collections."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_snapshots.return_value = ["snapshot1", "snapshot2", "snapshot3"]

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            snapshots = client.list_snapshots()

            assert len(snapshots) == 3
            mock_handler.list_snapshots.assert_called_once()
            call_kwargs = mock_handler.list_snapshots.call_args[1]
            assert call_kwargs['collection_name'] == ""

    def test_describe_snapshot(self):
        """Test describe_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.describe_snapshot.return_value = {
            "name": "test_snapshot",
            "description": "Test description",
            "collection_name": "test_collection",
            "partition_names": ["_default"],
            "create_ts": 1234567890,
            "s3_location": "s3://bucket/path"
        }

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            info = client.describe_snapshot(snapshot_name="test_snapshot")

            assert info["name"] == "test_snapshot"
            assert info["description"] == "Test description"
            assert info["collection_name"] == "test_collection"
            assert info["partition_names"] == ["_default"]
            assert info["create_ts"] == 1234567890
            assert info["s3_location"] == "s3://bucket/path"

            mock_handler.describe_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot",
                timeout=None
            )

    def test_restore_snapshot(self):
        """Test restore_snapshot method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.restore_snapshot.return_value = 12345

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            job_id = client.restore_snapshot(
                snapshot_name="test_snapshot",
                collection_name="restored_collection"
            )

            assert job_id == 12345
            mock_handler.restore_snapshot.assert_called_once_with(
                snapshot_name="test_snapshot",
                db_name="",
                collection_name="restored_collection",
                rewrite_data=False,
                timeout=None
            )

    def test_get_restore_snapshot_state(self):
        """Test get_restore_snapshot_state method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.get_restore_snapshot_state.return_value = {
            "job_id": 12345,
            "snapshot_name": "test_snapshot",
            "collection_id": 100,
            "state": 2,  # Completed
            "progress": 100,
            "reason": "",
            "time_cost": 5000
        }

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            state = client.get_restore_snapshot_state(job_id=12345)

            assert state["job_id"] == 12345
            assert state["snapshot_name"] == "test_snapshot"
            assert state["collection_id"] == 100
            assert state["state"] == 2
            assert state["progress"] == 100
            assert state["time_cost"] == 5000

            mock_handler.get_restore_snapshot_state.assert_called_once_with(
                job_id=12345,
                timeout=None
            )

    def test_list_restore_snapshot_jobs(self):
        """Test list_restore_snapshot_jobs method."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_restore_snapshot_jobs.return_value = [
            {
                "job_id": 12345,
                "snapshot_name": "snapshot1",
                "collection_id": 100,
                "state": 2,
                "progress": 100,
                "reason": "",
                "time_cost": 5000
            },
            {
                "job_id": 12346,
                "snapshot_name": "snapshot2",
                "collection_id": 101,
                "state": 1,
                "progress": 50,
                "reason": "",
                "time_cost": 2500
            }
        ]

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            jobs = client.list_restore_snapshot_jobs(collection_name="test_collection")

            assert len(jobs) == 2
            assert jobs[0]["job_id"] == 12345
            assert jobs[0]["progress"] == 100
            assert jobs[1]["job_id"] == 12346
            assert jobs[1]["progress"] == 50

            mock_handler.list_restore_snapshot_jobs.assert_called_once_with(
                collection_name="test_collection",
                timeout=None
            )

    def test_list_restore_snapshot_jobs_all(self):
        """Test list_restore_snapshot_jobs for all collections."""
        mock_handler = MagicMock()
        mock_handler.get_server_type.return_value = "milvus"
        mock_handler.list_restore_snapshot_jobs.return_value = []

        with patch('pymilvus.milvus_client.milvus_client.create_connection', return_value="test"), \
             patch('pymilvus.orm.connections.Connections._fetch_handler', return_value=mock_handler):
            client = MilvusClient()

            jobs = client.list_restore_snapshot_jobs()

            assert len(jobs) == 0
            mock_handler.list_restore_snapshot_jobs.assert_called_once()
            call_kwargs = mock_handler.list_restore_snapshot_jobs.call_args[1]
            assert call_kwargs['collection_name'] == ""
