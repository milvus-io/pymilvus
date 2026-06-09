"""Unit tests for pymilvus.orm.partition.Partition."""

from unittest import mock
from unittest.mock import MagicMock

import pytest
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Partition,
    connections,
)
from pymilvus.exceptions import MilvusException

from .conftest import GRPC_PREFIX

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _schema():
    return CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
    )


@pytest.fixture
def partition(mock_grpc_connect, mock_grpc_close, _schema):
    """Return a Partition backed by a mocked gRPC connection."""
    connections.connect(keep_alive=False)
    with mock.patch(f"{GRPC_PREFIX}.create_collection"), mock.patch(
        f"{GRPC_PREFIX}.has_collection", return_value=False
    ):
        coll = Collection("test_coll", schema=_schema)
    with mock.patch(f"{GRPC_PREFIX}.create_partition"), mock.patch(
        f"{GRPC_PREFIX}.has_partition", return_value=False
    ), mock.patch(
        f"{GRPC_PREFIX}.describe_collection",
        return_value={
            "schema": _schema.to_dict(),
            "collection_name": "test_coll",
        },
    ):
        part = Partition(coll, "test_part", description="test desc")
    yield part
    connections.disconnect("default")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestPartitionInit:
    def test_create_with_collection_object(self, partition):
        """Partition created via the fixture should have correct attributes."""
        assert partition.name == "test_part"
        assert partition._collection.name == "test_coll"

    def test_create_with_string_collection(self, mock_grpc_connect, mock_grpc_close, _schema):
        """Partition accepts a collection name string instead of an object."""
        connections.connect(keep_alive=False)
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection",
            return_value=_schema.to_dict(),
        ), mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=False), mock.patch(
            f"{GRPC_PREFIX}.create_partition"
        ):
            part = Partition("test_coll", "str_part")
        assert part.name == "str_part"
        assert part._collection.name == "test_coll"
        connections.disconnect("default")

    def test_create_with_invalid_collection_type(self, mock_grpc_connect, mock_grpc_close):
        """Passing an invalid type for collection raises MilvusException."""
        connections.connect(keep_alive=False)
        with pytest.raises(
            MilvusException,
            match=r"Collection must be of type pymilvus\.Collection or String",
        ):
            Partition(12345, "bad_part")
        connections.disconnect("default")

    def test_create_existing_partition_skips_create(
        self, mock_grpc_connect, mock_grpc_close, _schema
    ):
        """When the partition already exists, create_partition is NOT called."""
        connections.connect(keep_alive=False)
        with mock.patch(f"{GRPC_PREFIX}.create_collection"), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=False
        ):
            coll = Collection("test_coll", schema=_schema)
        with mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.create_partition"
        ) as m_create, mock.patch(
            f"{GRPC_PREFIX}.describe_collection",
            return_value={
                "schema": _schema.to_dict(),
                "collection_name": "test_coll",
            },
        ):
            Partition(coll, "existing_part")
        m_create.assert_not_called()
        connections.disconnect("default")

    def test_construct_only_skips_creation(self, mock_grpc_connect, mock_grpc_close, _schema):
        """construct_only=True skips has_partition/create_partition calls."""
        connections.connect(keep_alive=False)
        with mock.patch(f"{GRPC_PREFIX}.create_collection"), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=False
        ):
            coll = Collection("test_coll", schema=_schema)
        with mock.patch(f"{GRPC_PREFIX}.has_partition") as m_has, mock.patch(
            f"{GRPC_PREFIX}.create_partition"
        ) as m_create:
            Partition(coll, "lazy_part", construct_only=True)
        m_has.assert_not_called()
        m_create.assert_not_called()
        connections.disconnect("default")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestPartitionProperties:
    def test_name(self, partition):
        assert partition.name == "test_part"

    def test_description(self, partition):
        assert partition.description == "test desc"

    def test_is_empty_true(self, partition):
        fake_stat = MagicMock()
        fake_stat.key = "row_count"
        fake_stat.value = "0"
        with mock.patch(
            f"{GRPC_PREFIX}.get_partition_stats",
            return_value=[fake_stat],
        ):
            assert partition.is_empty is True

    def test_is_empty_false(self, partition):
        fake_stat = MagicMock()
        fake_stat.key = "row_count"
        fake_stat.value = "5"
        with mock.patch(
            f"{GRPC_PREFIX}.get_partition_stats",
            return_value=[fake_stat],
        ):
            assert partition.is_empty is False

    def test_num_entities(self, partition):
        fake_stat = MagicMock()
        fake_stat.key = "row_count"
        fake_stat.value = "42"
        with mock.patch(
            f"{GRPC_PREFIX}.get_partition_stats",
            return_value=[fake_stat],
        ) as m:
            assert partition.num_entities == 42
            m.assert_called_once()

    def test_repr(self, partition):
        r = repr(partition)
        assert "test_part" in r
        assert "test_coll" in r
        assert "test desc" in r


# ---------------------------------------------------------------------------
# Mutation methods
# ---------------------------------------------------------------------------


class TestPartitionDrop:
    def test_drop(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.drop_partition", return_value=None) as m:
            partition.drop()
            m.assert_called_once()

    def test_drop_with_timeout(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.drop_partition", return_value=None) as m:
            partition.drop(timeout=5.0)
            m.assert_called_once()
            call_kwargs = m.call_args
            assert call_kwargs[1]["timeout"] == 5.0


class TestPartitionFlush:
    def test_flush(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.flush", return_value=None) as m:
            partition.flush()
            m.assert_called_once()

    def test_flush_with_timeout(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.flush", return_value=None) as m:
            partition.flush(timeout=3.0)
            call_kwargs = m.call_args
            assert call_kwargs[1]["timeout"] == 3.0


class TestPartitionLoad:
    def test_load(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.load_partitions", return_value=None) as m:
            partition.load()
            m.assert_called_once()
            call_kwargs = m.call_args
            assert call_kwargs[1]["partition_names"] == ["test_part"]

    def test_load_with_replica_number(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.load_partitions", return_value=None) as m:
            partition.load(replica_number=2, timeout=10.0)
            call_kwargs = m.call_args
            assert call_kwargs[1]["replica_number"] == 2
            assert call_kwargs[1]["timeout"] == 10.0


class TestPartitionRelease:
    def test_release(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.release_partitions", return_value=None) as m:
            partition.release()
            m.assert_called_once()
            call_kwargs = m.call_args
            assert call_kwargs[1]["partition_names"] == ["test_part"]

    def test_release_with_timeout(self, partition):
        with mock.patch(f"{GRPC_PREFIX}.release_partitions", return_value=None) as m:
            partition.release(timeout=2.5)
            call_kwargs = m.call_args
            assert call_kwargs[1]["timeout"] == 2.5


class TestPartitionInsert:
    def test_insert(self, partition):
        data = [[1, 2], [[0.1] * 128, [0.2] * 128]]
        fake_res = MagicMock()
        fake_res.insert_count = 2
        with mock.patch(f"{GRPC_PREFIX}.batch_insert", return_value=fake_res) as m:
            partition.insert(data)
            m.assert_called_once()

    def test_insert_with_timeout(self, partition):
        data = [[1], [[0.1] * 128]]
        fake_res = MagicMock()
        fake_res.insert_count = 1
        with mock.patch(f"{GRPC_PREFIX}.batch_insert", return_value=fake_res):
            result = partition.insert(data, timeout=5.0)
            assert result is not None


class TestPartitionDelete:
    def test_delete(self, partition):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.delete", return_value=fake_res):
            result = partition.delete("id in [1, 2]")
            assert result is not None

    def test_delete_with_timeout(self, partition):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.delete", return_value=fake_res):
            result = partition.delete("id in [1]", timeout=3.0)
            assert result is not None


class TestPartitionUpsert:
    def test_upsert(self, partition):
        data = [[1, 2], [[0.1] * 128, [0.2] * 128]]
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.upsert", return_value=fake_res) as m:
            partition.upsert(data)
            m.assert_called_once()

    def test_upsert_with_timeout(self, partition):
        data = [[1], [[0.1] * 128]]
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.upsert", return_value=fake_res):
            result = partition.upsert(data, timeout=5.0)
            assert result is not None


# ---------------------------------------------------------------------------
# Search & query
# ---------------------------------------------------------------------------


class TestPartitionSearch:
    def test_search(self, partition):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.search", return_value=fake_res) as m:
            partition.search(
                data=[[0.1] * 128],
                anns_field="vec",
                param={"metric_type": "L2"},
                limit=10,
            )
            m.assert_called_once()
            call_kwargs = m.call_args
            assert call_kwargs[1]["partition_names"] == ["test_part"]

    def test_search_with_expr_and_output(self, partition):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.search", return_value=fake_res):
            result = partition.search(
                data=[[0.1] * 128],
                anns_field="vec",
                param={"metric_type": "L2"},
                limit=5,
                expr="id > 0",
                output_fields=["id"],
                timeout=10.0,
                round_decimal=2,
            )
            assert result is not None

    def test_search_empty_data(self, partition):
        result = partition.search(
            data=[],
            anns_field="vec",
            param={"metric_type": "L2"},
            limit=10,
        )
        assert result is not None


class TestPartitionHybridSearch:
    def test_hybrid_search(self, partition):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.hybrid_search", return_value=fake_res) as m:
            result = partition.hybrid_search(
                reqs=[MagicMock()],
                rerank=MagicMock(),
                limit=10,
            )
            m.assert_called_once()
            # partition_names is passed as a positional arg
            pos_args = m.call_args[0]
            assert ["test_part"] in pos_args
            assert result == fake_res

    def test_hybrid_search_empty_reqs(self, partition):
        result = partition.hybrid_search(reqs=[], rerank=None, limit=10)
        assert result is not None


class TestPartitionQuery:
    def test_query(self, partition):
        fake_res = [{"id": 1}]
        with mock.patch(f"{GRPC_PREFIX}.query", return_value=fake_res) as m:
            result = partition.query("id > 0")
            m.assert_called_once()
            # partition_names is passed as a positional arg
            pos_args = m.call_args[0]
            assert ["test_part"] in pos_args
            assert result == fake_res

    def test_query_with_output_fields(self, partition):
        fake_res = [{"id": 1, "vec": [0.1] * 128}]
        with mock.patch(f"{GRPC_PREFIX}.query", return_value=fake_res):
            result = partition.query(
                "id > 0",
                output_fields=["id", "vec"],
                timeout=5.0,
            )
            assert result == fake_res


# ---------------------------------------------------------------------------
# Replicas
# ---------------------------------------------------------------------------


class TestPartitionGetReplicas:
    def test_get_replicas(self, partition):
        fake_replica = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_replicas", return_value=fake_replica) as m:
            result = partition.get_replicas()
            m.assert_called_once()
            assert result == fake_replica

    def test_get_replicas_with_timeout(self, partition):
        fake_replica = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_replicas", return_value=fake_replica):
            result = partition.get_replicas(timeout=3.0)
            assert result == fake_replica
