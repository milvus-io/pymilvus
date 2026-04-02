import json
from unittest import mock
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
from pymilvus.exceptions import (
    AutoIDException,
    DataTypeNotMatchException,
    DataTypeNotSupportException,
    IndexNotExistException,
    PartitionAlreadyExistException,
    SchemaNotReadyException,
)

from .conftest import GRPC_PREFIX


class TestCollectionLifecycle:
    """Test Collection create -> load-by-name -> drop lifecycle with mocked gRPC."""

    @pytest.fixture
    def collection_schema(self):
        fields = [
            FieldSchema("int64", DataType.INT64),
            FieldSchema("float", DataType.FLOAT),
            FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=128),
            FieldSchema("binary_vector", DataType.BINARY_VECTOR, dim=128),
            FieldSchema("float16_vector", DataType.FLOAT16_VECTOR, dim=128),
            FieldSchema("bfloat16_vector", DataType.BFLOAT16_VECTOR, dim=128),
            FieldSchema("int8_vector", DataType.INT8_VECTOR, dim=128),
            FieldSchema("timestamptz", DataType.TIMESTAMPTZ),
        ]
        return CollectionSchema(fields, primary_field="int64")

    def test_create_load_drop(self, mock_grpc_connect, mock_grpc_close, collection_schema):
        """Create a collection, reload by name, then drop it."""
        coll_name = "ut_collection_lifecycle"

        connections.connect(keep_alive=False)

        with mock.patch(f"{GRPC_PREFIX}.create_collection", return_value=None), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=False
        ):
            collection = Collection(name=coll_name, schema=collection_schema)

        with mock.patch(f"{GRPC_PREFIX}.create_collection", return_value=None), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=True
        ), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=collection_schema.to_dict()
        ):
            collection = Collection(name=coll_name)

        with mock.patch(f"{GRPC_PREFIX}.drop_collection", return_value=None), mock.patch(
            f"{GRPC_PREFIX}.list_indexes", return_value=[]
        ), mock.patch(f"{GRPC_PREFIX}.release_collection", return_value=None):
            collection.drop()

        connections.disconnect("default")

    def test_init_with_timeout_kwarg(self, mock_grpc_connect, mock_grpc_close, collection_schema):
        """Collection(timeout=...) should not raise TypeError for duplicate kwarg."""
        connections.connect(keep_alive=False)

        with mock.patch(f"{GRPC_PREFIX}.create_collection", return_value=None), mock.patch(
            f"{GRPC_PREFIX}.has_collection", return_value=False
        ):
            # Before the fix, this raised:
            # TypeError: has_collection() got multiple values for keyword argument 'timeout'
            collection = Collection(name="ut_timeout_kwarg", schema=collection_schema, timeout=10.0)

        assert collection.name == "ut_timeout_kwarg"

        connections.disconnect("default")

    def test_init_with_timeout_kwarg_existing_collection(
        self, mock_grpc_connect, mock_grpc_close, collection_schema
    ):
        """Loading an existing Collection with timeout kwarg should not raise TypeError."""
        connections.connect(keep_alive=False)

        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=collection_schema.to_dict()
        ):
            collection = Collection(name="ut_timeout_existing", timeout=5.0)

        assert collection.name == "ut_timeout_existing"

        connections.disconnect("default")


# ---------------------------------------------------------------------------
# Shared fixture: a connected Collection ready for method-level tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _schema():
    return CollectionSchema(
        [
            FieldSchema("pk", DataType.INT64, is_primary=True),
            FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
        ]
    )


@pytest.fixture
def collection(mock_grpc_connect, mock_grpc_close, _schema):
    """Return a Collection object backed by a mocked gRPC connection."""
    connections.connect(keep_alive=False)
    with mock.patch(f"{GRPC_PREFIX}.create_collection", return_value=None), mock.patch(
        f"{GRPC_PREFIX}.has_collection", return_value=False
    ):
        coll = Collection("test_coll", schema=_schema)
    yield coll
    connections.disconnect("default")


# ---------------------------------------------------------------------------
# Tests for Collection properties
# ---------------------------------------------------------------------------


class TestCollectionProperties:
    def test_name(self, collection):
        assert collection.name == "test_coll"

    def test_schema_property(self, collection):
        assert isinstance(collection.schema, CollectionSchema)

    def test_description(self, collection):
        assert isinstance(collection.description, str)

    def test_primary_field(self, collection):
        pf = collection.primary_field
        assert pf is not None
        assert pf.name == "pk"

    def test_repr(self, collection):
        r = repr(collection)
        assert "<Collection>" in r
        assert "test_coll" in r

    def test_num_entities(self, collection):
        fake_stat = MagicMock()
        fake_stat.key = "row_count"
        fake_stat.value = "42"
        with mock.patch(f"{GRPC_PREFIX}.get_collection_stats", return_value=[fake_stat]):
            assert collection.num_entities == 42

    def test_is_empty_true(self, collection):
        fake_stat = MagicMock()
        fake_stat.key = "row_count"
        fake_stat.value = "0"
        with mock.patch(f"{GRPC_PREFIX}.get_collection_stats", return_value=[fake_stat]):
            assert collection.is_empty is True

    def test_is_empty_false(self, collection):
        fake_stat = MagicMock()
        fake_stat.key = "row_count"
        fake_stat.value = "5"
        with mock.patch(f"{GRPC_PREFIX}.get_collection_stats", return_value=[fake_stat]):
            assert collection.is_empty is False


# ---------------------------------------------------------------------------
# Tests for Collection methods that delegate to conn
# ---------------------------------------------------------------------------


class TestCollectionDrop:
    def test_drop(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.drop_collection", return_value=None) as m:
            collection.drop()
            m.assert_called_once()


class TestCollectionFlush:
    def test_flush(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.flush", return_value=None) as m:
            collection.flush()
            m.assert_called_once()


class TestCollectionTruncate:
    def test_truncate(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.truncate_collection", return_value=None) as m:
            collection.truncate()
            m.assert_called_once()


class TestCollectionLoad:
    def test_load_collection(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.load_collection", return_value=None) as m:
            collection.load()
            m.assert_called_once()

    def test_load_partitions(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.load_partitions", return_value=None) as m:
            collection.load(partition_names=["p1", "p2"])
            m.assert_called_once()


class TestCollectionRelease:
    def test_release(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.release_collection", return_value=None) as m:
            collection.release()
            m.assert_called_once()


class TestCollectionSetProperties:
    def test_set_properties(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.alter_collection_properties", return_value=None) as m:
            collection.set_properties({"collection.ttl.seconds": 60})
            m.assert_called_once()


class TestCollectionInsert:
    def test_insert_column_based(self, collection):
        data = [[1, 2], [[0.1] * 128, [0.2] * 128]]
        fake_res = MagicMock()
        fake_res.insert_count = 2
        with mock.patch(f"{GRPC_PREFIX}.batch_insert", return_value=fake_res) as m:
            collection.insert(data)
            m.assert_called_once()

    def test_insert_row_based(self, collection):
        data = [{"pk": 1, "vec": [0.1] * 128}]
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.insert_rows", return_value=fake_res) as m:
            collection.insert(data)
            m.assert_called_once()

    def test_insert_invalid_type(self, collection):
        with pytest.raises(DataTypeNotSupportException):
            collection.insert("invalid_data")


class TestCollectionDelete:
    def test_delete(self, collection):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.delete", return_value=fake_res):
            result = collection.delete("pk in [1, 2]")
            assert result is not None

    def test_delete_async(self, collection):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.delete", return_value=fake_res):
            result = collection.delete("pk in [1]", _async=True)
            assert result is not None


class TestCollectionUpsert:
    def test_upsert_column_based(self, collection):
        data = [[1, 2], [[0.1] * 128, [0.2] * 128]]
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.upsert", return_value=fake_res) as m:
            collection.upsert(data)
            m.assert_called_once()

    def test_upsert_row_based(self, collection):
        data = [{"pk": 1, "vec": [0.1] * 128}]
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.upsert_rows", return_value=fake_res) as m:
            collection.upsert(data)
            m.assert_called_once()

    def test_upsert_invalid_type(self, collection):
        with pytest.raises(DataTypeNotSupportException):
            collection.upsert("invalid_data")


class TestCollectionSearch:
    def test_search(self, collection):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.search", return_value=fake_res) as m:
            collection.search(
                data=[[0.1] * 128],
                anns_field="vec",
                param={"metric_type": "L2"},
                limit=10,
            )
            m.assert_called_once()

    def test_search_empty_data(self, collection):
        result = collection.search(
            data=[],
            anns_field="vec",
            param={"metric_type": "L2"},
            limit=10,
        )
        # Empty data returns an empty SearchResult
        assert result is not None

    def test_search_invalid_expr_type(self, collection):
        with pytest.raises(DataTypeNotMatchException):
            collection.search(
                data=[[0.1] * 128],
                anns_field="vec",
                param={"metric_type": "L2"},
                limit=10,
                expr=123,
            )

    def test_search_async(self, collection):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.search", return_value=fake_res):
            result = collection.search(
                data=[[0.1] * 128],
                anns_field="vec",
                param={"metric_type": "L2"},
                limit=10,
                _async=True,
            )
            assert result is not None


class TestCollectionHybridSearch:
    def test_hybrid_search_empty_reqs(self, collection):
        result = collection.hybrid_search(reqs=[], rerank=None, limit=10)
        assert result is not None

    def test_hybrid_search(self, collection):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.hybrid_search", return_value=fake_res) as m:
            collection.hybrid_search(
                reqs=[MagicMock()],
                rerank=MagicMock(),
                limit=10,
            )
            m.assert_called_once()

    def test_hybrid_search_async(self, collection):
        fake_res = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.hybrid_search", return_value=fake_res):
            result = collection.hybrid_search(
                reqs=[MagicMock()],
                rerank=MagicMock(),
                limit=10,
                _async=True,
            )
            assert result is not None


class TestCollectionQuery:
    def test_query(self, collection):
        fake_res = [{"pk": 1}]
        with mock.patch(f"{GRPC_PREFIX}.query", return_value=fake_res) as m:
            result = collection.query("pk > 0")
            m.assert_called_once()
            assert result == fake_res

    def test_query_invalid_expr_type(self, collection):
        with pytest.raises(DataTypeNotMatchException):
            collection.query(123)


class TestCollectionPartitions:
    def test_partitions(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.list_partitions", return_value=["_default", "p1"]):
            parts = collection.partitions
            assert len(parts) == 2

    def test_partition_exists(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=True):
            p = collection.partition("_default")
            assert p is not None

    def test_partition_not_exists(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=False):
            p = collection.partition("nonexistent")
            assert p is None

    def test_has_partition(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=True) as m:
            result = collection.has_partition("_default")
            assert result is True
            m.assert_called_once()

    def test_drop_partition(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.drop_partition", return_value=None) as m:
            collection.drop_partition("p1")
            m.assert_called_once()

    def test_create_partition_new(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=False), mock.patch(
            f"{GRPC_PREFIX}.create_partition", return_value=None
        ):
            p = collection.create_partition("new_part")
            assert p is not None

    def test_create_partition_already_exists(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.has_partition", return_value=True):
            with pytest.raises(PartitionAlreadyExistException):
                collection.create_partition("existing_part")


class TestCollectionIndexes:
    def test_indexes_empty(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.list_indexes", return_value=[]):
            assert collection.indexes == []

    def test_indexes_with_data(self, collection):
        fake_kv = MagicMock()
        fake_kv.key = "index_type"
        fake_kv.value = "IVF_FLAT"
        fake_index = MagicMock()
        fake_index.params = [fake_kv]
        fake_index.field_name = "vec"
        fake_index.index_name = "idx"
        with mock.patch(f"{GRPC_PREFIX}.list_indexes", return_value=[fake_index]):
            indexes = collection.indexes
            assert len(indexes) == 1

    def test_indexes_with_json_params(self, collection):
        kv_type = MagicMock()
        kv_type.key = "index_type"
        kv_type.value = "IVF_FLAT"
        kv_params = MagicMock()
        kv_params.key = "params"
        kv_params.value = json.dumps({"nlist": 128})
        fake_index = MagicMock()
        fake_index.params = [kv_type, kv_params]
        fake_index.field_name = "vec"
        fake_index.index_name = "idx"
        with mock.patch(f"{GRPC_PREFIX}.list_indexes", return_value=[fake_index]):
            indexes = collection.indexes
            assert len(indexes) == 1

    def test_index_found(self, collection):
        fake_desc = {
            "field_name": "vec",
            "index_name": "idx",
            "total_rows": 100,
            "indexed_rows": 100,
            "pending_index_rows": 0,
            "state": "Finished",
            "index_type": "IVF_FLAT",
        }
        with mock.patch(f"{GRPC_PREFIX}.describe_index", return_value=fake_desc):
            idx = collection.index(index_name="idx")
            assert idx is not None

    def test_index_not_found(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.describe_index", return_value=None):
            with pytest.raises(IndexNotExistException):
                collection.index()

    def test_create_index(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.create_index", return_value=None) as m:
            collection.create_index("vec", {"index_type": "FLAT", "metric_type": "L2"})
            m.assert_called_once()

    def test_has_index_true(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.describe_index", return_value={"index_type": "FLAT"}):
            assert collection.has_index() is True

    def test_has_index_false(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.describe_index", return_value=None):
            assert collection.has_index() is False

    def test_drop_index_exists(self, collection):
        fake_desc = {"field_name": "vec", "index_name": "idx"}
        with mock.patch(f"{GRPC_PREFIX}.describe_index", return_value=fake_desc), mock.patch(
            f"{GRPC_PREFIX}.drop_index", return_value=None
        ) as m:
            collection.drop_index(index_name="idx")
            m.assert_called_once()

    def test_drop_index_not_exists(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.describe_index", return_value=None):
            # Should not raise; silently does nothing
            collection.drop_index()

    def test_alter_index(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.alter_index_properties", return_value=None) as m:
            collection.alter_index("idx", {"mmap.enabled": True})
            m.assert_called_once()


class TestCollectionCompact:
    def test_compact(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.compact", return_value=123) as m:
            collection.compact()
            m.assert_called_once()
            assert collection.compaction_id == 123

    def test_compact_clustering(self, collection):
        with mock.patch(f"{GRPC_PREFIX}.compact", return_value=456) as m:
            collection.compact(is_clustering=True)
            m.assert_called_once()
            assert collection.clustering_compaction_id == 456

    def test_get_compaction_state(self, collection):
        collection.compaction_id = 123
        fake_state = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_compaction_state", return_value=fake_state) as m:
            result = collection.get_compaction_state()
            m.assert_called_once()
            assert result == fake_state

    def test_get_compaction_state_clustering(self, collection):
        collection.clustering_compaction_id = 456
        fake_state = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_compaction_state", return_value=fake_state) as m:
            result = collection.get_compaction_state(is_clustering=True)
            m.assert_called_once()
            assert result == fake_state

    def test_get_compaction_plans(self, collection):
        collection.compaction_id = 123
        fake_plans = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_compaction_plans", return_value=fake_plans) as m:
            result = collection.get_compaction_plans()
            m.assert_called_once()
            assert result == fake_plans

    def test_get_compaction_plans_clustering(self, collection):
        collection.clustering_compaction_id = 456
        fake_plans = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_compaction_plans", return_value=fake_plans) as m:
            result = collection.get_compaction_plans(is_clustering=True)
            m.assert_called_once()
            assert result == fake_plans

    def test_wait_for_compaction_completed(self, collection):
        collection.compaction_id = 123
        fake_state = MagicMock()
        with mock.patch(
            f"{GRPC_PREFIX}.wait_for_compaction_completed", return_value=fake_state
        ) as m:
            collection.wait_for_compaction_completed()
            m.assert_called_once()

    def test_wait_for_compaction_completed_clustering(self, collection):
        collection.clustering_compaction_id = 456
        fake_state = MagicMock()
        with mock.patch(
            f"{GRPC_PREFIX}.wait_for_compaction_completed", return_value=fake_state
        ) as m:
            collection.wait_for_compaction_completed(is_clustering=True)
            m.assert_called_once()


class TestCollectionMisc:
    def test_get_replicas(self, collection):
        fake_replica = MagicMock()
        with mock.patch(f"{GRPC_PREFIX}.get_replicas", return_value=fake_replica) as m:
            result = collection.get_replicas()
            m.assert_called_once()
            assert result == fake_replica

    def test_describe(self, collection):
        fake_desc = {"name": "test_coll", "schema": {}}
        with mock.patch(f"{GRPC_PREFIX}.describe_collection", return_value=fake_desc) as m:
            result = collection.describe()
            m.assert_called_once()
            assert result == fake_desc

    def test_aliases(self, collection):
        fake_desc = {"aliases": ["alias1", "alias2"]}
        with mock.patch(f"{GRPC_PREFIX}.describe_collection", return_value=fake_desc):
            assert collection.aliases == ["alias1", "alias2"]

    def test_num_shards(self, collection):
        fake_desc = {"num_shards": 2}
        with mock.patch(f"{GRPC_PREFIX}.describe_collection", return_value=fake_desc):
            assert collection.num_shards == 2


class TestCollectionInitEdgeCases:
    """Test __init__ edge cases."""

    def test_no_schema_collection_not_exists(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=False):
            with pytest.raises(SchemaNotReadyException):
                Collection("nonexistent")
        connections.disconnect("default")

    def test_schema_wrong_type(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=False):
            with pytest.raises(SchemaNotReadyException):
                Collection("test", schema="not_a_schema")
        connections.disconnect("default")

    def test_existing_collection_schema_mismatch(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        server_schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        different_schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=64),
            ]
        )
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=server_schema.to_dict()
        ):
            with pytest.raises(SchemaNotReadyException):
                Collection("test", schema=different_schema)
        connections.disconnect("default")

    def test_existing_collection_schema_wrong_type(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        server_schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=server_schema.to_dict()
        ):
            with pytest.raises(SchemaNotReadyException):
                Collection("test", schema="wrong_type")
        connections.disconnect("default")

    def test_existing_collection_consistency_level_mismatch(
        self, mock_grpc_connect, mock_grpc_close
    ):
        connections.connect(keep_alive=False)
        server_schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        resp = server_schema.to_dict()
        resp["consistency_level"] = "Strong"
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=resp
        ):
            with pytest.raises(SchemaNotReadyException):
                Collection("test", consistency_level="Eventually")
        connections.disconnect("default")

    def test_existing_collection_schema_matches_server(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=128),
            ]
        )
        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=schema.to_dict()
        ):
            coll = Collection("test", schema=schema)
        assert coll.schema == schema
        connections.disconnect("default")


# ---------------------------------------------------------------------------
# Tests for construct_from_dataframe
# ---------------------------------------------------------------------------


class TestConstructFromDataframe:
    def test_non_dataframe_raises(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        with pytest.raises(SchemaNotReadyException):
            Collection.construct_from_dataframe("test", "not_a_dataframe")
        connections.disconnect("default")

    def test_no_primary_field_raises(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        df = pd.DataFrame({"a": [1, 2], "vec": [[1.0] * 4, [2.0] * 4]})
        with pytest.raises(SchemaNotReadyException):
            Collection.construct_from_dataframe("test", df)
        connections.disconnect("default")

    def test_primary_field_not_exist_raises(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        df = pd.DataFrame({"a": [1, 2], "vec": [[1.0] * 4, [2.0] * 4]})
        with pytest.raises(SchemaNotReadyException):
            Collection.construct_from_dataframe("test", df, primary_field="missing")
        connections.disconnect("default")

    def test_auto_id_wrong_type_raises(self, mock_grpc_connect, mock_grpc_close):
        connections.connect(keep_alive=False)
        df = pd.DataFrame({"pk": [1, 2], "vec": [[1.0] * 4, [2.0] * 4]})
        with pytest.raises(AutoIDException):
            Collection.construct_from_dataframe("test", df, primary_field="pk", auto_id="yes")
        connections.disconnect("default")

    def test_construct_from_dataframe_timeout_kwarg(
        self, mock_grpc_connect, mock_grpc_close
    ):
        """construct_from_dataframe(timeout=...) should not raise TypeError for duplicate kwarg."""
        connections.connect(keep_alive=False)
        schema = CollectionSchema(
            [
                FieldSchema("pk", DataType.INT64, is_primary=True),
                FieldSchema("vec", DataType.FLOAT_VECTOR, dim=4),
            ]
        )
        df = pd.DataFrame({"pk": [1, 2], "vec": [[1.0] * 4, [2.0] * 4]})
        fake_res = MagicMock()
        fake_res.insert_count = 2

        with mock.patch(f"{GRPC_PREFIX}.has_collection", return_value=True), mock.patch(
            f"{GRPC_PREFIX}.describe_collection", return_value=schema.to_dict()
        ), mock.patch(f"{GRPC_PREFIX}.batch_insert", return_value=fake_res):
            coll, res = Collection.construct_from_dataframe(
                "ut_cdf_timeout", df, primary_field="pk", timeout=5.0
            )

        assert coll.name == "ut_cdf_timeout"
        connections.disconnect("default")


# ---------------------------------------------------------------------------
# Tests for search_iterator / query_iterator
# ---------------------------------------------------------------------------


class TestSearchIteratorMethod:
    def test_search_iterator_non_str_expr_raises(self, collection):
        with pytest.raises(DataTypeNotMatchException):
            collection.search_iterator(
                data=[[1.0] * 128],
                anns_field="vec",
                param={"metric_type": "L2"},
                batch_size=10,
                expr=123,
            )


class TestQueryIteratorMethod:
    def test_query_iterator_non_str_expr_raises(self, collection):
        with pytest.raises(DataTypeNotMatchException):
            collection.query_iterator(batch_size=10, expr=123)
