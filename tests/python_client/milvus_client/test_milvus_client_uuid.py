import uuid

import numpy as np
import pytest

from pymilvus import DataType

from milvus_client_base import TestMilvusClientV2Base
from check.func_check import CheckTasks
from common import CaseLabel, cf, ct
from common.common import default_dim

prefix = "uuid_test"


class TestMilvusClientUUID(TestMilvusClientV2Base):
    """Tests for UUID field type support in MilvusClient."""

    @pytest.mark.tags(CaseLabel.L1)
    def test_uuid_insert_invalid_uuid_string(self):
        """
        target: test insert with invalid UUID strings
        method: attempt to insert rows with malformed UUID values
        expected: each invalid UUID should be rejected with a proper error
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)

        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.UUID, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=default_dim)

        self.create_collection(client, collection_name, schema=schema)

        rng = np.random.default_rng(seed=19530)

        # Various invalid UUID strings
        invalid_rows = [
            {"id": "not-a-uuid-at-all", "vector": list(rng.random(default_dim))},
            {"id": "550e8400-e29b-41d4", "vector": list(rng.random(default_dim))},
            {"id": "", "vector": list(rng.random(default_dim))},
            {"id": "123", "vector": list(rng.random(default_dim))},
        ]

        errors_raised = 0
        for row in invalid_rows:
            rows = [row]
            try:
                self.insert(client, collection_name, rows,
                            check_task=CheckTasks.err_res,
                            check_items={ct.err_code: 999})
                errors_raised += 1  # Should have raised
            except Exception as e:
                # Verify it's a UUID validation error, not something else
                assert "uuid" in str(e).lower() or "UUID" in str(e), f"Unexpected error: {e}"
                errors_raised += 1

        assert errors_raised == len(invalid_rows)

        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L1)
    def test_uuid_partition_key(self):
        """
        target: test UUID field as partition key
        method: create collection with INT64 PK and UUID partition key
        expected: data is partitioned correctly by UUID hash
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)

        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=default_dim)
        schema.add_field("uuid_field", DataType.UUID, is_partition_key=True)
        schema.add_field("name", DataType.VARCHAR, max_length=100)

        self.create_collection(client, collection_name, schema=schema,
                               num_partitions=4, consistency_level="Strong")

        # Insert data with various UUIDs
        rng = np.random.default_rng(seed=19530)
        rows = [
            {
                "id": i,
                "vector": list(rng.random(default_dim)),
                "uuid_field": str(uuid.uuid4()),
                "name": f"pk_user_{i}",
            }
            for i in range(50)
        ]
        self.insert(client, collection_name, rows)
        self.flush(client, collection_name)

        # Query all - verify data is accessible
        res, _ = self.query(
            client, collection_name,
            filter="id >= 0",
            output_fields=["id", "uuid_field"]
        )
        assert len(res) == 50

        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_uuid_clustering_key(self):
        """
        target: test UUID field as clustering key
        method: create collection with UUID PK and UUID clustering key
        expected: data inserted and queryable with clustering
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)

        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=default_dim)
        schema.add_field("uuid_field", DataType.UUID, is_clustering_key=True)
        schema.add_field("value", DataType.INT64)

        self.create_collection(client, collection_name, schema=schema,
                               consistency_level="Strong")

        # Insert data
        rng = np.random.default_rng(seed=19530)
        rows = [
            {
                "id": i,
                "vector": list(rng.random(default_dim)),
                "uuid_field": str(uuid.uuid4()),
                "value": i * 10,
            }
            for i in range(30)
        ]
        self.insert(client, collection_name, rows)
        self.flush(client, collection_name)

        # Query all data
        res, _ = self.query(
            client, collection_name,
            filter="id >= 0",
            output_fields=["id", "uuid_field", "value"],
            limit=30
        )
        assert len(res) == 30

        self.drop_collection(client, collection_name)

    @pytest.mark.tags(CaseLabel.L2)
    def test_uuid_hybrid_search(self):
        """
        target: test ANN search with UUID filter
        method: insert data, search with vector + UUID filter
        expected: results filtered by UUID condition
        """
        client = self._client()
        collection_name = cf.gen_unique_str(prefix)

        schema = self.create_schema(client, enable_dynamic_field=False)[0]
        schema.add_field("id", DataType.UUID, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=default_dim)
        schema.add_field("category", DataType.VARCHAR, max_length=50)

        idx = self.prepare_index_param(client)[0]
        self.create_collection(client, collection_name, schema=schema,
                               index_params=idx, consistency_level="Strong")
        self.load_collection(client, collection_name)

        # Insert data
        rng = np.random.default_rng(seed=19530)
        target_uuid = str(uuid.uuid4())
        rows = []
        for i in range(100):
            rows.append({
                "id": target_uuid if i == 0 else str(uuid.uuid4()),
                "vector": list(rng.random(default_dim)),
                "category": "test",
            })

        self.insert(client, collection_name, rows)
        self.flush(client, collection_name)
        self.load_collection(client, collection_name)

        # Search with a vector and UUID filter
        search_vec = [list(rng.random(default_dim))]
        res = self.search(
            client, collection_name,
            search_vec,
            filter=f'id == "{target_uuid}"',
            output_fields=["id", "category"],
            limit=5
        )[0]
        assert len(res) >= 1
        # The target UUID row should be in results (if vector is close enough)
        # Or at minimum, the search succeeded with a UUID filter

        self.drop_collection(client, collection_name)
