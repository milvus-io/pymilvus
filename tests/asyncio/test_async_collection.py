import random
import unittest

from pymilvus import FieldSchema, CollectionSchema, DataType
from pymilvus.asyncio import connections, Collection


# this test case requires a running milvus instance. E.g.:
#     export IMAGE_REPO=milvusdb
#     export IMAGE_TAG=2.1.0-latest
#     docker-compose --file ci/docker/milvus/docker-compose.yml up
class TestAsyncCollections(unittest.IsolatedAsyncioTestCase):
    async def test_collection_search(self):
        await connections.connect()
        schema = CollectionSchema([
            FieldSchema("film_id", DataType.INT64, is_primary=True),
            FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
        ])
        collection = Collection("test_collection_search", schema)
        # insert
        data = [
            list(range(10)),
            [[random.random() for _ in range(2)] for _ in range(10)],
        ]
        await collection.insert(data)
        await collection.create_index("films", {"index_type": "FLAT", "metric_type": "L2", "params": {}})
        await collection.load()
        # search
        search_param = {
            "data": [[1.0, 1.0]],
            "anns_field": "films",
            "param": {"metric_type": "L2", "offset": 1},
            "limit": 2,
            "expr": "film_id > 0",
        }
        res = await collection.search(**search_param)
        assert len(res) == 1
        hits = res[0]
        assert len(hits) == 2
        print(f"- Total hits: {len(hits)}, hits ids: {hits.ids} ")
        # - Total hits: 2, hits ids: [8, 5]
        print(f"- Top1 hit id: {hits[0].id}, distance: {hits[0].distance}, score: {hits[0].score} ")
        # - Top1 hit id: 8, distance: 0.10143111646175385, score: 0.10143111646175385
