from pymilvus import (
    DataType,
    MilvusClient,
    AsyncMilvusClient,
    AnnSearchRequest,
    RRFRanker,
)
import numpy as np
import asyncio
import time
import random

fmt = "\n=== {:30} ===\n"
num_entities, dim = 100, 8
default_limit = 3
collection_name = "hello_milvus"
rng = np.random.default_rng(seed=19530)

milvus_client = MilvusClient("example.db")
async_milvus_client = AsyncMilvusClient("example.db")

loop = asyncio.get_event_loop()

schema = milvus_client.create_schema(auto_id=False, description="hello_milvus is the simplest demo to introduce the APIs")
schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=100)
schema.add_field("random", DataType.DOUBLE)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("embeddings2", DataType.FLOAT_VECTOR, dim=dim)

index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", index_type = "HNSW", metric_type="L2", nlist=128)
index_params.add_index(field_name = "embeddings2",index_type = "HNSW", metric_type="L2", nlist=128)

# Always use `await` when you want to guarantee the execution order of tasks.
async def recreate_collection():
    print(fmt.format("Start dropping collection"))
    await async_milvus_client.drop_collection(collection_name)
    print(fmt.format("Dropping collection done"))
    print(fmt.format("Start creating collection"))
    await async_milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")
    print(fmt.format("Creating collection done"))

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    loop.run_until_complete(recreate_collection())
else:
    print(fmt.format("Start creating collection"))
    loop.run_until_complete(async_milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong"))
    print(fmt.format("Creating collection done"))

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

async def async_insert(collection_name):
    entities = [
        # provide the pk field because `auto_id` is set to False
        [str(i) for i in range(num_entities)],
        rng.random(num_entities).tolist(),  # field random, only supports list
        rng.random((num_entities, dim)),  # field embeddings, supports numpy.ndarray and list
        rng.random((num_entities, dim)),  # field embeddings2, supports numpy.ndarray and list
    ]
    rows = [ {"pk": entities[0][i], "random": entities[1][i], "embeddings": entities[2][i], "embeddings2": entities[3][i]} for i in range (num_entities)]
    print(fmt.format("Start async inserting entities"))

    start_time = time.time()
    tasks = []
    for row in rows:
        task = async_milvus_client.insert(collection_name, [row])
        tasks.append(task)
    await asyncio.gather(*tasks)
    end_time = time.time()
    print(fmt.format("Total time: {:.2f} seconds".format(end_time - start_time)))
    print(fmt.format("Async inserting entities done"))

loop.run_until_complete(async_insert(collection_name))

async def other_async_task(collection_name):
    tasks = []
    # search
    random_vector = rng.random((1, dim))
    random_vector2 = rng.random((1, dim))
    task = async_milvus_client.search(collection_name, random_vector, limit=default_limit, output_fields=["pk"], anns_field="embeddings")
    tasks.append(task)
    # hybrid search
    search_param = {
        "data": random_vector,
        "anns_field": "embeddings",
        "param": {"metric_type": "L2"},
        "limit": default_limit,
        "expr": "random > 0.5"}
    req = AnnSearchRequest(**search_param)
    task = async_milvus_client.hybrid_search(collection_name, [req], RRFRanker(), default_limit, output_fields=["pk"])
    tasks.append(task)
    # get
    random_pk = random.randint(0, num_entities - 1)
    task = async_milvus_client.get(collection_name=collection_name, ids=[random_pk])
    tasks.append(task)
    # query
    task = async_milvus_client.query(collection_name=collection_name, filter="", limit=default_limit)
    tasks.append(task)
    # delete
    task = async_milvus_client.delete(collection_name=collection_name, ids=[random_pk])
    tasks.append(task)
    # insert
    task = async_milvus_client.insert(
        collection_name=collection_name,
        data=[{"pk": str(random_pk), "random": random_vector[0][0], "embeddings": random_vector[0], "embeddings2": random_vector[0]}],
    )
    tasks.append(task)
    # upsert
    task = async_milvus_client.upsert(
        collection_name=collection_name,
        data=[{"pk": str(random_pk), "random": random_vector2[0][0], "embeddings": random_vector2[0], "embeddings2": random_vector2[0]}],
    )
    tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results

results = loop.run_until_complete(other_async_task(collection_name))
for r in results:
    print(r)
