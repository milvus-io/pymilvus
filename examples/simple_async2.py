from pymilvus import (
    DataType,
    MilvusClient,
    AsyncMilvusClient,
)
import numpy as np
import asyncio

num_entities, dim = 10000, 128
nq, default_limit = 2, 3
collection_name = "hello_milvus"
partition_name = "p1"
index_field_name = "vector"
rng = np.random.default_rng(seed=19530)
output_fields = ["id"]
uri = "http://localhost:19530"

milvus_client = MilvusClient(uri=uri)
async_milvus_client = AsyncMilvusClient(uri=uri)

loop = asyncio.get_event_loop()

# create collection, partition, index
print("Start dropping all collection")
for c in milvus_client.list_collections():
    loop.run_until_complete(async_milvus_client.drop_collection(c))
print("Dropping collection done")
print("Start creating collection")
schema = async_milvus_client.create_schema(auto_id=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
loop.run_until_complete(async_milvus_client.create_collection(collection_name, schema=schema, consistency_level="Strong"))
print("Creating collection done")
print("Start creating partition")
loop.run_until_complete(async_milvus_client.create_partition(collection_name, partition_name=partition_name))
print("Creating partition done")
print("Start creating index")
index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name=index_field_name, index_type="HNSW", metric_type="COSINE", M=30, efConstruction=200)
loop.run_until_complete(async_milvus_client.create_index(collection_name, index_params))
print("Creating index done")

print(f"all collections: {milvus_client.list_collections()}")
print(f"schema of collection {collection_name}:", milvus_client.describe_collection(collection_name))
print(f"load state of collection {collection_name}:", milvus_client.get_load_state(collection_name, ""))
print(f"has partition p1:", milvus_client.has_partition(collection_name, partition_name))
print(f"load state of partition p1:", milvus_client.get_load_state(collection_name, partition_name))
print(f"describe index {index_field_name}:", milvus_client.describe_index(collection_name, index_field_name))

# load collecton, partition
loop.run_until_complete(async_milvus_client.load_partitions(collection_name, partition_name))
loop.run_until_complete(async_milvus_client.load_collection(collection_name))

print("Loading collecton, partition done")
print(f"load state of collection {collection_name}:", milvus_client.get_load_state(collection_name, ""))
print(f"load state of partition p1:", milvus_client.get_load_state(collection_name, partition_name))

# release collection, partition
loop.run_until_complete(async_milvus_client.release_partitions(collection_name, partition_name))
loop.run_until_complete(async_milvus_client.release_collection(collection_name))

print("Releasing collecton, partition done")
print(f"load state of collection {collection_name}:", milvus_client.get_load_state(collection_name, ""))
print(f"load state of partition p1:", milvus_client.get_load_state(collection_name, partition_name))

# drop collection, partition, index
loop.run_until_complete(async_milvus_client.drop_partition(collection_name, partition_name))
print("Dropping partition done")
print(f"has partition p1:", milvus_client.has_partition(collection_name, partition_name))
loop.run_until_complete(async_milvus_client.drop_index(collection_name, index_field_name))
print("Dropping index done")
print(f"describe index {index_field_name}:", milvus_client.describe_index(collection_name, index_field_name))
loop.run_until_complete(async_milvus_client.drop_collection(collection_name))
print("Dropping collection done")
print(f"all collections: {milvus_client.list_collections()}")
