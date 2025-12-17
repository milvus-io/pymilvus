import time
import numpy as np
from pymilvus import (
    MilvusClient,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)
milvus_client.create_collection(collection_name, dim, consistency_level="Strong", metric_type="L2")

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rng = np.random.default_rng(seed=19530)

milvus_client.create_partition(collection_name, partition_name = "p1")
milvus_client.insert(collection_name, {"id": 1, "vector": rng.random((1, dim))[0], "a": 100}, partition_name = "p1")
milvus_client.insert(collection_name, {"id": 2, "vector": rng.random((1, dim))[0], "b": 200}, partition_name = "p1")
milvus_client.insert(collection_name, {"id": 3, "vector": rng.random((1, dim))[0], "c": 300}, partition_name = "p1")

milvus_client.create_partition(collection_name, partition_name = "p2")
milvus_client.insert(collection_name, {"id": 4, "vector": rng.random((1, dim))[0], "e": 400}, partition_name = "p2")
milvus_client.insert(collection_name, {"id": 5, "vector": rng.random((1, dim))[0], "f": 500}, partition_name = "p2")
milvus_client.insert(collection_name, {"id": 6, "vector": rng.random((1, dim))[0], "g": 600}, partition_name = "p2")

has_p1 = milvus_client.has_partition(collection_name, "p1")
print("has partition p1", has_p1)

has_p3 = milvus_client.has_partition(collection_name, "p3")
print("has partition p3", has_p3)

partitions = milvus_client.list_partitions(collection_name)
print("partitions:", partitions)

milvus_client.release_collection(collection_name)
milvus_client.load_partitions(collection_name, partition_names =["p1", "p2"])

replicas=milvus_client.describe_replica(collection_name)
print("replicas:", replicas)

print(fmt.format("Start search in partiton p1"))
vectors_to_search = rng.random((1, dim))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["pk", "a", "b"], partition_names = ["p1"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

milvus_client.release_partitions(collection_name, partition_names = ["p1"])
milvus_client.drop_partition(collection_name, partition_name = "p1", timeout = 2.0)
print("successfully drop partition p1")

try:
    milvus_client.drop_partition(collection_name, partition_name = "p2", timeout = 2.0)
except Exception as e:
    print(f"cacthed {e}")

has_p1 = milvus_client.has_partition(collection_name, "p1")
print("has partition of p1:", has_p1)

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[2])
assert len(query_results) == 0

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[4])
print(query_results[0])

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter= "f == 500")
for ret in query_results:
    print(ret)

print(fmt.format(f"Start search with retrieve several fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["pk", "a", "b"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

milvus_client.drop_collection(collection_name)
