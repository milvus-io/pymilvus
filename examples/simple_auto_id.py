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
milvus_client.create_collection(collection_name, dim, consistency_level="Strong", metric_type="L2", auto_id=True)

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rng = np.random.default_rng(seed=19530)
rows = [
        {"vector": rng.random((1, dim))[0], "a": 100},
        {"vector": rng.random((1, dim))[0], "b": 200},
        {"vector": rng.random((1, dim))[0], "c": 300},
        {"vector": rng.random((1, dim))[0], "d": 400},
        {"vector": rng.random((1, dim))[0], "e": 500},
        {"vector": rng.random((1, dim))[0], "f": 600},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
print("insert done:", insert_result)

print(fmt.format("Start query by specifying filter"))
query_results = milvus_client.query(collection_name, filter= "f == 600")
for ret in query_results:
    print(ret)


print(f"start to delete by specifying filter in collection {collection_name}")
delete_result = milvus_client.delete(collection_name, filter = "f == 600")
print(delete_result)

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter= "f == 600")
assert len(query_results) == 0

rng = np.random.default_rng(seed=19530)
vectors_to_search = rng.random((1, dim))

print(fmt.format(f"Start search with retrieve several fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["pk", "a", "b"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

milvus_client.drop_collection(collection_name)
