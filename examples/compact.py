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

rng = np.random.default_rng(seed=19530)
rows = [
        {"id": 1, "vector": rng.random((1, dim))[0], "a": 100},
        {"id": 2, "vector": rng.random((1, dim))[0], "b": 200},
        {"id": 3, "vector": rng.random((1, dim))[0], "c": 300},
        {"id": 4, "vector": rng.random((1, dim))[0], "d": 400},
        {"id": 5, "vector": rng.random((1, dim))[0], "e": 500},
        {"id": 6, "vector": rng.random((1, dim))[0], "f": 600},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows)
print(fmt.format("Inserting entities done"))
print(insert_result)

upsert_ret = milvus_client.upsert(collection_name, {"id": 2 , "vector": rng.random((1, dim))[0], "g": 100})
print(upsert_ret)

print(fmt.format("Start flush"))
milvus_client.flush(collection_name)
print(fmt.format("flush done"))

result = milvus_client.query(collection_name, "", output_fields = ["count(*)"])
print(f"final entities in {collection_name} is {result[0]['count(*)']}")

rows = [
        {"id": 7, "vector": rng.random((1, dim))[0], "g": 700},
        {"id": 8, "vector": rng.random((1, dim))[0], "h": 800},
        {"id": 9, "vector": rng.random((1, dim))[0], "i": 900},
        {"id": 10, "vector": rng.random((1, dim))[0], "j": 1000},
        {"id": 11, "vector": rng.random((1, dim))[0], "k": 1100},
        {"id": 12, "vector": rng.random((1, dim))[0], "l": 1200},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows)
print(fmt.format("Inserting entities done"))
print(insert_result)

print(fmt.format("Start flush"))
milvus_client.flush(collection_name)
print(fmt.format("flush done"))

result = milvus_client.query(collection_name, "", output_fields = ["count(*)"])
print(f"final entities in {collection_name} is {result[0]['count(*)']}")

print(fmt.format("Start compact"))
job_id = milvus_client.compact(collection_name)
print(f"job_id:{job_id}")

cnt = 0
state = milvus_client.get_compaction_state(job_id)
while (state != "Completed" and cnt < 10):
    time.sleep(1.0)
    state = milvus_client.get_compaction_state(job_id) 
    print(f"compaction state: {state}")
    cnt += 1

if state == "Completed":
    print(fmt.format("compact done"))
else:
    print(fmt.format("compact timeout"))

result = milvus_client.query(collection_name, "", output_fields = ["count(*)"])
print(f"final entities in {collection_name} is {result[0]['count(*)']}")

milvus_client.drop_collection(collection_name)
