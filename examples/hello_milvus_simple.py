import time
import numpy as np
from pymilvus import (
    MilvusClient,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")
milvus_client.drop_collection(collection_name)
milvus_client.create_collection(collection_name, dim, consistency_level="Bounded", metric_type="L2", auto_id=True)

print("collections:", milvus_client.list_collections())
print(f"{collection_name} :", milvus_client.describe_collection(collection_name))
rng = np.random.default_rng(seed=19530)

rows = [
    {"vector": rng.random((1, dim))[0], "a": 1},
    {"vector": rng.random((1, dim))[0], "b": 1},
    {"vector": rng.random((1, dim))[0], "c": 1},
    {"vector": rng.random((1, dim))[0], "d": 1},
    {"vector": rng.random((1, dim))[0], "e": 1},
    {"vector": rng.random((1, dim))[0], "f": 1},
]
print(fmt.format("Start inserting entities"))
pks = milvus_client.insert(collection_name, rows, progress_bar=True)
pks2 = milvus_client.insert(collection_name, {"vector": rng.random((1, dim))[0], "g": 1})
pks.extend(pks2)
print(fmt.format("Start searching based on vector similarity"))

print("len of pks:", len(pks), "first pk is :", pks[0])

print(f"get first primary key {pks[0]} from {collection_name}")
first_pk_data = milvus_client.get(collection_name, pks[0:1])
print(f"data of primary key {pks[0]} is", first_pk_data)

print(f"start to delete first 2 of primary keys in collection {collection_name}")
delete_pks = milvus_client.delete(collection_name, pks[0:2])
print("deleted:", delete_pks)
rng = np.random.default_rng(seed=19530)
vectors_to_search = rng.random((1, dim))

start_time = time.time()

print(fmt.format(f"Start search with retrieve all fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["pk", "a", "b"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

query_pks = [str(entry) for entry in pks[2:4]]
filter = f"id in [{','.join(query_pks)}]"

# filter = f" id in [{','.join(pks[2:4])}]"
print(fmt.format(f"Start query({filter}) with retrieve all fields."))

filter_data = milvus_client.query(collection_name, filter)
print("filter_data:", filter_data)

milvus_client.drop_collection(collection_name)
