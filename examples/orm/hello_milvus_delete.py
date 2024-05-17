import time
import numpy as np
from pymilvus import (
    MilvusClient,
    exceptions
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")
milvus_client.drop_collection(collection_name)
milvus_client.create_collection(collection_name, dim, consistency_level="Strong", metric_type="L2")

print("collections:", milvus_client.list_collections())
print(f"{collection_name} :", milvus_client.describe_collection(collection_name))
rng = np.random.default_rng(seed=19530)

rows = [
        {"id": 1, "vector": rng.random((1, dim))[0], "a": 1},
        {"id": 2, "vector": rng.random((1, dim))[0], "b": 2},
        {"id": 3, "vector": rng.random((1, dim))[0], "c": 3},
        {"id": 4, "vector": rng.random((1, dim))[0], "d": 4},
        {"id": 5, "vector": rng.random((1, dim))[0], "e": 5},
        {"id": 6, "vector": rng.random((1, dim))[0], "f": 6},
]

print(fmt.format("Start inserting entities"))
pks = milvus_client.insert(collection_name, rows, progress_bar=True)
pks2 = milvus_client.insert(collection_name, {"id": 7, "vector": rng.random((1, dim))[0], "g": 1})
pks.extend(pks2)


def fetch_data_by_pk(pk):
    print(f"get primary key {pk} from {collection_name}")
    pk_data = milvus_client.get(collection_name, pk)

    if pk_data:
        print(f"data of primary key {pk} is", pk_data[0])
    else:
        print(f"data of primary key {pk} is empty")

fetch_data_by_pk(pks[2])

print(f"start to delete primary key {pks[2]} in collection {collection_name}")
milvus_client.delete(collection_name, pks = pks[2])

fetch_data_by_pk(pks[2])


fetch_data_by_pk(pks[4])
filter = "e == 5 or f == 6"
print(f"start to delete by expr {filter} in collection {collection_name}")
milvus_client.delete(collection_name, filter=filter)

fetch_data_by_pk(pks[4])

print(f"start to delete by expr '{filter}' or by primary 4 in collection {collection_name}, expect get exception")
try:
    milvus_client.delete(collection_name, pks = 4, filter=filter)
except Exception as e:
    assert isinstance(e, exceptions.ParamError)
    print("catch exception", e)

print(f"start to delete without specify any expr '{filter}' or any primary key in collection {collection_name}, expect get exception")
try:
    milvus_client.delete(collection_name)
except Exception as e:
    print("catch exception", e)

result = milvus_client.query(collection_name, "", output_fields = ["count(*)"])
print(f"final entities in {collection_name} is {result[0]['count(*)']}")

milvus_client.drop_collection(collection_name)
