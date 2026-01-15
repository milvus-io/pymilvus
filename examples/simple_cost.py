import time
import numpy as np
from pymilvus import (
    MilvusClient,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_client_cost"
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
rows = [
        {"id": 1, "vector": rng.random((1, dim))[0], "a": 100},
        {"id": 2, "vector": rng.random((1, dim))[0], "b": 200},
        {"id": 3, "vector": rng.random((1, dim))[0], "c": 300},
        {"id": 4, "vector": rng.random((1, dim))[0], "d": 400},
        {"id": 5, "vector": rng.random((1, dim))[0], "e": 500},
        {"id": 6, "vector": rng.random((1, dim))[0], "f": 600},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
print(fmt.format("Inserting entities done"))
# OUTPUT:
# insert result: {'insert_count': 6, 'ids': [1, 2, 3, 4, 5, 6], 'cost': '1'};
# insert cost: 1
print(f"insert result: {insert_result};\ninsert cost: {insert_result['cost']}")

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[2])
# OUTPUT:
# query result: data: ["{'id': 2, 'vector': [0.9007387, 0.44944635, 0.18477614, 0.42930314, 0.40345728, 0.3957196, 0.6963897, 0.24356908], 'b': 200}"], extra_info: {'cost': '21'}
# query cost: 21
print(f"query result: {query_results}\nquery cost: {query_results.extra['cost']}")

upsert_ret = milvus_client.upsert(collection_name, {"id": 2 , "vector": rng.random((1, dim))[0], "g": 100})
# OUTPUT:
# upsert result: {'upsert_count': 1, 'cost': '2'}
# upsert cost: 2
print(f"upsert result: {upsert_ret}\nupsert cost: {upsert_ret['cost']}")

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[2])
print(f"query result: {query_results}\nquery cost: {query_results.extra['cost']}")

print(f"start to delete by specifying filter in collection {collection_name}")
delete_result = milvus_client.delete(collection_name, ids=[6])
# OUTPUT:
# delete result: {'delete_count': 1, 'cost': '1'}
# delete cost: 1
print(f"delete result: {delete_result}\ndelete cost: {delete_result['cost']}")

rng = np.random.default_rng(seed=19530)
vectors_to_search = rng.random((1, dim))

print(fmt.format(f"Start search with retrieve several fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["pk", "a", "b"])
print(f"search result: {result}\nsearch cost: {result.extra['cost']}")

milvus_client.drop_collection(collection_name)
