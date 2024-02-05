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
milvus_client.create_collection(collection_name, dim, consistency_level="Strong", metric_type="L2", auto_id=True)

collection_name2 = "hello_milvus2"
milvus_client.drop_collection(collection_name2)
milvus_client.create_collection(collection_name2, dim, consistency_level="Strong", metric_type="L2", auto_id=True)


print("collections:", milvus_client.list_collections())

desc_c1 = milvus_client.describe_collection(collection_name)
print(f"{collection_name} :", desc_c1)

rng = np.random.default_rng(seed=19530)

rows = [
    {"vector": rng.random((1, dim))[0], "a": 100},
    {"vector": rng.random((1, dim))[0], "b": 200},
    {"vector": rng.random((1, dim))[0], "c": 300},
]

print(fmt.format(f"Start inserting entities to {collection_name}"))
insert_result = milvus_client.insert(collection_name, rows)
print(insert_result)

rows = [
    {"vector": rng.random((1, dim))[0], "d": 400},
    {"vector": rng.random((1, dim))[0], "e": 500},
    {"vector": rng.random((1, dim))[0], "f": 600},
]

print(fmt.format(f"Start inserting entities to {collection_name2}"))
insert_result2 = milvus_client.insert(collection_name2, rows)
print(insert_result2)



alias = "alias_hello_milvus"
milvus_client.drop_alias(alias)
milvus_client.create_alias(collection_name, alias)

aliases =  milvus_client.list_aliases(collection_name)
print(f"aliases of {collection_name} is:", aliases)


alias_info =  milvus_client.describe_alias(alias)
print(f"info of {alias} is:", alias_info)

assert milvus_client.describe_collection(alias) == milvus_client.describe_collection(collection_name)

milvus_client.alter_alias(collection_name2, alias)
assert milvus_client.describe_collection(alias) == milvus_client.describe_collection(collection_name2)

query_results = milvus_client.query(alias, filter= "f == 600")
print("results of query 'f == 600' is ")
for ret in query_results: 
    print(ret)


milvus_client.drop_alias(alias)
has_collection = milvus_client.has_collection(alias)
assert not has_collection
has_collection = milvus_client.has_collection(collection_name2)
assert has_collection

milvus_client.drop_collection(collection_name)
milvus_client.drop_collection(collection_name2)
