import time
import numpy as np
from pymilvus import (
    MilvusClient,
    DataType
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema(enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("title", DataType.VARCHAR, max_length=64)


index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", metric_type="L2")
milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rng = np.random.default_rng(seed=19530)
rows = [
        {"id": 1, "embeddings": rng.random((1, dim))[0], "a": 100, "title": "t1"},
        {"id": 2, "embeddings": rng.random((1, dim))[0], "b": 200, "title": "t2"},
        {"id": 3, "embeddings": rng.random((1, dim))[0], "c": 300, "title": "t3"},
        {"id": 4, "embeddings": rng.random((1, dim))[0], "d": 400, "title": "t4"},
        {"id": 5, "embeddings": rng.random((1, dim))[0], "e": 500, "title": "t5"},
        {"id": 6, "embeddings": rng.random((1, dim))[0], "f": 600, "title": "t6"},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows)
print(fmt.format("Inserting entities done"))
print(insert_result)


print(fmt.format("Start load collection "))
milvus_client.load_collection(collection_name)

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[2])
print(query_results[0])

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter= "f == 600 or title == 't2'")
for ret in query_results:
    print(ret)

rng = np.random.default_rng(seed=19530)
vectors_to_search = rng.random((1, dim))

print(fmt.format(f"Start search with retrieve several fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["pk", "a", "b"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

milvus_client.drop_collection(collection_name)
