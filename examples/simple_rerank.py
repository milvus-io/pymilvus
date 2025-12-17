import time
import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema(enable_dynamic_field=False, auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("ts", DataType.INT64)


index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", index_type="FLAT", metric_type="L2")
milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rng = np.random.default_rng(seed=19530)
rows = [
        {"embeddings": rng.random((1, dim))[0], "ts": 100},
        {"embeddings": rng.random((1, dim))[0], "ts": 200},
        {"embeddings": rng.random((1, dim))[0], "ts": 300},
        {"embeddings": rng.random((1, dim))[0], "ts": 400},
        {"embeddings": rng.random((1, dim))[0], "ts": 500},
        {"embeddings": rng.random((1, dim))[0], "ts": 600},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows)
print(fmt.format("Inserting entities done"))
print(insert_result)


print(fmt.format("Start load collection "))
milvus_client.load_collection(collection_name)

rng = np.random.default_rng(seed=19530)
vectors_to_search = rng.random((1, dim))

ranker = Function(
    name="rerank_fn",
    input_field_names=["ts"],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "decay",
        "function": "exp",
        "origin": 0,
        "offset": 200,
        "decay": 0.9,
        "scale": 100
    }
)

print(fmt.format(f"Start search with retrieve several fields."))
result = milvus_client.search(collection_name, vectors_to_search, limit=3, output_fields=["*"], ranker=ranker)
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

vectors_to_search = rng.random((1, dim))
search_param = {
    "data": vectors_to_search,
    "anns_field": "embeddings",
    "param": {"metric_type": "L2"},
    "limit": 3,
}
req = AnnSearchRequest(**search_param)

hybrid_res = milvus_client.hybrid_search(collection_name, [req, req], ranker=ranker, limit=3, output_fields=["ts"])
for hits in hybrid_res:
    for hit in hits:
        print(f" hybrid search hit: {hit}")

milvus_client.drop_collection(collection_name)
