import numpy as np
import time
import random
from pymilvus import (
    MilvusClient,
    DataType
)
from pymilvus.client.embedding_list import EmbeddingList

COLLECTION_NAME = "Documents"
EMBEDDING_DIM = 32
analyzer_params = {
    "type": "standard",
    "filter": ["lowercase"],
}
fmt = "\n=== {:30} ===\n"

milvus_client = MilvusClient("http://localhost:19530")
has_collection = milvus_client.has_collection(COLLECTION_NAME, timeout=5)
if has_collection:
    milvus_client.drop_collection(COLLECTION_NAME)

schema = milvus_client.create_schema(enable_dynamic_field=True, auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True, max_length=100)
schema.add_field("content", DataType.VARCHAR, max_length=10000, enable_analyzer=True,
                 analyzer_params=analyzer_params, enable_match=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=16)


struct_schema = milvus_client.create_struct_field_schema()
struct_schema.add_field("struct_str", DataType.VARCHAR, max_length=65535)
struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
# we can have multiple vector field in a struct
# struct_schema.add_field("struct_float16_vec", DataType.FLOAT16_VECTOR, dim=EMBEDDING_DIM)
schema.add_field("struct_field",datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema, max_capacity=1000)

struct_schema = milvus_client.create_struct_field_schema()
struct_schema.add_field("struct_str", DataType.VARCHAR, max_length=65535)
struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
schema.add_field("struct_field2",datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema, max_capacity=1000)


milvus_client.create_collection(COLLECTION_NAME, schema=schema)

index_params = milvus_client.prepare_index_params()
dense_index_params = {
    "index_type": "IVF_FLAT",  # Choose an appropriate index type (e.g., IVF_FLAT)
    "metric_type": "COSINE",   # Metric type (e.g., COSINE for cosine similarity)
    "params": {"nlist": 128}   # Index parameters
}

index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", index_params={"nlist": 128})
index_params.add_index(field_name="struct_field[struct_float_vec]", index_type="HNSW", index_name="struct_float_vec_index1", metric_type="MAX_SIM_COSINE", index_params={"M": 16, "efConstruction": 200})
# index_params.add_index(field_name="struct_field[struct_float16_vec]", index_type="HNSW", index_name="struct_float_vec_index2",metric_type="MAX_SIM_COSINE", index_params={"M": 16, "efConstruction": 200})
index_params.add_index(field_name="struct_field2[struct_float_vec]", index_type="HNSW", index_name="struct_float_vec_index3", metric_type="MAX_SIM_IP", index_params={"M": 16, "efConstruction": 200})
milvus_client.create_index(COLLECTION_NAME, schema=schema, index_params=index_params)

coll = milvus_client.describe_collection(COLLECTION_NAME)
print("Describe collection", coll)

rng = np.random.default_rng(seed=19530)

N = 100
content = ["aaa",
           "jjj",
           "sss",
           "asdasd",
           "dvzicxv",
           "dcmxvxv",
           "dvxcnvmlzc",
           "aczxcc",
           "qiqixcc"]
iterations = 10
for _ in range(iterations):
    arr_len = random.randint(2, 5)
    rows = [{"embedding": rng.random((1, 16))[0],
             "content": content[random.randint(0, len(content) - 1)],
             "struct_field": [
                 {
                  "struct_str": content[random.randint(0, len(content) - 1)],
                  "struct_float_vec": rng.random((1, EMBEDDING_DIM))[0],
                  # "struct_float16_vec": rng.random((1, EMBEDDING_DIM))[0].astype(np.float16),
                  }
                  for _ in range(arr_len)],
             "struct_field2": [
                 {
                     "struct_str": content[random.randint(0, len(content) - 1)],
                     "struct_float_vec": rng.random((1, EMBEDDING_DIM))[0],
                 }
                 for _ in range(arr_len)],
             }
            for _ in range(N)]
    result = milvus_client.insert(COLLECTION_NAME, rows)

milvus_client.flush(COLLECTION_NAME)

milvus_client.load_collection(COLLECTION_NAME)

result = milvus_client.query(
    collection_name=COLLECTION_NAME,
    filter="",
    output_fields=["struct_field"],
    limit=2,
)

print("Query", result)

rng = np.random.default_rng(seed=19530)



# Create search queries using EmbeddingList
# For testing purposes, using random test data
queries = [
    EmbeddingList._from_random_test(7, EMBEDDING_DIM, seed=19530),  # Query with 7 vectors
    EmbeddingList._from_random_test(4, EMBEDDING_DIM, seed=19531),  # Query with 4 vectors
]

embeddingList = EmbeddingList()
# Query with 2 vectors
embeddingList.add(np.random.randn(EMBEDDING_DIM))
embeddingList.add(np.random.randn(EMBEDDING_DIM))

queries.append(embeddingList)

field = "struct_field[struct_float_vec]"
res = milvus_client.search(COLLECTION_NAME, data=queries, limit=10, anns_field=field,
                     output_fields=["struct_field[struct_str]"])

for i, (hits, query) in enumerate(zip(res, queries)):
    print(f"============== Query {i+1} ({query}) - Total hits: {len(hits)}")
    for hit in hits:
        print(f"    Hit id: {hit.id}, distance: {hit.distance}")
        print(f"        entity: {hit}")