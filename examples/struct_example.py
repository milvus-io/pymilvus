import numpy as np
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

milvus_client = MilvusClient("http://localhost:19530")
has_collection = milvus_client.has_collection(COLLECTION_NAME, timeout=5)
if has_collection:
    milvus_client.drop_collection(COLLECTION_NAME)

# ---- Schema ----
schema = milvus_client.create_schema(enable_dynamic_field=True, auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True, max_length=100)
schema.add_field("content", DataType.VARCHAR, max_length=10000, enable_analyzer=True,
                 analyzer_params=analyzer_params, enable_match=True)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=16)

# struct_field: VARCHAR + INT32 + FLOAT_VECTOR
struct_schema = milvus_client.create_struct_field_schema()
struct_schema.add_field("struct_str", DataType.VARCHAR, max_length=65535)
struct_schema.add_field("struct_int", DataType.INT32)
struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
schema.add_field("struct_field", datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema, max_capacity=1000)

# struct_field2: VARCHAR + FLOAT_VECTOR
struct_schema = milvus_client.create_struct_field_schema()
struct_schema.add_field("struct_str", DataType.VARCHAR, max_length=65535)
struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
schema.add_field("struct_field2", datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema, max_capacity=1000)

milvus_client.create_collection(COLLECTION_NAME, schema=schema)

# ---- Index ----
index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", index_params={"nlist": 128})
# struct_field vector uses COSINE (for element-level search)
index_params.add_index(field_name="struct_field[struct_float_vec]", index_type="HNSW", index_name="struct_float_vec_index1", metric_type="COSINE", index_params={"M": 16, "efConstruction": 200})
# struct_field2 vector uses MAX_SIM_IP (for EmbeddingList search)
index_params.add_index(field_name="struct_field2[struct_float_vec]", index_type="HNSW", index_name="struct_float_vec_index3", metric_type="MAX_SIM_IP", index_params={"M": 16, "efConstruction": 200})
milvus_client.create_index(COLLECTION_NAME, schema=schema, index_params=index_params)

coll = milvus_client.describe_collection(COLLECTION_NAME)
print("Describe collection", coll)

# ---- Insert ----
rng = np.random.default_rng(seed=19530)
N = 1000
content = ["aaa", "jjj", "sss", "ddd", "fff", "ggg", "hhh", "iii", "czx", "www", "qqq", "eee"]

for _ in range(10):
    arr_len = random.randint(2, 10)
    rows = [{"embedding": rng.random((1, 16))[0],
             "content": content[random.randint(0, len(content) - 1)],
             "struct_field": [
                 {
                  "struct_str": content[random.randint(0, len(content) - 1)],
                  "struct_int": random.randint(0, 100),
                  "struct_float_vec": rng.random((1, EMBEDDING_DIM))[0],
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
    milvus_client.insert(COLLECTION_NAME, rows)

milvus_client.flush(COLLECTION_NAME)
milvus_client.load_collection(COLLECTION_NAME)

# ---- Query ----
result = milvus_client.query(
    collection_name=COLLECTION_NAME,
    filter="",
    output_fields=["struct_field"],
    limit=2,
)
print("=== Query ===")
print(result)

# ---- EmbeddingList Search ----
# EmbeddingList search on struct_field2 vector (multi-vector batch query)
rng = np.random.default_rng(seed=19530)
queries = [
    EmbeddingList._from_random_test(7, EMBEDDING_DIM, seed=19530),
    EmbeddingList._from_random_test(4, EMBEDDING_DIM, seed=19531),
]
embeddingList = EmbeddingList()
embeddingList.add(np.random.randn(EMBEDDING_DIM))
embeddingList.add(np.random.randn(EMBEDDING_DIM))
queries.append(embeddingList)

print("\n=== EmbeddingList Search ===")
res = milvus_client.search(COLLECTION_NAME, data=queries, limit=10,
                           anns_field="struct_field2[struct_float_vec]",
                           output_fields=["struct_field[struct_str]"])
for i, (hits, query) in enumerate(zip(res, queries)):
    print(f"--- Query {i+1} ({query}) - Total hits: {len(hits)} ---")
    for hit in hits:
        print(f"    id: {hit.id}, distance: {hit.distance}, entity: {hit}")

# ---- Element-Level Search (No Filter) ----
# Vector search on struct_field at element level, returning per-element results with offset
print("\n=== Element-Level Search (No Filter) ===")
query_vec = [np.random.randn(EMBEDDING_DIM)]
res = milvus_client.search(COLLECTION_NAME, data=query_vec, limit=10,
                           anns_field="struct_field[struct_float_vec]",
                           output_fields=["struct_field[struct_int]", "struct_field[struct_str]"])
for i, (hits, query) in enumerate(zip(res, query_vec)):
    print(f"--- Total hits: {len(hits)} ---")
    for hit in hits:
        print(f"    {hit}")

# ---- Element-Level Search (With Filter) ----
# Vector search on struct_field with element_filter to filter by struct_int values
print("\n=== Element-Level Search (With Filter) ===")
query_vec = [np.random.randn(EMBEDDING_DIM)]
filter_expr = "content == 'aaa' && element_filter(struct_field, $[struct_int] == 3 || $[struct_int] == 5 || $[struct_int] == 7)"
res = milvus_client.search(COLLECTION_NAME, data=query_vec, limit=10,
                           anns_field="struct_field[struct_float_vec]",
                           filter=filter_expr,
                           search_params={"hints": "iterative_filter"},
                           output_fields=["struct_field[struct_int]", "struct_field[struct_str]"])
for i, (hits, query) in enumerate(zip(res, query_vec)):
    print(f"--- Total hits: {len(hits)} ---")
    for hit in hits:
        print(f"    {hit}")

# ---- Element-Level Search (With Filter + Group By PK) ----
# Group by primary key to deduplicate, returning only the best matching element per entity
print("\n=== Element-Level Search (With Filter + Group By PK) ===")
query_vec = [np.random.randn(EMBEDDING_DIM)]
res = milvus_client.search(COLLECTION_NAME, data=query_vec, limit=10,
                           anns_field="struct_field[struct_float_vec]",
                           filter='element_filter(struct_field, $[struct_int] > 50)',
                           group_by_field="id",
                           output_fields=["struct_field[struct_int]", "struct_field[struct_str]"])
for i, (hits, query) in enumerate(zip(res, query_vec)):
    print(f"--- Total hits: {len(hits)} ---")
    for hit in hits:
        print(f"    {hit}")

# ---- Element-Level Query ----
# Query with element_filter, returning matched elements with offset
print("\n=== Element-Level Query ===")
result = milvus_client.query(
    collection_name=COLLECTION_NAME,
    filter='element_filter(struct_field, $[struct_int] > 50)',
    output_fields=["struct_field[struct_int]", "struct_field[struct_str]"],
    limit=5,
)
for r in result:
    print(f"    {r}")