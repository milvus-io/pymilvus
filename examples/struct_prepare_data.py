from gc import enable
from time import sleep

import numpy as np
import time
import random
from pymilvus import (
    MilvusClient,
    DataType
)

COLLECTION_NAME = "Documents"
EMBEDDING_DIM = 16
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
struct_schema.add_field("struct_float_vec2", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
schema.add_field("struct_field",datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema, max_capacity=1000)

milvus_client.create_collection(COLLECTION_NAME, schema=schema)

index_params = milvus_client.prepare_index_params()
dense_index_params = {
    "index_type": "IVF_FLAT",  # Choose an appropriate index type (e.g., IVF_FLAT)
    "metric_type": "COSINE",   # Metric type (e.g., COSINE for cosine similarity)
    "params": {"nlist": 128}   # Index parameters
}

index_params.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="COSINE", index_params={"nlist": 128})
index_params.add_index(field_name="struct_float_vec", index_type="EMB_LIST_HNSW", index_name="struct_float_vec_index1", metric_type="MAX_SIM", index_params={"nlist": 128})
index_params.add_index(field_name="struct_float_vec2", index_type="EMB_LIST_HNSW", index_name="struct_float_vec_index2",metric_type="MAX_SIM", index_params={"nlist": 128})
milvus_client.create_index(COLLECTION_NAME, schema=schema, index_params=index_params)

coll = milvus_client.describe_collection(COLLECTION_NAME)
print(coll)

rng = np.random.default_rng(seed=19530)

N = 20
content = ["aaa",
           "jjj",
           "sss",
           "asdasd",
           "dvzicxv",
           "dcmxvxv",
           "dvxcnvmlzc",
           "aczxcc",
           "qiqixcc"]
iterations = 100
for _ in range(iterations):
    arr_len = random.randint(3, 10)
    rows = [{"embedding": rng.random((1, 16))[0],
             "content": content[random.randint(0, len(content) - 1)],
             "struct_field": [
                 {"struct_str": content[random.randint(0, len(content) - 1)],
                  "struct_float_vec": rng.random((1, EMBEDDING_DIM))[0],
                  "struct_float_vec2": rng.random((1, EMBEDDING_DIM))[0],
                  }
                  for _ in range(arr_len)],
             }
            for _ in range(N)]
    result = milvus_client.insert(COLLECTION_NAME, rows)

milvus_client.flush(COLLECTION_NAME)