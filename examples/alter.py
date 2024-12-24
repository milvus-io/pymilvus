import numpy as np
from pymilvus import (
    MilvusClient,
    DataType
)

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
milvus_client.create_collection(collection_name, schema=schema)
rng = np.random.default_rng(seed=19530)
rows = [
        {"id": 1, "embeddings": rng.random((1, dim))[0], "a": 100, "title": "t1"},
        {"id": 2, "embeddings": rng.random((1, dim))[0], "b": 200, "title": "t2"},
        {"id": 3, "embeddings": rng.random((1, dim))[0], "c": 300, "title": "t3"},
        {"id": 4, "embeddings": rng.random((1, dim))[0], "d": 400, "title": "t4"},
        {"id": 5, "embeddings": rng.random((1, dim))[0], "e": 500, "title": "t5"},
        {"id": 6, "embeddings": rng.random((1, dim))[0], "f": 600, "title": "t6"},
]

insert_result = milvus_client.insert(collection_name, rows)
index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name = "embeddings", metric_type="L2")
milvus_client.create_index(collection_name, index_params)
milvus_client.alter_index_properties(collection_name,index_name="embeddings", properties={"mmap.enabled":True})
milvus_client.drop_index_properties(collection_name,index_name="embeddings", property_keys=["mmap.enabled"])
milvus_client.alter_collection_field(collection_name,field_name="title",field_params={"mmap.enabled":True,"max_length": 2500})
milvus_client.alter_collection_properties(collection_name, properties={"mmap.enabled":True,"collection.ttl.seconds": 500})
milvus_client.drop_collection_properties(collection_name,index_name="embeddings", property_keys=["mmap.enabled"])
milvus_client.load_collection(collection_name)
print(milvus_client.describe_index(collection_name,index_name="embeddings"))
print(milvus_client.describe_collection(collection_name))
milvus_client.release_collection(collection_name)
milvus_client.drop_index(collection_name, "embeddings")
milvus_client.drop_collection(collection_name)
