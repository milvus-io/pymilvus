import numpy as np
from pymilvus import MilvusClient, DataType

dimension = 128
collection_name = "books"
client = MilvusClient("http://localhost:19530")
client.drop_collection(collection_name)

schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dimension)
schema.add_field("title", DataType.VARCHAR, max_length=64)

index_param = client.prepare_index_params("embeddings", metric_type="L2")
client.create_collection_with_schema(collection_name, schema, index_param)

info = client.describe_collection(collection_name)
print(f"{collection_name}'s info:{info}")

rng = np.random.default_rng(seed=19530)
rows = [
    {"title": "The Catcher in the Rye", "embeddings": rng.random((1, dimension))[0], "a":1,},
    {"title": "Lord of the Flies", "embeddings": rng.random((1, dimension))[0], "b":2},
    {"title": "The Hobbit", "embeddings": rng.random((1, dimension))[0]},
    {"title": "The Outsiders", "embeddings": rng.random((1, dimension))[0]},
    {"title": "The Old Man and the Sea", "embeddings": rng.random((1, dimension))[0]},
]

client.insert(collection_name, rows)
client.insert(collection_name,
        {"title": "The Great Gatsby", "embeddings": rng.random((1, dimension))[0]})

search_vec = rng.random((1, dimension))
result = client.search(collection_name, search_vec, limit=3, output_fields=["title"])
# we may get empty result
for i, hits in enumerate(result):
    if not hits:
        print(f"get empty results for search_vec[{i}]")
        continue
    for hit in hits:
        print(f"hit: {hit}")

# use strong consistency level ensure that we can see the data we inserted before
result = client.search(collection_name, search_vec, limit=3, output_fields=["title", "*"], consistency_level="Strong")
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")
