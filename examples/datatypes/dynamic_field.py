import time
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
dim = 8

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")
if has:
    utility.drop_collection("hello_milvus")

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs", enable_dynamic_field=True)

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")

################################################################################
# 3. insert data
hello_milvus2 = Collection("hello_milvus")
print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)

rows = [
        {"pk": "1", "random": 1.0, "embeddings": rng.random((1, dim))[0], "a": 1},
        {"pk": "2", "random": 1.0, "embeddings": rng.random((1, dim))[0], "b": 1},
        {"pk": "3", "random": 1.0, "embeddings": rng.random((1, dim))[0], "c": 1},
        {"pk": "4", "random": 1.0, "embeddings": rng.random((1, dim))[0], "d": 1},
        {"pk": "5", "random": 1.0, "embeddings": rng.random((1, dim))[0], "e": 1},
        {"pk": "6", "random": 1.0, "embeddings": rng.random((1, dim))[0], "f": 1},
        ]

insert_result = hello_milvus.insert(rows)

hello_milvus.insert({"pk": "7", "random": 1.0, "embeddings": rng.random((1, dim))[0], "g": 1})
hello_milvus.flush()
print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entites

# 4. create index
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)

print(fmt.format("Start loading"))
hello_milvus.load()
# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))

rng = np.random.default_rng(seed=19530)
vectors_to_search = rng.random((1, dim))
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["pk", "embeddings"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}")


result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["pk", "embeddings", "$meta"])
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

expr = f'pk in ["1" , "2"] || g == 1'

print(fmt.format(f"Start query with expr `{expr}`"))
result = hello_milvus.query(expr=expr, output_fields=["random", "a", "g"])
for hit in result:
    print("hit:", hit)

###############################################################################
# 7. drop collection
print(fmt.format("Drop collection `hello_milvus`"))
utility.drop_collection("hello_milvus")
