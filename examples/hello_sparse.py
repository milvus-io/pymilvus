# hello_sprase.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus,
# while operating on sparse float vectors.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
import time

import numpy as np
import random
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "=== {:30} ==="
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 1000, 3000
# non zero count of randomly generated sparse vectors
nnz = 30

def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + msg)

# -----------------------------------------------------------------------------
# connect to Milvus
log(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_sparse")
log(f"Does collection hello_sparse exist in Milvus: {has}")

# -----------------------------------------------------------------------------
# create collection with a sparse float vector column
hello_sparse = None
if not has:
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="random", dtype=DataType.DOUBLE),
        FieldSchema(name="embeddings", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields, "hello_sparse is the simplest demo to introduce sparse float vector usage")
    log(fmt.format("Create collection `hello_sparse`"))
    hello_sparse = Collection("hello_sparse", schema, consistency_level="Strong")
else:
    hello_sparse = Collection("hello_sparse")

log(f"hello_sparse has {hello_sparse.num_entities} entities({hello_sparse.num_entities/1000000}M), indexed {hello_sparse.has_index()}")

# -----------------------------------------------------------------------------
# insert
log(fmt.format("Start creating entities to insert"))
rng = np.random.default_rng(seed=19530)

def generate_sparse_vector(dimension: int, non_zero_count: int) -> dict:
    indices = random.sample(range(dimension), non_zero_count)
    values = [random.random() for _ in range(non_zero_count)]
    sparse_vector = {index: value for index, value in zip(indices, values)}
    return sparse_vector

entities = [
    rng.random(num_entities).tolist(),
    [generate_sparse_vector(dim, nnz) for _ in range(num_entities)],
]

log(fmt.format("Start inserting entities"))
insert_result = hello_sparse.insert(entities)

# -----------------------------------------------------------------------------
# create index
if not hello_sparse.has_index():
    log(fmt.format("Start Creating index SPARSE_INVERTED_INDEX"))
    index = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
        "params":{
            "drop_ratio_build": 0.2,
        }
    }
    hello_sparse.create_index("embeddings", index)

log(fmt.format("Start loading"))
hello_sparse.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
log(fmt.format("Start searching based on vector similarity"))
vectors_to_search = entities[-1][-1:]
search_params = {
    "metric_type": "IP",
    "params": {
        "drop_ratio_search": "0.2",
    }
}

start_time = time.time()
result = hello_sparse.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["pk", "random", "embeddings"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}")
log(search_latency_fmt.format(end_time - start_time))
# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
print(fmt.format("Start querying with `random > 0.5`"))

start_time = time.time()
result = hello_sparse.query(expr="random > 0.5", output_fields=["random", "embeddings"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# pagination
r1 = hello_sparse.query(expr="random > 0.5", limit=4, output_fields=["random"])
r2 = hello_sparse.query(expr="random > 0.5", offset=1, limit=3, output_fields=["random"])
print(f"query pagination(limit=4):\n\t{r1}")
print(f"query pagination(offset=1, limit=3):\n\t{r2}")


# -----------------------------------------------------------------------------
# hybrid search
print(fmt.format("Start hybrid searching with `random > 0.5`"))

start_time = time.time()
result = hello_sparse.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5", output_fields=["random"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, random field: {hit.entity.get('random')}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# delete entities by PK
# You can delete entities by their PK values using boolean expressions.
ids = insert_result.primary_keys

expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
print(fmt.format(f"Start deleting with expr `{expr}`"))

result = hello_sparse.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

hello_sparse.delete(expr)

result = hello_sparse.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query after delete by expr=`{expr}` -> result: {result}\n")


# -----------------------------------------------------------------------------
# drop collection
print(fmt.format("Drop collection `hello_sparse`"))
utility.drop_collection("hello_sparse")
