import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    AnnSearchRequest, RRFRanker, WeightedRanker,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")
if has:
    utility.drop_collection("hello_milvus")

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="embeddings2", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong", num_shards = 4)

print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),  # field random, only supports list
    rng.random((num_entities, dim)),    # field embeddings, supports numpy.ndarray and list
    rng.random((num_entities, dim)),    # field embeddings2, supports numpy.ndarray and list
]

insert_result = hello_milvus.insert(entities)

hello_milvus.flush()
print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entities

print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)
hello_milvus.create_index("embeddings2", index)

print(fmt.format("Start loading"))
hello_milvus.load()

field_names = ["embeddings", "embeddings2"]

req_list = []
nq = 1
weights = [0.2, 0.3]
default_limit = 5
vectors_to_search = []

for i in range(len(field_names)):
    # 4. generate search data
    vectors_to_search = rng.random((nq, dim))
    search_param = {
        "data": vectors_to_search,
        "anns_field": field_names[i],
        "param": {"metric_type": "L2"},
        "limit": default_limit,
        "expr": "random > 0.5"}
    req = AnnSearchRequest(**search_param)
    req_list.append(req)

print(fmt.format("rank by WightedRanker"))
hybrid_res = hello_milvus.hybrid_search(req_list, WeightedRanker(*weights, norm_score=True), default_limit, output_fields=["random"])
for hits in hybrid_res:
    for hit in hits:
        print(f" hybrid search hit: {hit}")

print(fmt.format("rank by RRFRanker"))
hybrid_res = hello_milvus.hybrid_search(req_list, RRFRanker(), default_limit, output_fields=["random"])
for hits in hybrid_res:
    for hit in hits:
        print(f" hybrid search hit: {hit}")

req_list = []
for i in range(len(field_names)):
    # 4. generate search data
    vectors_to_search = rng.random((nq, dim))
    search_param = {
        "data": vectors_to_search,
        "anns_field": field_names[i],
        "param": {"metric_type": "L2"},
        "limit": default_limit,
        "expr": "random > {radius}",
        "expr_params": {"radius": 0.5}}
    req = AnnSearchRequest(**search_param)
    req_list.append(req)

print(fmt.format("rank by WightedRanker with expression template"))
hybrid_res = hello_milvus.hybrid_search(req_list, WeightedRanker(*weights), default_limit, output_fields=["random"])
for hits in hybrid_res:
    for hit in hits:
        print(f" hybrid search hit: {hit}")

print(fmt.format("rank by RRFRanker with expression template"))
hybrid_res = hello_milvus.hybrid_search(req_list, RRFRanker(), default_limit, output_fields=["random"])
for hits in hybrid_res:
    for hit in hits:
        print(f" hybrid search hit: {hit}")
