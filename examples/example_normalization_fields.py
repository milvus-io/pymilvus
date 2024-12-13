import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8


print(fmt.format("start connecting to Milvus"))
# this is milvus standalone
connection = connections.connect(
  alias="default", 
  host='localhost', # or '0.0.0.0' or 'localhost'
  port='19530'
)

client = MilvusClient(connections=connection)

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")
if has:
    utility.drop_collection("hello_milvus")

fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings1", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="embeddings2", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus`"))

print(fmt.format("Message for handling an invalid format in the normalization_fields value")) # you can try with other value like: dict,...
try:
    hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong", normalization_fields='embeddings1')
except BaseException as e:
    print(e)
    

print(fmt.format("Message for handling the invalid vector fields"))
try:
    hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong", normalization_fields=['embddings'])
except BaseException as e:
    print(e)

print(fmt.format("Insert data, without conversion to standard form"))

hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")

print(fmt.format("Start inserting a row"))
rng = np.random.default_rng(seed=19530)

row = {
    "pk": "19530",
    "random": 0.5,
    "embeddings1": rng.random((1, dim), np.float32)[0],
    "embeddings2": rng.random((1, dim), np.float32)[0]
}
hello_milvus.insert(row)
utility.drop_collection("hello_milvus")
    
print(fmt.format("Insert data, with conversion to standard form"))

hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong", normalization_fields=['embeddings1'])

print(fmt.format("Start inserting a row"))
rng = np.random.default_rng(seed=19530)

row = {
    "pk": "19530",
    "random": 0.5,
    "embeddings1": rng.random((1, dim), np.float32)[0],
    "embeddings2": rng.random((1, dim), np.float32)[0]
}
_row = row.copy()
hello_milvus.insert(row)

index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
hello_milvus.create_index("embeddings1", index_param)
hello_milvus.create_index("embeddings2", index_param)
hello_milvus.load()

original_vector = _row['embeddings1']
insert_vector = hello_milvus.query(
    expr="pk == '19530'",
    output_fields=["embeddings1"],
)[0]['embeddings1']

print(fmt.format("Mean and standard deviation before normalization."))
print("Mean: ", np.mean(original_vector))
print("Std: ", np.std(original_vector))

print(fmt.format("Mean and standard deviation after normalization."))
print("Mean: ", np.mean(insert_vector))
print("Std: ", np.std(insert_vector))


print(fmt.format("Start inserting entities"))

entities = [
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),
    rng.random((num_entities, dim), np.float32),
    rng.random((num_entities, dim), np.float32),
]

insert_result = hello_milvus.insert(entities)

insert_vector = hello_milvus.query(
    expr="pk == '1'",
    output_fields=["embeddings1"],
)[0]['embeddings1']

print(fmt.format("Mean and standard deviation after normalization."))
print("Mean: ", np.mean(insert_vector))
print("Std: ", np.std(insert_vector))

utility.drop_collection("hello_milvus")