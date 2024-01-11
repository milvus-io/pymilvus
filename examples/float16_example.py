import time
import random
import numpy as np
from pymilvus import (
     connections,
     utility,
     FieldSchema, CollectionSchema, DataType,
     Collection,
 )
from pymilvus import MilvusClient

fp16_index_types = ["FLAT"]

default_fp16_index_params = [{"nlist": 128}]

def gen_fp16_vectors(num, dim):
    raw_vectors = []
    fp16_vectors = []
    for _ in range(num):
        raw_vector = [random.random() for _ in range(dim)]
        raw_vectors.append(raw_vector)
        fp16_vector = np.array(raw_vector, dtype=np.float16).view(np.uint8).tolist()
        fp16_vectors.append(bytes(fp16_vector))
    return raw_vectors, fp16_vectors

def fp16_vector_search():
    connections.connect()

    int64_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=True)
    dim = 128
    nb = 3000
    vector_field_name = "float16_vector"
    fp16_vector = FieldSchema(name=vector_field_name, dtype=DataType.FLOAT16_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[int64_field, fp16_vector])

    has = utility.has_collection("hello_milvus")
    if has:
        hello_milvus = Collection("hello_milvus_fp16")
        hello_milvus.drop()
    else:
        hello_milvus = Collection("hello_milvus_fp16", schema)

    _, vectors = gen_fp16_vectors(nb, dim)
    rows = [
        {vector_field_name: vectors[0]},
        {vector_field_name: vectors[1]},
        {vector_field_name: vectors[2]},
        {vector_field_name: vectors[3]},
        {vector_field_name: vectors[4]},
        {vector_field_name: vectors[5]},
    ]

    hello_milvus.insert(rows)
    hello_milvus.flush()

    for i, index_type in enumerate(fp16_index_types):
        index_params = default_fp16_index_params[i]
        hello_milvus.create_index(vector_field_name,
                                  index_params={"index_type": index_type, "params": index_params, "metric_type": "L2"})
        hello_milvus.load()
        print("index_type = ", index_type)
        res = hello_milvus.search(vectors[0:10], vector_field_name, {"metric_type": "L2"}, limit=1)
        print(res)
        hello_milvus.release()
        hello_milvus.drop_index()

    hello_milvus.drop()

if __name__ == "__main__":
    fp16_vector_search()