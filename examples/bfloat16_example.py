import time
import random
import numpy as np
import tensorflow as tf
from pymilvus import (
     connections,
     utility,
     FieldSchema, CollectionSchema, DataType,
     Collection,
 )
from pymilvus import MilvusClient

bf16_index_types = ["FLAT"]

default_bf16_index_params = [{"nlist": 128}]

def gen_bf16_vectors(num, dim):
    raw_vectors = []
    bf16_vectors = []
    for _ in range(num):
        raw_vector = [random.random() for _ in range(dim)]
        raw_vectors.append(raw_vector)
        # bf16_vector = np.array(raw_vector, dtype=tf.bfloat16).view(np.uint8).tolist()
        bf16_vector = tf.cast(raw_vector, dtype=tf.bfloat16).numpy().view(np.uint8).tolist()
        bf16_vectors.append(bytes(bf16_vector))
    return raw_vectors, bf16_vectors

def bf16_vector_search():
    connections.connect()

    int64_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=True)
    dim = 128
    nb = 3000
    vector_field_name = "bfloat16_vector"
    bf16_vector = FieldSchema(name=vector_field_name, dtype=DataType.BFLOAT16_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[int64_field, bf16_vector])

    has = utility.has_collection("hello_milvus")
    if has:
        hello_milvus = Collection("hello_milvus_fp16")
        hello_milvus.drop()
    else:
        hello_milvus = Collection("hello_milvus_fp16", schema)

    _, vectors = gen_bf16_vectors(nb, dim)
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

    for i, index_type in enumerate(bf16_index_types):
        index_params = default_bf16_index_params[i]
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
    bf16_vector_search()