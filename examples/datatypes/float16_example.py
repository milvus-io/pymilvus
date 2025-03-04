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

# float16, little endian
fp16_little = np.dtype('e').newbyteorder('<')

def gen_fp16_vectors(num, dim):
    raw_vectors = []
    fp16_vectors = []
    for _ in range(num):
        raw_vector = [random.random() for _ in range(dim)]
        raw_vectors.append(raw_vector)
        fp16_vector = np.array(raw_vector, dtype=fp16_little)
        fp16_vectors.append(fp16_vector)
    return raw_vectors, fp16_vectors

def fp16_vector_search():
    connections.connect()

    int64_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=True)
    dim = 128
    nb = 3000
    vector_field_name = "float16_vector"
    fp16_vector = FieldSchema(name=vector_field_name, dtype=DataType.FLOAT16_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[int64_field, fp16_vector])

    if utility.has_collection("hello_milvus_fp16"):
        utility.drop_collection("hello_milvus_fp16")

    hello_milvus = Collection("hello_milvus_fp16", schema)

    _, vectors = gen_fp16_vectors(nb, dim)
    hello_milvus.insert([vectors[:6]])
    rows = [
        {vector_field_name: vectors[6]},
        {vector_field_name: vectors[7]},
        {vector_field_name: vectors[8]},
        {vector_field_name: vectors[9]},
        {vector_field_name: vectors[10]},
        {vector_field_name: vectors[11]},
    ]
    hello_milvus.insert(rows)
    hello_milvus.flush()

    for i, index_type in enumerate(fp16_index_types):
        index_params = default_fp16_index_params[i]
        hello_milvus.create_index(vector_field_name,
                                  index_params={"index_type": index_type, "params": index_params, "metric_type": "L2"})
        hello_milvus.load()
        print("index_type = ", index_type)
        res = hello_milvus.search(vectors[0:10], vector_field_name, {"metric_type": "L2"}, limit=1, output_fields=["float16_vector"])
        print("raw bytes: ", res[0][0].get("float16_vector"))
        print("numpy ndarray: ", np.frombuffer(res[0][0].get("float16_vector"), dtype=fp16_little))
        hello_milvus.release()
        hello_milvus.drop_index()

    hello_milvus.drop()

if __name__ == "__main__":
    fp16_vector_search()
