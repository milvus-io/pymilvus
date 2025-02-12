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

int8_index_types = ["HNSW"]

default_int8_index_params = [{"M": 8, "efConstruction": 200}]


def gen_int8_vectors(num, dim):
    raw_vectors = []
    int8_vectors = []
    for _ in range(num):
        raw_vector = [random.randint(-128, 127) for _ in range(dim)]
        raw_vectors.append(raw_vector)
        int8_vector = np.array(raw_vector, dtype=np.int8)
        int8_vectors.append(int8_vector)
    return raw_vectors, int8_vectors


def int8_vector_search():
    connections.connect()

    int64_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=True)
    dim = 128
    nb = 3000
    vector_field_name = "int8_vector"
    int8_vector = FieldSchema(name=vector_field_name, dtype=DataType.INT8_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[int64_field, int8_vector])

    if utility.has_collection("hello_milvus_int8"):
        utility.drop_collection("hello_milvus_int8")

    hello_milvus = Collection("hello_milvus_int8", schema)

    _, vectors = gen_int8_vectors(nb, dim)
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

    for i, index_type in enumerate(int8_index_types):
        index_params = default_int8_index_params[i]
        hello_milvus.create_index(vector_field_name,
                                  index_params={"index_type": index_type, "params": index_params, "metric_type": "L2"})
        hello_milvus.load()
        print("index_type = ", index_type)
        res = hello_milvus.search(vectors[0:10], vector_field_name, {"metric_type": "L2"}, limit=1, output_fields=["int8_vector"])
        print("raw bytes: ", res[0][0].get("int8_vector"))
        print("numpy ndarray: ", np.frombuffer(res[0][0].get("int8_vector"), dtype=np.int8))
        hello_milvus.release()
        hello_milvus.drop_index()

    hello_milvus.drop()


if __name__ == "__main__":
    int8_vector_search()
