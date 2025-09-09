import time
import random
import numpy as np
from pymilvus import (
     connections,
     utility,
     FieldSchema, CollectionSchema, DataType,
     Collection,
 )


bin_index_types = ["BIN_FLAT", "BIN_IVF_FLAT"]

default_bin_index_params = [{"nlist": 128}, {"nlist": 128}]

def gen_binary_vectors(num, dim):
    raw_vectors = []
    binary_vectors = []
    for _ in range(num):
        raw_vector = [random.randint(0, 1) for _ in range(dim)]
        raw_vectors.append(raw_vector)
        # packs a binary-valued array into bits in a unit8 array, and bytes array_of_ints
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return raw_vectors, binary_vectors


def binary_vector_search():
    connections.connect()
    int64_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, auto_id=True)
    dim = 128
    nb = 3000
    vector_field_name = "binary_vector"
    binary_vector = FieldSchema(name=vector_field_name, dtype=DataType.BINARY_VECTOR, dim=dim)
    schema = CollectionSchema(fields=[int64_field, binary_vector], enable_dynamic_field=True)

    has = utility.has_collection("hello_milvus")
    if has:
        hello_milvus = Collection("hello_milvus_bin")
        hello_milvus.drop()
    else:
        hello_milvus = Collection("hello_milvus_bin", schema)

    _, vectors = gen_binary_vectors(nb, dim)
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
    for i, index_type in enumerate(bin_index_types):
        index_params = default_bin_index_params[i]
        hello_milvus.create_index(vector_field_name,
                                  index_params={"index_type": index_type, "params": index_params, "metric_type": "HAMMING"})
        hello_milvus.load()
        print("index_type = ", index_type)
        res = hello_milvus.search(vectors[:1], vector_field_name, {"metric_type": "HAMMING"}, limit=1)
        print("res = ", res)
        hello_milvus.release()
        hello_milvus.drop_index()
    hello_milvus.drop()


if __name__ == "__main__":
    binary_vector_search()
