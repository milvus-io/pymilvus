import random

import numpy as np

from pymilvus import (
    DataType,
    MilvusClient,
)

default_int8_index_param = {"M": 8, "efConstruction": 200}


def gen_int8_vectors(num: int, dim: int):
    raw_vectors = []
    int8_vectors = []
    for _ in range(num):
        raw_vector = [random.randint(-128, 127) for _ in range(dim)]
        raw_vectors.append(raw_vector)
        int8_vector = np.array(raw_vector, dtype=np.int8)
        int8_vectors.append(int8_vector)
    return raw_vectors, int8_vectors


def int8_vector_search():
    c = MilvusClient()

    dim = 128
    nb = 3000
    collection_name = "hello_milvus_int8"

    schema = MilvusClient.create_schema()
    schema.add_field("int64", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("int8_vector", DataType.INT8_VECTOR, dim=dim)

    if c.has_collection(collection_name):
        c.drop_collection(collection_name)
    c.create_collection(collection_name, schema=schema)

    _, vectors = gen_int8_vectors(nb, dim)
    rows = [{"int8_vector": vectors[i] for i in range(12)}]
    c.insert(collection_name, rows)
    c.flush(collection_name)

    index_params = MilvusClient.prepare_index_params(
        field_name="int8_vector",
        index_type="HNSW",
        index_name="int8_index",
        **default_int8_index_param,
    )
    c.create_index(collection_name, index_params)
    c.load_collection(collection_name)
    res = c.search(
        collection_name=collection_name,
        data=vectors[0:10],
        search_params={"metric_type": "L2"},
        limit=1, output_fields=["int8_vector"],
    )
    print("raw bytes: ", res[0][0].get("int8_vector"))
    print("numpy ndarray: ", np.frombuffer(res[0][0].get("int8_vector"), dtype=np.int8))
    c.release_collection(collection_name)
    c.drop_collection(collection_name)


if __name__ == "__main__":
    int8_vector_search()
