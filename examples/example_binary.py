# This program demos how to use binary vectors in milvus.
import random
import time
import numpy as np

from milvus import *

COLLECTION_NAME = "binary_collection"
DIM = 512

COLLECTION_PARAM = {
    "collection_name": COLLECTION_NAME,
    "dimension": DIM,
    "index_file_size": 1024,
    # MetricType `HAMMING`, `JACCARD`, `TANIMOTO` are only used for binary vectors
    "metric_type": MetricType.JACCARD
}


def gen_vectors(dim, num):
    return [[random.randint(0, 255) for _ in range(dim)] for _ in range(num)]


## gen binary vectors
def gen_binary_vectors(dim, num):
    # in binary vectors, a bit represent a dimension.
    # a value of `uint8` describe 8 dimension
    vectors = gen_vectors(dim // 8, num)
    data = np.array(vectors, dtype='uint8').astype("uint8")

    # convert to bytes
    return [bytes(vector) for vector in data]


if __name__ == '__main__':
    datas = gen_binary_vectors(DIM, 1000)

    client = Milvus()
    client.connect(host="localhost", port="19530")

    _, ok = client.has_collection(COLLECTION_NAME)

    if not ok:
        client.create_collection(COLLECTION_PARAM)

    status, _ = client.insert(COLLECTION_NAME, datas)
    if 0 != status.code:
        print("Insert binary vectors into collection {} failed.".format(COLLECTION_NAME))
        exit(1)

    # Wait for 6 seconds, until Milvus server persist vector data.
    time.sleep(6)

    # select top 3 vectors for similarity search
    query_vectors = datas[0:3]

    # search param
    search_param = {
        "nprobe": 10
    }

    status, results = client.search(COLLECTION_NAME, top_k=2, query_records=query_vectors, params=search_param)
    if 0 != status.code:
        print("Search failed. reason:{}".format(status.message))
    else:
        print(results)

    client.drop_collection(COLLECTION_NAME)
    client.disconnect()
