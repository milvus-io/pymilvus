import random
import time
import numpy as np

from milvus import *

TABLE_NAME = "binary_table"
DIM = 512

TABLE_PARAM = {
    "table_name": TABLE_NAME,
    "dimension": DIM,
    "index_file_size": 1024,
    # MetricType `HAMMING`, `JACCARD`, `TANIMOTO` are only used for binary vectors
    "metric_type": MetricType.JACCARD
}


def gen_vectors(dim, num):
    return [[random.randint(0, 255) for _ in range(dim)] for _ in range(num)]


def gen_binary_vectors(dim, num):
    # in binary vectors, a bit represent a dimensionality
    vectors = gen_vectors(dim // 8, num)
    data = np.array(vectors, dtype='uint8').astype("uint8")

    # convert to bytes
    return [bytes(vector) for vector in vectors]


if __name__ == '__main__':
    datas = gen_binary_vectors(DIM, 1000)

    client = Milvus()
    client.connect(host="localhost", port="19530")

    _, ok = client.has_table(TABLE_NAME)

    if not ok:
        client.create_table(TABLE_PARAM)

    status, _ = client.insert(TABLE_NAME, datas)
    if 0 != status.code:
        print("Insert binary vectors into talbe {} failed.".format(TABLE_NAME))
        exit(1)

    # Wait for 6 seconds, until Milvus server persist vector data.
    time.sleep(6)

    # select top 3 vectors for similarity search
    query_vectors = datas[0:3]

    status, results = client.search(TABLE_NAME, top_k=1, nprobe=1, query_records=query_vectors)
    if 0 != status.code:
        print("Search failed. reason:{}".format(status.message))

    print(results)

    client.drop_table(TABLE_NAME)
    client.disconnect()
