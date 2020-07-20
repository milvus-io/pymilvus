# This program demos how to use binary vectors in milvus.
import random
import sys

import numpy as np

from milvus import *

COLLECTION_NAME = "binary_collection"
DIM = 512

COLLECTION_PARAM = {
    # "collection_name": COLLECTION_NAME,
    "dimension": DIM,
    "index_file_size": 1024,
    # MetricType `HAMMING`, `JACCARD`, `TANIMOTO` are only used for binary vectors
    # "metric_type": MetricType.JACCARD
}


def gen_vectors(dim, num):
    return [[random.randint(0, 255) for _ in range(dim)] for _ in range(num)]


# gen binary vectors
def gen_binary_vectors(dim, num):
    # in binary vectors, a bit represent a dimension.
    # a value of `uint8` describe 8 dimension
    vectors = gen_vectors(dim // 8, num)
    data = np.array(vectors, dtype='uint8').astype("uint8")

    # convert to bytes
    return [bytes(vector) for vector in data]


def main(collection_name, metric_type):
    client = Milvus(host="localhost", port="19121", handler="HTTP")

    _, ok = client.has_collection(collection_name)

    if not ok:
        COLLECTION_PARAM["collection_name"] = collection_name
        COLLECTION_PARAM["metric_type"] = metric_type
        status = client.create_collection(COLLECTION_PARAM)
        if not status.OK():
            print("Create collection {} fail: {}".format(collection_name, status))
            sys.exit(1)

    datas = gen_binary_vectors(DIM, 1000)
    status, _ = client.insert(collection_name, datas)
    if 0 != status.code:
        print("Insert binary vectors into collection {} failed: {}".format(collection_name, status))
        exit(1)

    # Wait for 6 seconds, until Milvus server persist vector data.
    client.flush([collection_name])

    # select top 3 vectors for similarity search
    query_vectors = datas[0:3]

    # search param
    search_param = {
        "nprobe": 10
    }

    status, results = client.search(collection_name, top_k=2, query_records=query_vectors, params=search_param)
    if 0 != status.code:
        print("Search failed. reason:{}".format(status.message))
    else:
        print(results)

    client.drop_collection(collection_name)


if __name__ == '__main__':
    # Here is all binary metric type in list.
    binary_metrics = [MetricType.HAMMING, MetricType.JACCARD, MetricType.TANIMOTO,
                      MetricType.SUBSTRUCTURE, MetricType.SUPERSTRUCTURE]

    for metric in binary_metrics:
        collection_name = "{}_{}".format(COLLECTION_NAME, metric.name)
        print("\n\n*** Collection: {}   Metric: {} ***\n\n".format(collection_name, metric))
        main(collection_name, metric)
