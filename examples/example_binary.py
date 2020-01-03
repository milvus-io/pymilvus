import sys

sys.path.append(".")

import random
import datetime
import numpy as np

from milvus import *

TABLE_NAME = "test_debug_medines"
DIM = 512

TABLE_PARAM = {
    "table_name": TABLE_NAME,
    "dimension": DIM,
    "index_file_size": 1024,
    "metric_type": MetricType.JACCARD
}


def load_data():
    file_path = "test_1w.csv"

    data = list()
    with open(file_path, "r") as f:
        while True:
            line = f.readline().strip("\n")
            if line:
                line_bytes = bytes.fromhex(line[3:])
                data.append(line_bytes)
            else:
                break

    return data


def gen_vectors(dim, num):
    return [[random.randint(0, 255) for _ in range(dim)] for _ in range(num)]


def gen_uint8_vectors(dim, num):
    vectors = gen_vectors(dim // 8, num)
    data = np.array(vectors, dtype='uint8').astype("uint8")

    bytes_list = list()
    # return data
    for vector in vectors:
        byte_vector = bytes(vector)

        len(byte_vector) #  == 512 / 8
        bytes_list.append(byte_vector)

    return bytes_list


if __name__ == '__main__':
    print("Loading data")
    # datas = load_data()
    datas = gen_uint8_vectors(DIM, 1000)
    print("Load done ... ")

    client = Milvus()
    client.connect(host="192.168.1.57", port="19530")

    _, ok = client.has_table(TABLE_NAME)

    if not ok:
        client.create_table(TABLE_PARAM)

    status, ids = client.insert(TABLE_NAME, datas)
    print(status, len(ids))
