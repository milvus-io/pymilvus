import random
import sys
import math
import time
import numpy as np

from pymilvus import Milvus, DataType

# This example shows how to use milvus to calculate vectors distance

_HOST = '127.0.0.1'
_PORT = '19530'

# Create milvus client instance
milvus = Milvus(_HOST, _PORT)

_PRECISION = 1e-3

def gen_float_vectors(num, dim):
    vec_list = [[random.random() for _ in range(dim)] for _ in range(num)]
    return vec_list

def gen_binary_vectors(num, dim):
    zero_fill = 0
    if dim%8 > 0:
        zero_fill = 8 - dim%8
    binary_vectors = []
    raw_vectors = []
    for i in range(num):
        raw_vector = [random.randint(0, 1) for i in range(dim)]
        for k in range(zero_fill):
            raw_vector.append(0)
        raw_vectors.append(raw_vector)
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return binary_vectors, raw_vectors

def l2_distance(vec_l, vec_r, sqrt = False):
    if len(vec_l) != len(vec_r):
        return -1.0

    dist = 0.0
    for i in range(len(vec_l)):
        dist = dist + math.pow(vec_l[i]-vec_r[i], 2)
    if sqrt:
        dist = math.sqrt(dist)
    return dist

def ip_distance(vec_l, vec_r):
    if len(vec_l) != len(vec_r):
        return -1.0

    dist = 0.0
    for i in range(len(vec_l)):
        dist = dist + vec_l[i]*vec_r[i]
    return dist

def calc_float_distance(l_count, r_count, dim, metric):
    vectors_l = gen_float_vectors(l_count, dim)
    vectors_r = gen_float_vectors(r_count, dim)

    op_l = {"float_vectors": vectors_l}
    op_r = {"float_vectors": vectors_r}

    sqrt = True
    params = {"metric": metric, "sqrt": sqrt}
    time_start = time.time()
    results = milvus.calc_distance(vectors_left=op_l, vectors_right=op_r, params=params)
    time_end = time.time()
    print(metric, "distance(","l_count",l_count, "r_count",r_count, "dim",dim,") time cost", (time_end-time_start)*1000, "ms")
    print(results)
    if len(results) != l_count * r_count:
        print("Incorrect results returned by calc_distance()")

    all_correct = True
    for i in range(l_count):
        vec_l = vectors_l[i]
        for j in range(r_count):
            vec_r = vectors_r[j]
            dist_1 = l2_distance(vec_l, vec_r, sqrt) if metric == "L2" else ip_distance(vec_l, vec_r)
            dist_2 = results[i*r_count+j]
            if math.fabs(dist_1 - dist_2) > _PRECISION:
                print("Incorrect result", dist_1, "vs", dist_2)
                all_correct = False

    if all_correct:
        print("All distance are correct\n")

def hamming_distance(vec_l, vec_r):
    if len(vec_l) != len(vec_r):
        return -1

    hamming = 0
    for i in range(len(vec_l)):
        if vec_l[i] != vec_r[i]:
            hamming = hamming + 1
    return hamming

def tanimoto_distance(vec_l, vec_r, dim):
    if len(vec_l) != len(vec_r):
        return -1

    hamming = hamming_distance(vec_l, vec_r)
    return hamming/(dim*2-hamming)

def calc_binary_distance(l_count, r_count, dim, metric):
    vectors_l, raw_l = gen_binary_vectors(l_count, dim)
    vectors_r, raw_r = gen_binary_vectors(r_count, dim)

    op_l = {"bin_vectors": vectors_l}
    op_r = {"bin_vectors": vectors_r}

    params = {"metric": metric, "dim": dim}
    time_start = time.time()
    results = milvus.calc_distance(vectors_left=op_l, vectors_right=op_r, params=params)
    time_end = time.time()
    print(metric, "distance(","l_count",l_count, "r_count",r_count, "dim",dim,") time cost", (time_end-time_start)*1000, "ms")
    print(results)
    if len(results) != l_count * r_count:
        print("Incorrect results returned by calc_distance()")

    all_correct = True
    for i in range(l_count):
        vec_l = raw_l[i]
        for j in range(r_count):
            vec_r = raw_r[j]
            dist_1 = hamming_distance(vec_l, vec_r) if metric == "HAMMING" else tanimoto_distance(vec_l, vec_r, dim)
            dist_2 = results[i * r_count + j]
            if math.fabs(dist_1 - dist_2) > _PRECISION:
                print("Incorrect result", dist_1, "vs", dist_2)
                all_correct = False

    if all_correct:
        print("All distance are correct\n")

def input_vectors_to_calc():
    calc_float_distance(100, 50, 256, "L2")
    calc_float_distance(5, 1000, 256, "IP")

    calc_binary_distance(10, 20, 99, "HAMMING")
    calc_binary_distance(5, 100, 1, "HAMMING")
    calc_binary_distance(100, 15, 16, "HAMMING")
    calc_binary_distance(10, 10, 7, "TANIMOTO")
    calc_binary_distance(500, 5, 1, "TANIMOTO")
    calc_binary_distance(50, 150, 16, "TANIMOTO")

def create_collection(collection, vec_field, dim):
    id_field = {
        "name": "id",
        "type": DataType.INT64,
        "auto_id": True,
        "is_primary": True,
    }
    vector_field = {
        "name": vec_field,
        "type": DataType.FLOAT_VECTOR,
        "metric_type": "L2",
        "params": {"dim": dim},
        "indexes": [{"metric_type": "L2"}]
    }
    fields = {"fields": [id_field, vector_field]}

    milvus.create_collection(collection_name=collection, fields=fields)
    print("collection created:", collection)

def drop_collection(collection):
    if milvus.has_collection(collection):
        milvus.drop_collection(collection)
        print("collection dropped:", collection)

def insert(collection, vec_field, num, dim):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    entities = [{"name": vec_field, "type": DataType.FLOAT_VECTOR, "values": vectors}]
    ids = milvus.insert(collection, entities)
    print("insert", len(list(ids._primary_keys)), "vectors into", collection)
    return list(ids._primary_keys), vectors

def flush(collection):
    milvus.flush([collection])
    print("collection flushed:", collection)

def load_collection(collection):
    milvus.load_collection(collection)
    print("collection loaded:", collection)

def insert_vectors_to_calc():
    collection_name = "test"
    vec_field = "vec"
    dim = 128
    drop_collection(collection_name)
    create_collection(collection_name, vec_field, dim)

    l_count = 10
    db_ids, db_vectors = insert(collection_name, vec_field, l_count, dim)
    flush(collection_name)
    load_collection(collection_name)

    r_count = 50
    vectors_l = gen_float_vectors(r_count, dim)

    op_l = {"float_vectors": vectors_l}
    op_r = {"ids": db_ids, "collection": collection_name, "field": vec_field}

    metric = "L2"
    sqrt = True
    params = {"metric": metric, "sqrt": sqrt}
    time_start = time.time()
    results = milvus.calc_distance(vectors_left=op_l, vectors_right=op_r, params=params)
    time_end = time.time()
    print(metric, "distance(", "l_count", l_count, "r_count", r_count, "dim", dim, ") time cost",
          (time_end - time_start) * 1000, "ms")
    print(results)
    if len(results) != l_count * r_count:
        print("Incorrect results returned by calc_distance()")

    all_correct = True
    for i in range(l_count):
        vec_l = vectors_l[i]
        for j in range(r_count):
            vec_r = db_vectors[j]
            dist_1 = l2_distance(vec_l, vec_r, sqrt) if metric == "L2" else ip_distance(vec_l, vec_r)
            dist_2 = results[i * r_count + j]
            if math.fabs(dist_1 - dist_2) > _PRECISION:
                print("Incorrect result", dist_1, "vs", dist_2)
                all_correct = False

    if all_correct:
        print("All distance are correct\n")

if __name__ == '__main__':
    input_vectors_to_calc()
    # insert_vectors_to_calc()
