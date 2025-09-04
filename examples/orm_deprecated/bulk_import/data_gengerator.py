import random
import tensorflow as tf
import numpy as np


# optional input for binary vector:
# 1. list of int such as [1, 0, 1, 1, 0, 0, 1, 0]
# 2. numpy array of uint8
def gen_binary_vector(to_numpy_arr: bool, dim: int):
    raw_vector = [random.randint(0, 1) for i in range(dim)]
    if to_numpy_arr:
        return np.packbits(raw_vector, axis=-1)
    return raw_vector


# optional input for float vector:
# 1. list of float such as [0.56, 1.859, 6.55, 9.45]
# 2. numpy array of float32
def gen_float_vector(to_numpy_arr: bool, dim: int):
    raw_vector = [random.random() for _ in range(dim)]
    if to_numpy_arr:
        return np.array(raw_vector, dtype="float32")
    return raw_vector


# optional input for bfloat16 vector:
# 1. list of float such as [0.56, 1.859, 6.55, 9.45]
# 2. numpy array of bfloat16
def gen_bf16_vector(to_numpy_arr: bool, dim: int):
    raw_vector = [random.random() for _ in range(dim)]
    if to_numpy_arr:
        return tf.cast(raw_vector, dtype=tf.bfloat16).numpy()
    return raw_vector


# optional input for float16 vector:
# 1. list of float such as [0.56, 1.859, 6.55, 9.45]
# 2. numpy array of float16
def gen_fp16_vector(to_numpy_arr: bool, dim: int):
    raw_vector = [random.random() for _ in range(dim)]
    if to_numpy_arr:
        return np.array(raw_vector, dtype=np.float16)
    return raw_vector


# optional input for sparse vector:
# only accepts dict like {2: 13.23, 45: 0.54} or {"indices": [1, 2], "values": [0.1, 0.2]}
# note: no need to sort the keys
def gen_sparse_vector(pair_dict: bool):
    raw_vector = {}
    dim = random.randint(2, 20)
    if pair_dict:
        raw_vector["indices"] = [i for i in range(dim)]
        raw_vector["values"] = [random.random() for _ in range(dim)]
    else:
        for i in range(dim):
            raw_vector[i] = random.random()
    return raw_vector


# optional input for int8 vector:
# 1. list of int8 such as [-6, 18, 65, -94]
# 2. numpy array of int8
def gen_int8_vector(to_numpy_arr: bool, dim: int):
    raw_vector = [random.randint(-128, 127) for _ in range(dim)]
    if to_numpy_arr:
        return np.array(raw_vector, dtype=np.int8)
    return raw_vector
