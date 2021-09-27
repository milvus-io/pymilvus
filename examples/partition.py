# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

from pymilvus import (
    connections, list_collections, has_partition,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition
)

import random
from sklearn import preprocessing
import string

default_dim = 128
default_nb = 3000
default_float_vec_field_name = "float_vector"
default_segment_row_limit = 1000


all_index_types = [
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    # "IVF_SQ8_HYBRID",
    "IVF_PQ",
    "HNSW",
    # "NSG",
    "ANNOY",
    "RHNSW_FLAT",
    "RHNSW_PQ",
    "RHNSW_SQ",
    "BIN_FLAT",
    "BIN_IVF_FLAT"
]

default_index_params = [
    {"nlist": 128},
    {"nlist": 128},
    {"nlist": 128},
    # {"nlist": 128},
    {"nlist": 128, "m": 16, "nbits": 8},
    {"M": 48, "efConstruction": 500},
    # {"search_length": 50, "out_degree": 40, "candidate_pool_size": 100, "knng": 50},
    {"n_trees": 50},
    {"M": 48, "efConstruction": 500},
    {"M": 48, "efConstruction": 500, "PQM": 64},
    {"M": 48, "efConstruction": 500},
    {"nlist": 128},
    {"nlist": 128}
]


default_index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}


def gen_default_fields(auto_id=True):
    default_fields = [
        FieldSchema(name="count", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="float", dtype=DataType.FLOAT),
        FieldSchema(name=default_float_vec_field_name, dtype=DataType.FLOAT_VECTOR, dim=default_dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="test collection",
                                      segment_row_limit=default_segment_row_limit, auto_id=False)
    return default_schema


def gen_vectors(num, dim, is_normal=True):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


def gen_data(nb, is_normal=False):
    vectors = gen_vectors(nb, default_dim, is_normal)
    entities = [
        [i for i in range(nb)],
        [float(i) for i in range(nb)],
        vectors
    ]
    return entities


def gen_unique_str(str_value=None):
    prefix = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
    return "collection_" + prefix if str_value is None else str_value + "_" + prefix


def binary_support():
    return ["BIN_FLAT", "BIN_IVF_FLAT"]


def gen_simple_index():
    index_params = []
    for i in range(len(all_index_types)):
        if all_index_types[i] in binary_support():
            continue
        dic = {"index_type": all_index_types[i], "metric_type": "L2"}
        dic.update({"params": default_index_params[i]})
        index_params.append(dic)
    return index_params


def test_partition():
    connections.connect(alias="default")
    print("create collection")
    collection = Collection(name=gen_unique_str(), schema=gen_default_fields())
    print("create partition")
    partition = Partition(collection, name=gen_unique_str())
    print(list_collections())
    assert has_partition(collection.name, partition.name) is True

    data = gen_data(default_nb)
    print("insert data to partition")
    partition.insert(data)
    assert partition.is_empty is False
    assert partition.num_entities == default_nb

    print("load partition")
    partition.load()
    topK = 5
    round_decimal = 3
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    print("search partition")
    res = partition.search(data[2][-2:], "float_vector", search_params, topK, "count > 100", round_decimal=round_decimal)
    for hits in res:
        for hit in hits:
            print(hit)

    print("release partition")
    partition.release()
    print("drop partition")
    partition.drop()
    print("drop collection")
    collection.drop()


if __name__ == "__main__":
    test_partition()
