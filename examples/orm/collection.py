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

from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from pymilvus.client.types import LoadState

import random
import numpy as np
import pandas

import string

from pymilvus.orm import db

default_dim = 128
default_nb = 3000
default_float_vec_field_name = "float_vector"
default_binary_vec_field_name = "binary_vector"

all_index_types = [
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    # "IVF_SQ8_HYBRID",
    "IVF_PQ",
    "HNSW",
    # "NSG",
    # "ANNOY",
    # "RHNSW_FLAT",
    # "RHNSW_PQ",
    # "RHNSW_SQ",
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
default_binary_index = {"index_type": "BIN_FLAT", "params": {"nlist": 1024}, "metric_type": "JACCARD"}


def gen_default_fields(description="test collection"):
    default_fields = [
        FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name=default_float_vec_field_name, dtype=DataType.FLOAT_VECTOR, dim=default_dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description=description)
    return default_schema


def gen_default_fields_with_primary_key_1():
    default_fields = [
        FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name=default_float_vec_field_name, dtype=DataType.FLOAT_VECTOR, dim=default_dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="test collection")
    return default_schema


def gen_default_fields_with_primary_key_2():
    default_fields = [
        FieldSchema(name="int64", dtype=DataType.INT64),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name=default_float_vec_field_name, dtype=DataType.FLOAT_VECTOR, dim=default_dim)
    ]
    default_schema = CollectionSchema(fields=default_fields, description="test collection", primary_field="int64")
    return default_schema


def gen_binary_schema():
    binary_fields = [
        FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name=default_binary_vec_field_name, dtype=DataType.BINARY_VECTOR, dim=default_dim)
    ]
    default_schema = CollectionSchema(fields=binary_fields, description="test collection")
    return default_schema


def gen_float_vectors(num, dim):
    return [[random.random() for _ in range(dim)] for _ in range(num)]


def gen_float_data(nb):
    entities = [
        [i for i in range(nb)],
        [float(i) for i in range(nb)],
        gen_float_vectors(nb, default_dim),
    ]
    return entities


def gen_dataframe(nb):
    vectors = gen_float_vectors(nb, default_dim)
    data = {
        "int64": [i for i in range(nb)],
        "float": np.array([i for i in range(nb)], dtype=np.float32),
        "float_vector": vectors
    }

    return pandas.DataFrame(data)


def gen_binary_vectors(num, dim):
    raw_vectors = []
    binary_vectors = []
    for i in range(num):
        raw_vector = [random.randint(0, 1) for i in range(dim)]
        raw_vectors.append(raw_vector)
        binary_vectors.append(bytes(np.packbits(raw_vector, axis=-1).tolist()))
    return raw_vectors, binary_vectors


def gen_binary_data(nb):
    raw_vectors, binary_vectors = gen_binary_vectors(nb, dim=default_dim)
    entities = [
        [i for i in range(nb)],
        [float(i) for i in range(nb)],
        binary_vectors
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


connections.connect(alias="default")


def test_create_collection():
    name = gen_unique_str()
    collection = Collection(name=name, schema=gen_default_fields())
    assert collection.is_empty is True
    assert collection.num_entities == 0
    assert utility.load_state(name) == LoadState.NotLoad
    return name


def test_exist_collection(name):
    assert utility.has_collection(name) is True
    collection = Collection(name)
    collection.drop()


def test_collection_only_name():
    name = gen_unique_str()
    Collection(name=name, schema=gen_default_fields())
    collection = Collection(name=name)
    data = gen_float_data(default_nb)
    collection.insert(data)
    collection.flush()
    collection.create_index(field_name=default_float_vec_field_name, index_params=default_index)
    collection.load()
    assert collection.is_empty is False
    assert collection.num_entities == default_nb
    assert utility.load_state(name) == LoadState.Loaded
    collection.drop()


def test_collection_with_dataframe():
    data = gen_dataframe(default_nb)
    collection, _ = Collection.construct_from_dataframe(name=gen_unique_str(), dataframe=data, primary_field="int64")
    collection.flush()
    collection.create_index(field_name=default_float_vec_field_name, index_params=default_index)
    collection.load()
    assert collection.is_empty is False
    assert collection.num_entities == default_nb
    collection.drop()


def test_create_index_float_vector():
    data = gen_float_data(default_nb)
    collection = Collection(name=gen_unique_str(), data=data, schema=gen_default_fields())
    for index_param in gen_simple_index():
        collection.create_index(field_name=default_float_vec_field_name, index_params=index_param)
        assert len(collection.indexes) != 0
        collection.drop_index()
    collection.drop()


def test_create_index_binary_vector():
    collection = Collection(name=gen_unique_str(), schema=gen_binary_schema())
    data = gen_binary_data(default_nb)
    collection.insert(data)
    collection.create_index(field_name=default_binary_vec_field_name, index_params=default_binary_index)
    assert len(collection.indexes) != 0
    collection.drop()


def test_specify_primary_key():
    data = gen_float_data(default_nb)
    collection = Collection(name=gen_unique_str(), data=data, schema=gen_default_fields_with_primary_key_1())
    for index_param in gen_simple_index():
        collection.create_index(field_name=default_float_vec_field_name, index_params=index_param)
        assert len(collection.indexes) != 0
        collection.drop_index()
    collection.drop()

    collection2 = Collection(name=gen_unique_str(), data=data, schema=gen_default_fields_with_primary_key_2())
    for index_param in gen_simple_index():
        collection2.create_index(field_name=default_float_vec_field_name, index_params=index_param)
        assert len(collection2.indexes) != 0
        collection2.drop_index()
    collection2.drop()


def test_alias():
    """ Test alias follows the following steps
    1. Prepare tests
    2. Create collection_A and create an alias `latest_collection`
    3. Create collection_B and shift alias `latest_collection` to collection_B
    4. Drop alias `latest_collection`
    5. Clear up tests

    """

    name_A, name_B = "collection_A", "collection_B"

    def setup() -> (Collection, Collection):
        collection_A = Collection(name=name_A, schema=gen_default_fields("collection A"))
        collection_B = Collection(name=name_B, schema=gen_default_fields("collection B"))
        return collection_A, collection_B

    def teardown():
        if utility.has_collection(name_A):
            utility.drop_collection(name_A)
        if utility.has_collection(name_B):
            utility.drop_collection(name_B)

    def alias_cases():
        teardown()
        A, B = setup()

        latest_coll_alias = "latest_collection"

        utility.create_alias(A.name, latest_coll_alias)

        alias_collection = Collection(latest_coll_alias)
        assert alias_collection.description == A.description

        utility.alter_alias(B.name, latest_coll_alias)

        alias_collection = Collection(latest_coll_alias)
        assert alias_collection.description == B.description

        utility.drop_alias(latest_coll_alias)
        try:
            alias_collection = Collection(latest_coll_alias)
        except BaseException as e:
            print(f" - Alias [{latest_coll_alias}] dropped, cannot get collection from it. Error msg: {e}")
        finally:
            teardown()

    alias_cases()


def test_rename_collection():
    connections.connect(alias="default")
    schema = CollectionSchema(fields=[
        FieldSchema("int64", DataType.INT64, description="int64", is_primary=True),
        FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=128),
    ])

    old_collection = gen_unique_str()
    new_collection = gen_unique_str()
    Collection(old_collection, schema=schema)

    print("\nlist collections:")
    print("rename collection name start, db:default, collections:", utility.list_collections())
    assert utility.has_collection(old_collection)

    utility.rename_collection(old_collection, new_collection)
    assert utility.has_collection(new_collection)
    assert not utility.has_collection(old_collection)
    print("rename collection name end, db:default, collections:", utility.list_collections())

    # create db1
    new_db = "new_db"
    if new_db not in db.list_database():
        print("\ncreate database: new_db")
        db.create_database(db_name=new_db)
    utility.rename_collection(new_collection, new_collection, new_db)
    print("rename db name end, db:default, collections:", utility.list_collections())

    db.using_database(db_name=new_db)
    assert utility.has_collection(new_collection)
    print("rename db name end, db:", new_db, "collections:", utility.list_collections())

    db.using_database(db_name="default")
    assert not utility.has_collection(new_collection)
    print("db:default, collections:", utility.list_collections())


if __name__ == "__main__":
    print("test collection and get an existing collection")
    name = test_create_collection()
    print("test an existing collection")
    test_exist_collection(name)
    print("test collection only name")
    test_collection_only_name()
    print("test collection with dataframe")
    test_collection_with_dataframe()
    print("test collection index float vector")
    test_create_index_float_vector()
    print("test collection binary vector")
    test_create_index_binary_vector()
    print("test collection specify primary key")
    test_specify_primary_key()
    print("test alias")
    test_alias()
    print("test rename collection")
    test_rename_collection()
    print("test end")
