import random

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    db,
)
from pymilvus.orm import utility

_HOST = '127.0.0.1'
_PORT = '19530'
_ROOT = "root"
_ROOT_PASSWORD = "Milvus"
_METRIC_TYPE = 'IP'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3

# Vector parameters
_DIM = 128
_INDEX_FILE_SIZE = 32  # max file size of stored index


def connect_to_milvus(db_name="default"):
    print(f"connect to milvus\n")
    connections.connect(host=_HOST,
                        port=_PORT,
                        user=_ROOT,
                        password=_ROOT_PASSWORD,
                        db_name=db_name,
                        )


def connect_to_milvus_with_uri(db_name="default"):
    print(f"connect to milvus\n")
    connections.connect(
        alias="uri-connection",
        uri="http://{}:{}/{}".format(_HOST, _PORT, db_name),
    )


def create_collection(collection_name, db_name):
    default_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name="fv", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    default_schema = CollectionSchema(fields=default_fields)
    print(f"Create collection:{collection_name} within db:{db_name}")
    return Collection(name=collection_name, schema=default_schema)


def insert(collection, num, dim):
    data = [
        [i for i in range(num)],
        [float(i) for i in range(num)],
        [[random.random() for _ in range(dim)] for _ in range(num)],
    ]
    collection.insert(data)
    return data[2]


def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")


def search(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id >= 0"}
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))


def collection_read_write(collection, db_name):
    col_name = "{}:{}".format(db_name, collection.name)
    vectors = insert(collection, 10000, _DIM)
    collection.flush()
    print("\nInsert {} rows data into collection:{}".format(collection.num_entities, col_name))

    # create index
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index("fv", index_param)
    print("\nCreated index:{} for collection:{}".format(collection.index().params, col_name))

    # load data to memory
    print("\nLoad collection:{}".format(col_name))
    collection.load()
    # search
    print("\nSearch collection:{}".format(col_name))
    search(collection, "fv", "id", vectors[:3])

    # release memory
    collection.release()
    # drop collection index
    collection.drop_index()
    print("\nDrop collection:{}".format(col_name))


if __name__ == '__main__':
    # connect to milvus and using database db1
    # there will not check db1 already exists during connect
    connect_to_milvus(db_name="default")

    # create collection within default
    col1_db1 = create_collection("col1_db1", "default")

    db1Name = "db1"
    # create db1
    if db1Name not in db.list_database():
        print("\ncreate database: db1")
        db.create_database(db_name=db1Name, properties={"key1":"value1"})
        db_info = db.describe_database(db_name=db1Name)
        print(db_info)

    # use database db1
    db.using_database(db_name=db1Name)
    # create collection within default
    col2_db1 = create_collection("col1_db1", db1Name)

    # verify read and write
    collection_read_write(col2_db1, db1Name)

    # list collections within db1
    print("\nlist collections of database db1:")
    print(utility.list_collections())
    
    # set properties of db1
    db_info = db.describe_database(db_name=db1Name)
    print(db_info)
    print("\nset properties of db1:")
    db.set_properties(db_name=db1Name, properties={"key": "value"})
    db_info = db.describe_database(db_name=db1Name)
    print(db_info)

    print("\ndrop collection: col1_db2 from db1")
    col2_db1.drop()
    print("\ndrop database: db1")
    db.drop_database(db_name=db1Name)

    # list database
    print("\nlist databases:")
    print(db.list_database())
