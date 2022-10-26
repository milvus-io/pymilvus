import random

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search


_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Vector parameters
_DIM = 128
_INDEX_FILE_SIZE = 32  # max file size of stored index

# Scalar
_ATTR1_NAME = "attr1"
_ATTR2_NAME = "attr2"

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3


# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# Create a collection named 'demo'
def create_collection(name, id_field, vector_field, attr1_name, attr2_name):
    field1 = FieldSchema(name=id_field, dtype=DataType.VARCHAR, description="varchar", is_primary=True, max_length=10)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM)
    field3 = FieldSchema(name=attr1_name, dtype=DataType.INT64, description="attr1")
    field4 = FieldSchema(name=attr2_name, dtype=DataType.DOUBLE, description="attr2")
    schema = CollectionSchema(fields=[field1, field2, field3, field4])
    collection = Collection(name=name, data=None, schema=schema)
    print("\ncollection created:", name)
    return collection


def has_collection(name):
    return utility.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    utility.drop_collection(name)
    print("\nDrop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())


def insert(collection, num, dim):
    data = [
        [str(i) for i in range(num)],
        [[random.random() for _ in range(dim)] for _ in range(num)],
        [random.randint(1, 10000) for _ in range(num)],
        [random.random() for _ in range(num)],
    ]
    collection.insert(data)
    return data[1]


def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)


def create_index(collection, filed_name, index_param, index_name):
    collection.create_index(filed_name, index_param, index_name=index_name)
    index = collection.index(index_name=index_name)
    print(f"\nCreated index:\n{index.params}")


def drop_index(collection, index_name):
    collection.drop_index(index_name=index_name)
    print("\nDrop index sucessfully")


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()


def search(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": 'id_field > "0"'}

    results = collection.search(**search_param)

    for i, result in enumerate(results):
        print(f"\nSearch result for {i}th vector: ")
        for j, res in enumerate(result):
            print(f"Top {j}: {res}")


def main():
    # create a connection
    create_connection()

    # drop collection if the collection exists
    if has_collection(_COLLECTION_NAME):
        drop_collection(_COLLECTION_NAME)

    # create collection
    collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME, _ATTR1_NAME, _ATTR2_NAME)

    # show collections
    list_collections()

    # insert 10000 vectors with 128 dimension
    vectors = insert(collection, 10000, _DIM)

    collection.flush()
    # get the number of entities
    get_entity_num(collection)

    # create index
    vec_index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    scalar_index_param = {"index_type": "inverted_index"}  # TODO: replace this with real impl.

    vector_index_name = "vector_index"
    create_index(collection, _VECTOR_FIELD_NAME, vec_index_param, vector_index_name)
    print(f"has_index {vector_index_name}: ", collection.has_index())
    print("index: ", collection.index().to_dict())
    utility.wait_for_index_building_complete(collection.name)
    print("index building progress: ", utility.index_building_progress(collection.name))

    varchar_index_name = "varchar_id_index"
    create_index(collection, _ID_FIELD_NAME, scalar_index_param, varchar_index_name)
    print(f"has_index {varchar_index_name}: ", collection.has_index(index_name=varchar_index_name))
    print("all indexes:")
    for index in collection.indexes:
        print(index.to_dict())
    utility.wait_for_index_building_complete(collection.name, varchar_index_name)
    print("index building progress: ", utility.index_building_progress(collection.name, varchar_index_name))

    # load data to memory
    load_collection(collection)

    # search
    search(collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, vectors[:3])

    # release memory
    release_collection(collection)

    # drop collection index
    drop_index(collection, vector_index_name)
    drop_index(collection, varchar_index_name)

    # drop collection
    drop_collection(_COLLECTION_NAME)


if __name__ == '__main__':
    main()
