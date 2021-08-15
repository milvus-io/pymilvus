import random
from pymilvus import Milvus, DataType

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

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 10

# Create milvus client instance
milvus = Milvus(_HOST, _PORT)

def create_collection(name):
    id_field = {
        "name": _ID_FIELD_NAME,
        "type": DataType.INT64,
        "auto_id": True,
        "is_primary": True,
    }
    vector_field = {
        "name": _VECTOR_FIELD_NAME,
        "type": DataType.FLOAT_VECTOR,
        "metric_type": "L2",
        "params": {"dim": _DIM},
        "indexes": [{"metric_type": "L2"}]
    }
    fields = {"fields": [id_field, vector_field]}

    milvus.create_collection(collection_name=name, fields=fields)
    print("collection created:", name)

def drop_collection(name):
    if milvus.has_collection(name):
        milvus.drop_collection(name)
        print("collection dropped:", name)

def list_collections():
    collections = milvus.list_collections()
    print("list collection:")
    print(collections)

def get_collection_stats(name):
    stats = milvus.get_collection_stats(name)
    print("collection stats:")
    print(stats)

def insert(name, num, dim):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    entities = [{"name": _VECTOR_FIELD_NAME, "type": DataType.FLOAT_VECTOR, "values": vectors}]
    ids = milvus.insert(name, entities)
    return ids, vectors

def flush(name):
    milvus.flush([name])

def create_index(name, field_name):
    index_param = {
        "metric_type": _METRIC_TYPE,
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST}
    }
    milvus.create_index(name, field_name, index_param)
    print("Create index: {}".format(index_param))

def drop_index(name, field_name):
    milvus.drop_index(name, field_name)
    print("Drop index:", field_name)

def load_collection(name):
    milvus.load_collection(name)

def release_collection(name):
    milvus.release_collection(name)

def search(name, vector_field, search_vectors, ids):
    nq = len(search_vectors)
    search_params = {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}}
    results = milvus.search_with_expression(name, search_vectors, vector_field, param=search_params, limit=_TOPK)
    for i in range(nq):
        if results[i][0].distance == 0.0 or results[i][0].id == ids[0]:
            print("OK! search results: ", results[i][0].entity)
        else:
            print("FAIL! search results: ", results[i][0].entity)

def main():
    name = _COLLECTION_NAME
    vector_field = _VECTOR_FIELD_NAME

    drop_collection(name)
    create_collection(name)

    # show collections
    list_collections()

    # generate 10000 vectors with 128 dimension
    ids, vectors = insert(name, 10000, _DIM)

    # flush
    flush(name)

    # show row_count
    get_collection_stats(name)

    # create index
    create_index(name, vector_field)

    # load
    load_collection(name)

    # search
    search(name, vector_field, vectors[:10], ids)

    drop_index(name, vector_field)
    release_collection(name)
    drop_collection(name)

if __name__ == '__main__':
    main()
