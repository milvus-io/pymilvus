#This program demos how to connect to Milvus vector database, 
# create a vector collection,
# insert 10 vectors, 
# and execute a vector similarity search.

import random
import numpy as np

from milvus import Milvus, DataType

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value
# _PORT = '19121'  # default http value

# Vector parameters
_DIM = 8  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    # Specify server addr when create milvus client instance
    # milvus client instance maintain a connection pool, param
    # `pool_size` specify the max connection num.
    milvus = Milvus(_HOST, _PORT)

    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_collection'

    ok = milvus.has_collection(collection_name)
    field_name = 'example_field'
    if not ok:
        fields = {"fields":[{
            "name": field_name,
            "type": DataType.FLOAT_VECTOR,
            "metric_type": "L2",
            "params": {"dim": _DIM},
            "indexes": [{"metric_type": "L2"}]
        }]}

        milvus.create_collection(collection_name=collection_name, fields=fields)
    else:
        milvus.drop_collection(collection_name=collection_name)

    # Show collections in Milvus server
    collections = milvus.list_collections()
    print(collections)

    # Describe demo_collection
    stats = milvus.get_collection_stats(collection_name)
    print(stats)

    # 10000 vectors with 128 dimension
    # element per dimension is float32 type
    # vectors should be a 2-D array
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10)]
    print(vectors)
    # You can also use numpy to generate random vectors:
    #   vectors = np.random.rand(10000, _DIM).astype(np.float32)

    # Insert vectors into demo_collection, return status and vectors id list
    entities = [{"name": field_name, "type": DataType.FLOAT_VECTOR, "values": vectors}]

    res_ids = milvus.insert(collection_name=collection_name, entities=entities)
    print("ids:",res_ids)

    # Flush collection  inserted data to disk.
    milvus.flush([collection_name])

    # present collection statistics info
    stats = milvus.get_collection_stats(collection_name)
    print(stats)

    # create index of vectors, search more rapidly
    index_param = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }

    # Create ivflat index in demo_collection
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    print("Creating index: {}".format(index_param))
    status = milvus.create_index(collection_name, field_name, index_param)

    # execute vector similarity search

    print("Searching ... ")

    dsl = {"bool": {"must": [{"vector": {
                        field_name: {
                            "metric_type": "L2",
                            "query": vectors,
                            "topk": 10,
                            "params": {"nprobe": 16}
                        }
    }}]}}

    milvus.load_collection(collection_name)
    results = milvus.search(collection_name, dsl)
    # indicate search result
    # also use by:
    #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
    if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
        print('Query result is correct')
    else:
        print('Query result isn\'t correct')

    milvus.drop_index(collection_name,field_name)
    milvus.release_collection(collection_name)

    # Delete demo_collection
    status = milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
