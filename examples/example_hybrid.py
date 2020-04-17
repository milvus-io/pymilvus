# This program demos how to connect to Milvus vector database, 
# create a vector collection,
# insert 10 vectors, 
# and execute a vector similarity search.
import datetime
import sys

sys.path.append(".")
import random
import threading
import time
from milvus import Milvus, IndexType, MetricType, Status
from milvus import DataType

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 128  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    milvus = Milvus(_HOST, _PORT)

    # Check if server is accessible
    if not milvus.ping():
        print("Server is unreachable")
        sys.exit(0)

    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_hybrid_collection_'

    collection_field = {
        "A": {"data_type": DataType.INT8},
        "Vec": {"dimension": 128}
    }
    status = milvus.create_hybrid_collection(collection_name, collection_field, None)
    print(status)
    sys.exit(0)

    status, ok = milvus.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.L2  # optional
        }

        milvus.create_collection(param)

    # Show collections in Milvus server
    _, collections = milvus.show_collections()

    # present collection info
    _, info = milvus.collection_info(collection_name)
    print(info)

    # Describe demo_collection
    _, collection = milvus.describe_collection(collection_name)
    print(collection)

    # 10000 vectors with 16 dimension
    # element per dimension is float32 type
    # vectors should be a 2-D array
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10000)]
    # You can also use numpy to generate random vectors:
    #     `vectors = np.random.rand(10000, 16).astype(np.float32)`

    # Insert vectors into demo_collection, return status and vectors id list
    status, ids = milvus.insert(collection_name=collection_name, records=vectors)

    # Flush collection  inserted data to disk.
    milvus.flush([collection_name])

    # Get demo_collection row count
    status, result = milvus.count_collection(collection_name)

    # create index of vectors, search more rapidly
    index_param = {
        'nlist': 2048
    }

    # Create ivflat index in demo_collection
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    print("Creating index: {}".format(index_param))
    status = milvus.create_index(collection_name, IndexType.IVF_FLAT, index_param)

    # describe index, get information of index
    status, index = milvus.describe_index(collection_name)
    print(index)

    # Use the top 10 vectors for similarity search
    query_vectors = vectors[0:10]

    # execute vector similarity search
    search_param = {
        "nprobe": 16
    }

    print("Searching ... ")

    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,
        'top_k': 1,
        'params': search_param,
    }

    status, results = milvus.search(**param)
    if status.OK():
        # indicate search result
        # also use by:
        #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
        if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
            print('Query result is correct')
        else:
            print('Query result isn\'t correct')

        # print results
        print(results)
    else:
        print("Search failed. ", status)

    # Delete demo_collection
    status = milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
