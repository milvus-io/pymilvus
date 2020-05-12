# This program demos how to connect to Milvus vector database,
# create a vector collection,
# insert 10 vectors,
# and execute a vector similarity search.
import datetime
import sys
import threading

sys.path.append(".")
import random
import time
from milvus import Milvus, IndexType, MetricType, Status

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
# _PORT = '19530'  # default value
_PORT = '19121'  # default http value

# Vector parameters
_DIM = 128  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index


def _insert(client, collection, vectors):
    _round = 0
    while True:
        print("== Insert round {} ==".format(_round))
        status, _ = client.insert(collection, vectors)
        if not status.OK():
            print("Insert fail.", status.message)
            break
        _round += 1
        time.sleep(0.5)


def _search(client, collection, query_vectors):
    _round = 0
    while True:
        print("== Search round {} ==".format(_round))
        status, _ = client.search(collection, 10, query_vectors, params={"nprobe": 10})
        if not status.OK():
            print("Search fail.", status.message)
            break
        _round += 1
        time.sleep(0.5)


def main():
    # Specify server addr when create milvus client instance
    milvus = Milvus(_HOST, _PORT, pool_size=10, handler="HTTP")

    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_http_collection_10'

    status, ok = milvus.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.L2  # optional
        }

        milvus.create_collection(param)

    vectors = [[random.random() for _ in range(_DIM)] for _ in range(100000)]
    #
    # insert_thread = threading.Thread(target=_insert, args=(milvus, collection_name, vectors))
    # insert_thread.start()

    # query_vectors = vectors[0:10]
    # search_thread = threading.Thread(target=_search, args=(milvus, collection_name, query_vectors))
    # search_thread.start()

    # insert_thread.join()
    # search_thread.join()
    # sys.exit(0)

    t0 = time.time()
    status, ids = milvus.insert(collection_name=collection_name, records=vectors)
    print("Inseert time: {}".format(time.time() - t0))
    sys.exit(0)

    # Flush collection inserted data to disk.
    milvus.flush([collection_name])

    # create index of vectors, search more rapidly
    index_param = {
        'nlist': 2048
    }

    # Create ivflat index in demo_collection
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    print("Creating index: {}".format(index_param))
    status = milvus.create_index(collection_name, IndexType.IVF_FLAT, index_param)

    # Use the top 10 vectors for similarity search

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
    # status = milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
