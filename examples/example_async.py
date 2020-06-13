# This program demos how to connect to Milvus vector database,
# create a vector collection,
# insert 10 vectors,
# and execute a vector similarity search.
import datetime
import sys
import random
from milvus import Milvus, IndexType, MetricType

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 128  # dimension of vector


def main():
    # Specify server addr when create milvus client instance
    milvus = Milvus(_HOST, _PORT)

    # Create collection demo_collection if it dosen't exist.
    collection_name = 'example_async_collection_'

    status, ok = milvus.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': 128,  # optional
            'metric_type': MetricType.L2  # optional
        }

        status = milvus.create_collection(param)
        if not status.OK():
            print("Create collection failed: {}".format(status.message), file=sys.stderr)
            print("exiting ...", file=sys.stderr)
            sys.exit(1)

    # Show collections in Milvus server
    _, collections = milvus.list_collections()

    # Describe demo_collection
    _, collection = milvus.get_collection_info(collection_name)
    print(collection)

    # 10000 vectors with 16 dimension
    # element per dimension is float32 type
    # vectors should be a 2-D array
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(100000)]

    # You can also use numpy to generate random vectors:
    #     `vectors = np.random.rand(10000, 16).astype(np.float32)`

    def _insert_callback(status, ids):
        if status.OK():
            print("Insert successfully")
        else:
            print("Insert failed.", status.message)

    # Insert vectors into demo_collection, adding callback function
    insert_future = milvus.insert(collection_name=collection_name, records=vectors, _async=True,
                                  _callback=_insert_callback)
    # Or invoke result() to get results:
    #   insert_future = milvus.insert(collection_name=collection_name, records=vectors, _async=True)
    #   status, ids = insert_future.result()
    insert_future.done()

    # Flush collection  inserted data to disk.
    def _flush_callback(status):
        if status.OK():
            print("Flush successfully")
        else:
            print("Flush failed.", status.message)

    flush_future = milvus.flush([collection_name], _async=True, _callback=_flush_callback)
    # Or invoke result() to get results:
    #   flush_future = milvus.flush([collection_name], _async=True)
    #   status = flush_future.result()
    flush_future.done()

    def _compact_callback(status):
        if status.OK():
            print("Compact successfully")
        else:
            print("Compact failed.", status.message)

    compact_furure = milvus.compact(collection_name, _async=True, _cakkback=_compact_callback)
    # Or invoke result() to get results:
    #   compact_future = milvus.compact(collection_name, _async=True)
    #   status = compact_future.result()
    compact_furure.done()

    # Get demo_collection row count
    status, result = milvus.count_entities(collection_name)

    # present collection info
    _, info = milvus.get_collection_stats(collection_name)
    print(info)

    # create index of vectors, search more rapidly
    index_param = {
        'nlist': 2048
    }

    def _index_callback(status):
        if status.OK():
            print("Create index successfully")
        else:
            print("Create index failed.", status.message)

    # Create ivflat index in demo_collection
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    print("Creating index: {}".format(index_param))
    index_future = milvus.create_index(collection_name, IndexType.IVF_FLAT, index_param, _async=True,
                                       _callback=_index_callback)
    # Or invoke result() to get results:
    #   index_future = milvus.create_index(collection_name, IndexType.IVF_FLAT, index_param, _async=True)
    #   status = index_future.result()
    index_future.done()

    # describe index, get information of index
    status, index = milvus.get_index_info(collection_name)
    print(index)

    # Use the top 10 vectors for similarity search
    query_vectors = vectors[0:10]

    # execute vector similarity search
    search_param = {
        "nprobe": 16
    }

    print("Searching ... ")

    def _search_callback(status, results):
        # if status.OK():
        #     print("Search successfully")
        # else:
        #     print("Search failed.", status.message)
        if status.OK():
            # indicate search result
            # also use by:
            #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
            if results[0][0].distance == 0.0:  # or results[0][0].id == ids[0]:
                print('Query result is correct')
            else:
                print('Query result isn\'t correct')

            # print results
            print(results)
        else:
            print("Search failed. ", status)

    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,
        'top_k': 1,
        'params': search_param,
        "_async": True,
        "_callback": _search_callback
    }
    search_future = milvus.search(**param)
    # Or invoke result() to get results:
    #
    #   param = {
    #       'collection_name': collection_name,
    #       'query_records': query_vectors,
    #       'top_k': 1,
    #       'params': search_param,
    #       "_async": True,
    #   }
    #   search_future = milvus.search(param)
    #   status, results = index_future.result()

    search_future.done()

    # Delete demo_collection
    status = milvus.drop_collection(collection_name)


if __name__ == '__main__':
    main()
