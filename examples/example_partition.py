# This program demos how to connect to Milvus vector database, 
# create a vector table, 
# insert 10 vectors, 
# and execute a vector similarity search.
import random
from milvus import Milvus, IndexType, MetricType
import time

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 16  # dimension of vector
_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    # Connect to Milvus server
    # You may need to change _HOST and _PORT accordingly
    param = {'host': _HOST, 'port': _PORT}

    # You can create a instance specified server addr and
    # invoke rpc method directly
    client = Milvus(**param)
    # Create collection demo_collection if it dosen't exist.
    collection_name = 'demo_partition_collection'
    partition_tag = "random"

    # create collection
    param = {
        'collection_name': collection_name,
        'dimension': _DIM,
        'index_file_size': _INDEX_FILE_SIZE,  # optional
        'metric_type': MetricType.L2  # optional
    }

    client.create_collection(param)

    # Show collections in Milvus server
    _, collections = client.list_collections()

    # Describe collection
    _, collection = client.get_collection_info(collection_name)
    print(collection)

    # create partition
    client.create_partition(collection_name, partition_tag=partition_tag)
    # display partitions
    _, partitions = client.list_partitions(collection_name)

    # 10000 vectors with 16 dimension
    # element per dimension is float32 type
    # vectors should be a 2-D array
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10000)]
    # You can also use numpy to generate random vectors:
    #     `vectors = np.random.rand(10000, 16).astype(np.float32).tolist()`

    # Insert vectors into partition of collection, return status and vectors id list
    status, ids = client.insert(collection_name=collection_name, records=vectors, partition_tag=partition_tag)

    # Wait for 6 seconds, until Milvus server persist vector data.
    time.sleep(6)

    # Get demo_collection row count
    status, num = client.count_entities(collection_name)

    # create index of vectors, search more rapidly
    index_param = {
        'nlist': 2048
    }

    # Create ivflat index in demo_collection
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    status = client.create_index(collection_name, IndexType.IVF_FLAT, index_param)

    # describe index, get information of index
    status, index = client.get_index_info(collection_name)
    print(index)

    # Use the top 10 vectors for similarity search
    query_vectors = vectors[0:10]

    # execute vector similarity search, search range in partition `partition1`
    search_param = {
        "nprobe": 10
    }

    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,
        'top_k': 1,
        'partition_tags': ["random"],
        'params': search_param
    }
    status, results = client.search(**param)

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

    # Drop partition. You can also invoke `drop_collection()`, so that all of partitions belongs to
    # designated collections will be deleted.
    status = client.drop_partition(collection_name, partition_tag)

    # Delete collection. All of partitions of this collection will be dropped.
    status = client.drop_collection(collection_name)


if __name__ == '__main__':
    main()
