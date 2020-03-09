import random

from milvus import Milvus, IndexType, MetricType

from utils import gen_collection, insert_data

if __name__ == '__main__':
    collection_name = "milvus_demo_hnsw"

    client = Milvus()
    client.connect()

    # create collection: dimension is 128, metric type is L2
    gen_collection(client, collection_name, 128, MetricType.L2)

    # insert 10000 vectors into collection
    insert_data(client, collection_name, 128, 100000)

    # flush data into disk for persistent storage
    client.flush([collection_name])

    # specify index param
    index_param = {
        "search_length": 45,
        "out_degree": 50,
        "candidate_pool_size": 300,
        "knng": 100
    }

    # create `RNSG` index
    status = client.create_index(collection_name, IndexType.RNSG, index_param)
    if status.OK():
        print("Create index RNSG successfully\n")
    else:
        print("Create index RNSG fail: ", status)

    # randomly generate 10 query vectors
    query_vectors = [[random.random() for _ in range(128)] for _ in range(10)]

    # specify search param
    search_param = {
        "search_length": 60
    }

    # specify topk is 1, search approximate nearest 1 neighbor
    status, result = client.search(collection_name, 1, query_vectors, params=search_param)

    if status.OK():
        # show search result
        print("Search successfully. Result:\n", result)
    else:
        print("Search fail")

    # drop collection
    client.drop_collection(collection_name)

    # disconnect from server
    client.disconnect()
