# This python shell demonstrate how to create IVF_PQ index
# and search
# Note that index IVF_PQ is not supported on server of GPU version,
# please make sure milvus server you used is on CPU version.

import random

from milvus import Milvus, IndexType, MetricType

from utils import gen_collection, insert_data

if __name__ == '__main__':
    collection_name = "milvus_demo_ivf_pq"

    client = Milvus()

    # create collection: dimension is 128, metric type is L2
    gen_collection(client, collection_name, 128, MetricType.L2)

    # insert 10000 vectors into collection
    _, _, vectors = insert_data(client, collection_name, 128, 10000)

    # flush data into disk for persistent storage
    client.flush([collection_name])

    # specify index param
    index_param = {
        "m": 16,
        "nlist": 1024
    }

    # create `IVF_PQ` index
    status = client.create_index(collection_name, IndexType.IVF_PQ, index_param)
    if status.OK():
        print("Create index IVF_PQ successfully\n")
    else:
        print("Create index fail: ", status)

    # select top 10 vectors from inserted as query vectors
    query_vectors = vectors[:10]

    # specify search param
    search_param = {
        "nprobe": 10
    }

    # specify topk is 1, search approximate nearest 1 neighbor
    status, result = client.search(collection_name, 2, query_vectors, params=search_param)

    if status.OK():
        # show search result
        print("Search successfully. Result:\n", result)
    else:
        print("Search fail")

    # drop collection
    client.drop_collection(collection_name)

    client.close()
