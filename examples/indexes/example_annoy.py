from milvus import Milvus, IndexType, MetricType

from utils import gen_collection, insert_data

if __name__ == '__main__':
    collection_name = "milvus_demo_annoy"

    # use default host:127.0.0.1, port:19530
    client = Milvus()

    # create collection: dimension is 128, metric type is L2
    gen_collection(client, collection_name, 128, MetricType.L2)

    # insert 10000 vectors into collection
    _, _, vectors = insert_data(client, collection_name, 128, 100000)

    # flush data into disk for persistent storage
    client.flush([collection_name])

    # specify index param
    index_param = {
        "n_trees": 50,
    }

    # create `ANNOY` index
    print("Start create index ......")
    status = client.create_index(collection_name, IndexType.ANNOY, index_param)
    if status.OK():
        print("Create index ANNOY successfully\n")
    else:
        print("Create index ANNOY fail")

    # select top 10 vectors from inserted as query vectors
    query_vectors = vectors[:10]

    # specify search param
    search_param = {
        "search_k": 10
    }

    # specify topk is 1, search approximate nearest 1 neighbor
    status, result = client.search(collection_name, 1, query_vectors, params=search_param)

    if status.OK():
        # show search result
        print("Search successfully. Result:\n", result)
    else:
        print("Search fail: ", status)

    # drop collection
    client.drop_collection(collection_name)

    client.close()
