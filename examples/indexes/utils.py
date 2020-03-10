import random


def gen_collection(connection, collection_name, dimension, metric_type, index_file_size=1024):
    collection_param = {
        "collection_name": collection_name,
        "dimension": dimension,
        "metric_type": metric_type,
        "index_file_size": index_file_size
    }

    return connection.create_collection(collection_param)


def insert_data(connection, collection_name, dim, num, partition_tag=None):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]

    status, ids = connection.insert(collection_name, vectors, partition_tag=partition_tag)
    return status, ids, vectors
