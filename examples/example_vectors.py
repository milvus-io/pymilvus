import random
import sys

from milvus import Milvus, MetricType, IndexType

_DIM = 128

if __name__ == '__main__':
    client = Milvus(host="localhost", port=19121, handler="HTTP")

    collection_name = 'example_collection_vector'

    param = {
        'collection_name': collection_name,
        'dimension': _DIM,
        'index_file_size': 10,  # optional
        'metric_type': MetricType.L2  # optional
    }

    client.create_collection(param)

    # randomly generate 100000 vectors and insert collection
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10000)]
    client.insert(collection_name, vectors)

    # flush data to disk
    client.flush([collection_name])

    # query collection's statistical information.
    status, info = client.get_collection_stats(collection_name)
    if not status.OK():
        print("Query collection statistical information fail. exiting ....")
        sys.exit(1)

    # show collection information
    print("Total amount of vectors in collection {} is {}".format(collection_name, info["row_count"]))
    for par in info["partitions"]:
        print("\tpartition tag: {}, vector count: {}".format(par["tag"], par["row_count"]))
        # show segment information
        for seg in par["segments"]:
            print("\t\tsegment name: {}, vector count: {}, index: {}, storage size {:.3f} MB"
                  .format(seg["name"], seg["row_count"], seg["index_name"], seg["data_size"] / 1024 / 1024))

    # obtain vector ids from segment, then
    # get vector by specifying vector id
    segment0 = info["partitions"][0]["segments"][0]
    status, ids = client.list_id_in_segment(collection_name, segment0["name"])
    if not status.OK():
        print("Cannot obtain vector ids from segment {}. exiting ....".format(segment0["name"]))
        sys.exit(1)

    # obtain top 5 vector
    status, vectors = client.get_entity_by_id(collection_name, ids[:5])
    if not status.OK():
        print("Cannot obtain vector. exiting ....")
        sys.exit(1)

    # delete top 10 vectors
    status = client.delete_entity_by_id(collection_name, ids[:10])
    if status.OK():
        print("Delete top 10 vectors successfully")
    else:
        print("Error occurred when try to delete top 10 vectors. Reason: ", status.message)

    client.drop_collection(collection_name)

