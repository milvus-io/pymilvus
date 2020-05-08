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
    status, info = client.collection_info(collection_name)
    if not status.OK():
        print("Query collection statistical information fail. exiting ....")
        sys.exit(1)

    # show collection information
    print("Total amount of vectors in collection {} is {}".format(collection_name, info["row_count"]))
    for par in info["partitions"]:
        print("\tpartition tag: {}, vector count: {}".format(par.tag, par.count))
        # show segment information
        for seg in par.segments_stat:
            print("\t\tsegment name: {}, vector count: {}, index: {}, storage size {:.3f} MB"
                  .format(seg.segment_name, seg.count, seg.index_name, seg.data_size / 1024 / 1024))

    # obtain vector ids from segment, then
    # get vector by specifying vector id
    segment0 = info.partitions_stat[0].segments_stat[0]
    status, ids = client.get_vector_ids(collection_name, segment0.segment_name)
    if not status.OK():
        print("Cannot obtain vector ids from segment {}. exiting ....".format(segment0.segment_name))
        sys.exit(1)

    # obtain first vector
    status, vector = client.get_vector_by_id(collection_name, ids[0])
    if not status.OK():
        print("Cannot obtain vector. exiting ....")
        sys.exit(1)

    # delete top 10 vectors
    status = client.delete_by_id(collection_name, ids[:10])
    if status.OK():
        print("Delete top 10 vectors successfully")
    else:
        print("Error occurred when try to delete top 10 vectors. Reason: ", status.message)

    client.drop_collection(collection_name)

