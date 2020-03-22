import datetime
import random
import time

_DIM = 128

from milvus import Milvus, MetricType, IndexType

if __name__ == '__main__':

    lam = lambda x: "single_" + str(x)

    collection_list = list(map(lam, range(1)))

    vectors = [[random.random() for _ in range(_DIM)] for _ in range(100000)]

    t0 = time.time()
    for index, collection_name in enumerate(collection_list):
        collection_param = {
            "collection_name": collection_name,
            "dimension": _DIM,
            "metric_type": MetricType.L2,
            "index_file_size": 100
        }

        client = Milvus()
        client.connect()

        status = client.create_collection(collection_param)
        if not status.OK():
            print("[{}] create collection failed: ".format(index), status)
            break

        print("[{}] | [{}] Start insert vector".format(datetime.datetime.now(), index))
        status, _ = client.insert(collection_name, vectors)
        if not status.OK():
            print("[{}] insert failed: ".format(index), status)
            break

        print("[{}] insert done.".format(index))

        client.drop_collection(collection_name)

    t1 = time.time()

    print("===========\nInsert done. Total cost {} s".format(t1 - t0))
