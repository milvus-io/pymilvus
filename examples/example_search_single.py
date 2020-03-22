import datetime
import random
import time

_DIM = 128

from milvus import Milvus, MetricType, IndexType

if __name__ == '__main__':

    lam = lambda x: "connection_pool_demo_g100_" + str(x)

    collection_list = list(map(lam, range(10)))

    vectors = [[random.random() for _ in range(_DIM)] for _ in range(100)]

    t0 = time.time()

    for index, name in enumerate(collection_list):
        client = Milvus()
        client.connect()
        status, _ = client.search(name, 1, vectors, params={"nprobe": 10})
        if not status.OK():
            print("[{}] search failed: ".format(index), status)
            break

        print("[{}] search done.".format(index))
        client.disconnect()

    # client.drop_collection(collection_name)

    t1 = time.time()

    print("===========\nSearch done. Total cost {} s".format(t1 - t0))
