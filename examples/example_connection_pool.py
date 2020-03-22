import datetime
import random
import threading
import time

from milvus import Milvus, MetricType, IndexType

from milvus.client.pool import ConnectionPool

_DIM = 128
_COLLECTION_COUNT = 100

def run(t_id, collection_name, vectors, connection_pool):
    collection_param = {
        "collection_name": collection_name,
        "dimension": _DIM,
        "metric_type": MetricType.L2,
        "index_file_size": 100
    }

    scoped_conn = connection_pool.fetch()
    # client = scoped_conn.connection().connection()
    client = scoped_conn.client()
    print("[{}] [{}] | client <{}><{}>: ".format(t_id, threading.currentThread().ident, type(client), id(client)))

    status = client.create_collection(collection_param)
    if not status.OK():
        print("[{}] create collection failed: ".format(t_id), status)
        return

    print("[{}] [{}] | [{}] Start insert vector".format(datetime.datetime.now(), threading.currentThread().ident, t_id))
    status, _ = client.insert(collection_name, vectors)
    if not status.OK():
        print("[{}] insert failed: ".format(t_id), status)
        return

    print("[{}] [{}] | [{}] insert done.".format(datetime.datetime.now(), threading.currentThread().ident, t_id))
    # status = client.drop_collection(collection_name)
    # if not status.OK():
    #     print("[{}] [{}] | [{}] drop table fail.".format(datetime.datetime.now(), threading.currentThread().ident, t_id))


if __name__ == '__main__':
    pool = ConnectionPool(uri="tcp://127.0.0.1:19530")

    thread_list = []

    collection_name_prefix = "connection_pool_demo_g100_"

    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10000)]
    # vectors_list = []
    # for _ in range(10):
    #     vectors = [[random.random() for _ in range(_DIM)] for _ in range(100000)]
    #     vectors_list.append(vectors)

    t0 = time.time()
    print("[{}] Insert start\n===========\n\n".format(datetime.datetime.now()))
    for i in range(_COLLECTION_COUNT):
        collection_name = collection_name_prefix + str(i)
        # thread = threading.Thread(target=run, args=(i, collection_name, vectors_list[i], pool))
        thread = threading.Thread(target=run, args=(i, collection_name, vectors, pool))
        thread.start()
        thread_list.append(thread)

    for t in thread_list:
        t.join()

    t1 = time.time()

    print("\n=========\n[{}] Insert done. Total cost {} s".format(datetime.datetime.now(), t1 - t0))
