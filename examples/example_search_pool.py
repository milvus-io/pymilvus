import datetime
import random
import threading
import time

from milvus import Milvus, MetricType, IndexType

from milvus.client.pool import ConnectionPool

_DIM = 128


def search(t_id, collection_name, query_vectors, connection_pool):
    scoped_conn = connection_pool.fetch()
    # client = scoped_conn.connection().connection()
    client = scoped_conn.client()
    print("[{}] [{}] | client <{}><{}>: ".format(t_id, threading.currentThread().ident, type(client), id(client)))

    status, _ = client.search(collection_name, 1, vectors, params={'nprobe': 10})
    if not status.OK():
        print("[{}] search failed: ".format(t_id), status)
        return

    print("[{}] [{}] | [{}] search done.".format(datetime.datetime.now(), threading.currentThread().ident, t_id))
    # client.drop_collection(collection_name)


if __name__ == '__main__':
    pool = ConnectionPool(uri="tcp://127.0.0.1:19530")

    thread_list = []

    collection_name_prefix = "connection_pool_demo_g100_"

    vectors = [[random.random() for _ in range(_DIM)] for _ in range(1)]
    # vectors_list = []
    # for _ in range(10):
    #     vectors = [[random.random() for _ in range(_DIM)] for _ in range(100000)]
    #     vectors_list.append(vectors)

    t0 = time.time()
    print("[{}] Search start\n===========\n\n".format(datetime.datetime.now()))
    for i in range(10):
        collection_name = collection_name_prefix + str(i)
        # thread = threading.Thread(target=run, args=(i, collection_name, vectors_list[i], pool))
        thread = threading.Thread(target=search, args=(i, collection_name, vectors, pool))
        thread.start()
        thread_list.append(thread)

    for t in thread_list:
        t.join()

    t1 = time.time()

    print("\n=========\n[{}] Search done. Total cost {} s".format(datetime.datetime.now(), t1 - t0))
