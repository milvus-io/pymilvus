import concurrent.futures
import datetime
import random
import sys
import threading
import time

sys.path.append('.')
from factorys import time_it
from milvus import Milvus

dimension = 512
number = 100000
table_name = 'multi_task'


def add_vector_task(milvus, vector):
    status, ids = milvus.add_vectors(table_name=table_name, records=vector)

    assert status.OK(), "add vectors failed"
    assert len(ids) == len(vector)


@time_it
def thread_pool_add_vector(milvus, pool_size, vectors):
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for _ in range(pool_size):
            executor.submit(add_vector_task, milvus, vectors)


def test_run(gcon):
    gmilvus = gcon

    if gmilvus is None:
        assert False, "Error occurred: connect failure"

    status, exists = gmilvus.has_table(table_name)
    if exists:
        gmilvus.drop_table(table_name)
        time.sleep(2)

    table_param = {
        'table_name': table_name,
        'dimension': dimension,
        'index_file_size': 1024,
        'metric_type': 1
    }

    gmilvus.create_table(table_param)

    for p in (1, 5, 10, 20, 40, 50, 100):
        pool_size = p
    step = number // pool_size

    vectors = [[random.random() for _ in range(dimension)] for _ in range(step)]

    print(f'Thread pool size................... {p}')
    thread_pool_add_vector(gmilvus, p, vectors)

    time.sleep(1.5)
    _, gcount = gmilvus.count_table(table_name)
    print(gcount)


def test_mult_insert():
    def multi_thread_opr(table_name, utid):
        print("[{}] | T{} | Running .....".format(datetime.datetime.now(), utid))

        client0 = Milvus(handler="HTTP")

        table_param = {
            'table_name': table_name,
            'dimension': 64
        }

        vectors = [[random.random() for _ in range(64)] for _ in range(10000)]

        client0.connect()
        client0.create_table(table_param)

        for i in range(20):
            print("[{}] | T{} | O{} | Start insert data .....".format(datetime.datetime.now(), utid, i))
            client0.insert(table_name, vectors)
            print("[{}] | T{} | O{} | Stop insert data .....".format(datetime.datetime.now(), utid, i))

        client0.disconnect()

        print("[{}] | T{} | Stopping .....".format(datetime.datetime.now(), utid))

    thread_list = []
    for i in range(10):
        t = threading.Thread(target=multi_thread_opr, args=("multi_table_{}".format(random.randint(0, 10000)), i))
        t.start()
        thread_list.append(t)

    for tr in thread_list:
        tr.join(timeout=None)

    print("Done")
