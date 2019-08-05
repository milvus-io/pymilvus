import time
import os
from multiprocessing import Process
import sys
from functools import wraps

from factorys import *

sys.path.append(".")
from milvus.client.GrpcClient import GrpcMilvus

table_name = 'test_test'
dim = 512

vector_num = 100000

process_value_list = [1, 2, 5, 10, 20, 50]

vectors_list = []

for _np in process_value_list:
    vectors_list.append(gen_vectors(vector_num // _np, dim))

param = {'table_name': table_name,
         'dimension': dim,
         'index_type': IndexType.FLAT,
         'store_raw_vector': False}


def timer(func):
    @wraps(func)
    def wrapper(*argc, **argv):
        t0 = time.time()
        result = func(*argc, **argv)
        t1 = time.time()

        print(f'run function {func.__name__}== use {argc[1]} processes cost {t1 - t0} s')

        return result

    return wrapper


def create_table(_table_name):
    milvus = GrpcMilvus()
    milvus.connect(host="localhost", port="19530")
    if milvus.has_table(_table_name):
        print(f"Table {_table_name} found, now going to delete it")
        status = milvus.delete_table(_table_name)
        assert status.OK(), f"delete table {_table_name} failed"
        time.sleep(5)

    milvus.create_table(param)
    milvus.disconnect()

    time.sleep(1)


@timer
def multi_conn(_table_name, proce_num):
    process_list = []

    def _conn():
        milvus = GrpcMilvus()
        status = milvus.connect()

        if not status.OK:
            print(f'PID: {os.getpid()}, connect failed')

        global vectors_list, process_value_list
        index = process_value_list.index(proce_num)
        print(f"index = {index}, ready to insert {len(vectors_list[index])} vectors")
        status, _ = milvus.add_vectors(_table_name, vectors_list[index])

        milvus.disconnect()

    for i in range(proce_num):
        p = Process(target=_conn)
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    return True


def check_insert(_table_name, proce_num):
    milvus = GrpcMilvus()
    milvus.connect(host="localhost", port="19530")

    status, count = milvus.get_table_row_count(_table_name)

    assert count == vector_num, f"Error: insert check not pass: " \
                                f"{proce_num * vector_num} expected but {count} instead!"
    print(f'')


def run_multi_proce(_table_name):
    for np in process_value_list:
        create_table(table_name)
        multi_conn(table_name, np)
        time.sleep(1)
        check_insert(table_name, np)
        time.sleep(1)


if __name__ == '__main__':
    run_multi_proce(table_name)
    pass
