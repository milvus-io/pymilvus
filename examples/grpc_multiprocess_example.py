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
    """
    record time when function execute.
    this function is a decorator

    :param func:
    :return:
    """
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

        # wait for table deleted
        time.sleep(5)

    milvus.create_table(param)

    # in main process, milvus must be closed before subprocess start
    milvus.disconnect()

    time.sleep(1)


@timer
def multi_conn(_table_name, proce_num):
    """
    insert vectors with multiple processes, and record execute time with decorator

    :param _table_name:
    :param proce_num:
    :return:
    """

    # contain whole subprocess
    process_list = []

    def _conn():
        milvus = GrpcMilvus()
        status = milvus.connect()

        if not status.OK:
            print(f'PID: {os.getpid()}, connect failed')

        global vectors_list, process_value_list
        index = process_value_list.index(proce_num)
        print(f"PID = {os.getpid()}, ready to insert {len(vectors_list[index])} vectors")
        status, _ = milvus.add_vectors(_table_name, vectors_list[index])

        milvus.disconnect()

    for i in range(proce_num):
        p = Process(target=_conn)
        process_list.append(p)
        p.start()

    # block main process util whole sub process exit
    for p in process_list:
        p.join()


def validate_insert(_table_name):
    milvus = GrpcMilvus()
    milvus.connect(host="localhost", port="19530")

    status, count = milvus.get_table_row_count(_table_name)

    assert count == vector_num, f"Error: validate insert not pass: " \
                                f"{vector_num} expected but {count} instead!"

    milvus.disconnect()


def run_multi_proce(_table_name):
    for np in process_value_list:
        create_table(table_name)
        multi_conn(table_name, np)

        # sleep 3 seconds to wait for vectors inserted Preservation
        time.sleep(3)

        validate_insert(table_name)


if __name__ == '__main__':
    run_multi_proce(table_name)
