import os
from multiprocessing import Process

import sys
sys.path.append(".")
from factorys import *
from milvus import Milvus

table_name = 'test_test'
dim = 512

process_num = 10

vector_num = 10000

vectors = [[random.random() for _ in range(dim)] for _ in range(vector_num)]

param = {'table_name': table_name,
         'dimension': dim,
         'index_type': IndexType.FLAT,
         'store_raw_vector': False}


def create_table(_table_name):
    milvus = Milvus()
    milvus.connect(host="localhost", port="19530")
    if milvus.has_table(_table_name):
        print(f"Table {_table_name} found, now going to delete it")
        status = milvus.delete_table(_table_name)
        assert status.OK(), "delete table {} failed".format(_table_name)

    time.sleep(5)

    if milvus.has_table(_table_name):
        raise Exception("Delete table error")

    print("delete table {} successfully!".format(_table_name))

    # wait for table deleted

    status = milvus.create_table(param)
    if not status.OK():
        print("Create table {} failed".format(_table_name))

    # in main process, milvus must be closed before subprocess start
    milvus.disconnect()

    time.sleep(1)


def multi_conn(_table_name, proce_num):
    """
    insert vectors with multiple processes, and record execute time with decorator

    :param _table_name:
    :param proce_num:
    :return:
    """

    # contain whole subprocess
    process_list = []

    def _add():
        milvus = Milvus()
        status = milvus.connect()

        if not status.OK:
            print(f'PID: {os.getpid()}, connect failed')

        status, _ = milvus.add_vectors(_table_name, vectors)

        milvus.disconnect()

    for i in range(proce_num):
        p = Process(target=_add)
        process_list.append(p)
        p.start()

    # block main process util whole sub process exit
    for p in process_list:
        p.join()

    print("insert vector successfully!")


def validate_insert(_table_name):
    milvus = Milvus()
    milvus.connect(host="localhost", port="19530")

    status, count = milvus.get_table_row_count(_table_name)

    assert count == vector_num * process_num, f"Error: validate insert not pass: " \
                                              f"{vector_num * process_num} expected but {count} instead!"

    milvus.disconnect()


def run_multi_process(_table_name, _process_num):
    create_table(_table_name)
    multi_conn(_table_name, _process_num)

    # sleep 3 seconds to wait for vectors inserted Preservation
    time.sleep(3)

    validate_insert(_table_name)


if __name__ == '__main__':
    run_multi_process(table_name, process_num)
