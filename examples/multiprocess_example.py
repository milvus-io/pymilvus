## This program demos how to use milvus python client with multi-process

import os
from multiprocessing import Process

from factorys import *
from milvus import Milvus

############### Global variable ###########
table_name = 'test_test'

param = {'table_name': table_name,
         'dimension': 128,
         'index_file_size': 1024,
         'metric_type': MetricType.L2}

server_config = {
    'host': "127.0.0.1",
    'port': "19530"
}


## Utils
def _generate_vectors(_dim, _num):
    return [[random.random() for _ in range(_dim)] for _ in range(_num)]


def prepare_table(_table_name):
    def _create_table(_table_param):
        milvus = Milvus()
        milvus.connect(**server_config)
        status, ok = milvus.has_table(_table_name)
        if ok:
            print("Table {} found, now going to delete it".format(_table_name))
            status = milvus.delete_table(_table_name)
            if not status.OK():
                raise Exception("Delete table error")
            print("delete table {} successfully!".format(_table_name))
        time.sleep(5)

        status, ok = milvus.has_table(_table_name)
        if ok:
            raise Exception("Delete table error")

        status = milvus.create_table(param)
        if not status.OK():
            print("Create table {} failed".format(_table_name))

        milvus.disconnect()

    # generate a process to run func `_create_table`. A exception will be raised if
    # func `_create_table` run in main process
    p = Process(target=_create_table, args=(param,))
    p.start()
    p.join()


def multi_insert(_table_name):
    # contain whole subprocess
    process_list = []

    def _add():
        milvus = Milvus()
        status = milvus.connect(**server_config)

        vectors = _generate_vectors(128, 10000)
        print('\n\tPID: {}, insert {} vectors'.format(os.getpid(), 10000))
        status, _ = milvus.add_vectors(_table_name, vectors)

        milvus.disconnect()

    for i in range(10):
        p = Process(target=_add)
        process_list.append(p)
        p.start()

    # block main process until whole sub process exit
    for p in process_list:
        p.join()

    print("insert vector successfully!")


def validate_insert(_table_name):
    milvus = Milvus()
    milvus.connect(**server_config)

    status, count = milvus.count_table(_table_name)
    assert count == 10 * 10000, "Insert validate fail. Vectors num is not matched."
    milvus.disconnect()


def main(_table_name):
    prepare_table(_table_name)

    # use multiple process to insert data
    multi_insert(_table_name)

    # sleep 3 seconds to wait for vectors inserted Preservation
    time.sleep(3)

    validate_insert(_table_name)


if __name__ == '__main__':
    main(table_name)
