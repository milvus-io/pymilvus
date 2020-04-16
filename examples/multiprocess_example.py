# This program demos how to use milvus python client with multi-process
# See more details and caution in:
# https://milvus.io/docs/v0.6.0/faq/operational_faq.md#Why-does-my-multiprocessing-program-fail

import os
from multiprocessing import Process

from factorys import *
from milvus import Milvus

############### Global variable ###########
collection_name = 'test_test'

param = {'collection_name': collection_name,
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


def prepare_collection(_collection_name):
    def _create_collection(_collection_param):
        milvus = Milvus(**server_config)
        milvus.connect()
        status, ok = milvus.has_collection(_collection_name)
        if ok:
            print("Table {} found, now going to delete it".format(_collection_name))
            status = milvus.drop_collection(_collection_name)
            if not status.OK():
                raise Exception("Delete collection error")
            print("delete collection {} successfully!".format(_collection_name))
        time.sleep(5)

        status, ok = milvus.has_collection(_collection_name)
        if ok:
            raise Exception("Delete collection error")

        status = milvus.create_collection(param)
        if not status.OK():
            print("Create collection {} failed".format(_collection_name))

        milvus.disconnect()

    # generate a process to run func `_create_collection`. A exception will be raised if
    # func `_create_collection` run in main process
    p = Process(target=_create_collection, args=(param,))
    p.start()
    p.join()


def multi_insert(_collection_name):
    # contain whole subprocess
    process_list = []

    def _add():
        milvus = Milvus(**server_config)
        status = milvus.connect()

        vectors = _generate_vectors(128, 10000)
        print('\n\tPID: {}, insert {} vectors'.format(os.getpid(), 10000))
        status, _ = milvus.add_vectors(_collection_name, vectors)

        milvus.disconnect()

    for i in range(10):
        p = Process(target=_add)
        process_list.append(p)
        p.start()

    # block main process until whole sub process exit
    for p in process_list:
        p.join()

    print("insert vector successfully!")


def validate_insert(_collection_name):
    milvus = Milvus()
    milvus.connect(**server_config)

    status, count = milvus.count_collection(_collection_name)
    assert count == 10 * 10000, "Insert validate fail. Vectors num is not matched."
    milvus.disconnect()


def main(_collection_name):
    prepare_collection(_collection_name)

    # use multiple process to insert data
    multi_insert(_collection_name)

    # sleep 3 seconds to wait for vectors inserted Preservation
    time.sleep(3)

    validate_insert(_collection_name)


if __name__ == '__main__':
    main(collection_name)
