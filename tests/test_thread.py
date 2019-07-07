import sys
sys.path.append('.')
from milvus import Milvus, IndexType
from pprint import pprint
import concurrent.futures
import threading
import random
import os

dimension = 256
number = 10000
pool_size = 1000
table_pool_size = 1000
step = number//pool_size

vectors = [[random.random()for _ in range(dimension)] for _ in range(number)]
table_name = 'multi_task'

start_count = 0
count = 0

def add_vector_task(milvus, vector):
    global count
    global start_count
    start_count += 1
    print('start....................{}'.format(start_count))
    status, ids = milvus.add_vectors(table_name=table_name, records=vector)
    count += 1
    print("end...............{}".format(count))


def thread_pool_add_vector():

    milvus = Milvus()
    milvus.connect(uri='tcp://127.0.0.1:19530')

    if not milvus.has_table(table_name):
        param = {
            'table_name': table_name,
            'dimension': dimension,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }

        milvus.create_table(param)

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for x in range(0, number, step):
            executor.submit(add_vector_task, milvus, vectors[x:x+step])


def add_table_task(milvus):
    global count
    global start_count
    start_count += 1
    param = {
        'table_name': unique_table_name(),
        'dimension': dimension,
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    }
    print('start....................{}'.format(start_count))
    status = milvus.create_table(param)
    count += 1
    print("end...............{}".format(count))

def unique_table_name():
    return str(random.random())

def thread_pool_add_table():
    milvus = Milvus()
    milvus.connect(uri='tcp://localhost:19530')

    with concurrent.futures.ThreadPoolExecutor(max_workers=table_pool_size) as executor:
        for x in range(0, table_pool_size):

            executor.submit(add_table_task, milvus)



if __name__ == '__main__':
    thread_pool_add_vector()
    #thread_pool_add_table()
    


