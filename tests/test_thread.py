import sys
sys.path.append('.')
from milvus import Milvus, IndexType, Prepare
from milvus.client.GrpcClient import GrpcMilvus as gMilvus
from milvus.client.GrpcClient import Prepare as gPrepare
from pprint import pprint
import concurrent.futures
import threading
import random
import os
import time
from factorys import time_it


dimension = 512
number = 100000
table_name = 'multi_task'

#@time_it
def add_vector_task(milvus, vector):
    status, ids = milvus.add_vectors(table_name=table_name, records=vector)

@time_it
def thrift_thread_pool_add_vector(milvus, pool_size, vectors):

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for x in range(pool_size):
            executor.submit(add_vector_task, milvus, vectors)

@time_it
def grpc_thread_pool_add_vector(milvus, pool_size, vectors):

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for x in range(pool_size):
            executor.submit(add_vector_task, milvus, vectors)


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


def thrift_run():

    milvus = Milvus()
    milvus.connect(uri='tcp://192.168.1.197:19531')
    #milvus.connect(uri='tcp://127.0.0.1:19531')

    if milvus.has_table(table_name):
        print('Deleting existing table %s' % table_name)
        milvus.delete_table(table_name)
        time.sleep(2)
    param = {
        'table_name': table_name,
        'dimension': dimension,
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    }
    print('Creating table %s' % table_name)

    milvus.create_table(param)

    for p in (1, 5, 10, 20, 40, 50, 100):
        pool_size = p
        step = number//pool_size

        vectors = [[random.random()for _ in range(dimension)] for _ in range(step)]

        print(f'Thread pool size................... {p}')
        thrift_thread_pool_add_vector(milvus, p, vectors)

        time.sleep(3)

        _, tcount = milvus.get_table_row_count(table_name)
        print(tcount)


def grpc_run():

    gmilvus = gMilvus()
    gmilvus.connect(uri='tcp://192.168.1.197:20000')
    #gmilvus.connect()

    if gmilvus.has_table(table_name):
        gmilvus.delete_table(table_name)
        time.sleep(2)

    gmilvus.create_table({
        'table_name': table_name,
        'dimension': dimension,
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
        })

    for p in (1, 5, 10, 20, 40, 50, 100):
        pool_size = p
        step = number//pool_size

        vectors = [[random.random()for _ in range(dimension)] for _ in range(step)]

        print(f'Thread pool size................... {p}')
        grpc_thread_pool_add_vector(gmilvus, p, vectors)


        time.sleep(1.5)
        _, gcount = gmilvus.get_table_row_count(table_name)
        print(gcount)


if __name__ == '__main__':

    #thrift_run()
    grpc_run()
