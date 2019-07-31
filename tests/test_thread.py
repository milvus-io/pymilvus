import sys
sys.path.append('.')
from milvus import Milvus, IndexType
from milvus.client.GrpcClient import GrpcMilvus as gMilvus
from pprint import pprint
import concurrent.futures
import threading
import random
import os
import time
from factorys import time_it

dimension = 512
number = 100000
pool_size = 100
table_pool_size = 1000
step = number//pool_size

vectors = [[random.random()for _ in range(dimension)] for _ in range(number)]
table_name = 'multi_task'

start_count = 0
count = 0

gmilvus = gMilvus()
gmilvus.connect()

if gmilvus.has_table(table_name):
    gmilvus.delete_table(table_name)
    time.sleep(2)

gmilvus.create_table({
    'table_name': table_name,
    'dimension': dimension,
    'index_type': IndexType.FLAT,
    'store_raw_vector': False
    })

milvus = Milvus()
milvus.connect(uri='tcp://127.0.0.1:19531')

if milvus.has_table(table_name):
    milvus.delete_table(table_name)
    time.sleep(2)
param = {
    'table_name': table_name,
    'dimension': dimension,
    'index_type': IndexType.FLAT,
    'store_raw_vector': False
}

milvus.create_table(param)

def add_vector_task(milvus, vector):
    global count
    global start_count
    start_count += 1
    print('start....................{}'.format(start_count))
    status, ids = milvus.add_vectors(table_name=table_name, records=vector)
    assert len(ids) == 10000
    count += 1
    print("end...............{}".format(count))

@time_it
def thrift_thread_pool_add_vector():

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for x in range(0, number, step):
            executor.submit(add_vector_task, milvus, vectors[x:x+step])

@time_it
def grpc_thread_pool_add_vector():

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        for x in range(0, number, step):
            executor.submit(add_vector_task, gmilvus, vectors[x:x+step])


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
    thrift_thread_pool_add_vector()
    grpc_thread_pool_add_vector()

    time.sleep(10)

    _, tcount = milvus.get_table_row_count(table_name)
    _, gcount = gmilvus.get_table_row_count(table_name)
    assert tcount == number
    assert gcount == number
    #thread_pool_add_table()
    print(tcount)
    print(gcount)
