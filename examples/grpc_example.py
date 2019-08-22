import random
import time
from functools import wraps

import sys

sys.path.append(".")
from milvus import Milvus, IndexType

_DIM = 512
nb = 10000  # number of vector dataset
nq = 10  # number of query vector
table_name = 'examples_grpc'
top_K = 1

server_config = {
    "host": 'localhost',
    "port": '19530',
}

milvus = Milvus()
milvus.connect(**server_config)


def timer(func):
    @wraps(func)
    def inner(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()

        print("[{}] Cost {:.2f} s".format(func.__name__, t1 - t0))

        return result

    return inner


def random_vectors(num):
    # generate vectors randomly
    return [[random.random() for _ in range(_DIM)] for _ in range(num)]


def create_table():
    param = {
        'table_name': table_name,
        'dimension': _DIM,
        'index_type': IndexType.IVFLAT,
        'store_raw_vector': False
    }

    if milvus.has_table(param['table_name']):
        milvus.delete_table(param['table_name'])
        time.sleep(5)

    print("Create table: {}".format(param))
    status = milvus.create_table(param)

    if status.OK():
        print("create table {} successfully!".format(table_name))
    else:
        print("create table {} failed: {}".format(table_name, status.message))


def delete_table():
    status = milvus.delete_table(table_name)
    if status.OK():
        print("table {} delete successfully!".format(table_name))
    else:
        print("table {} delete failed: {}".format(table_name, status.message))


def describe_table():
    """
    Get schema of table

    :return: None
    """
    status, schema = milvus.describe_table(table_name)
    if status.OK():
        print('Describing table `{}` ... :\n'.format(table_name))
        print('    {}'.format(schema), end='\n\n')
    else:
        print(status.message)


@timer
def insert_vectors(_vectors):
    """
    insert vectors to milvus server

    :param _vectors: list of vector to insert
    :return: None
    """
    print("Starting insert vectors ... ")

    status, ids = milvus.add_vectors(table_name=table_name, records=_vectors)

    if not status.OK():
        print("insert failed: {}".format(status.message))

    # sleep 6 seconds to wait data persisting
    time.sleep(6)

    status, count = milvus.get_table_row_count(table_name)
    # if status.OK() and count == len(_vectors):
    if status.OK():
        print("insert vectors into table `{}` successfully!".format(table_name))
    else:
        raise RuntimeError("Insert error")


@timer
def build_index():
    print("Start build index ...... ")

    status = milvus.build_index(table_name)

    if not status.OK():
        print("build index failed: {}".format(status.message))
    else:
        raise RuntimeError("Build index failed")


def search_vectors(_query_vectors):
    """
    search vectors and display results

    :param _query_vectors:
    :return: None
    """

    status, results = milvus.search_vectors(table_name=table_name, query_records=_query_vectors, top_k=top_K, nprobe=16)
    if status.OK():
        print("Search successfully!")
    else:
        print("search failed: {}".format(status.message))


if __name__ == '__main__':
    # generate dataset vectors
    vectors = random_vectors(nb)

    create_table()

    describe_table()

    insert_vectors(vectors)

    # wait for inserted vectors persisting
    time.sleep(10)

    milvus.preload_table(table_name)

    # build_index()
    # time.sleep(6)

    milvus.describe_index(table_name)

    query_vectors = random_vectors(nq)

    search_vectors(query_vectors)

    # delete index
    status = milvus.drop_index(table_name=table_name)

    delete_table()
