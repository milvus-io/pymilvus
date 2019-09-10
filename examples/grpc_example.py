import random
import time
from functools import wraps

import sys

sys.path.append(".")
from milvus import Milvus, IndexType, MetricType

_DIM = 512
nb = 100000  # number of vector dataset
nq = 2000  # number of query vector
table_name = 'examples_grpc001'
top_K = 10

server_config = {
    "host": 'localhost',
    "port": '19530',
}

milvus = Milvus()
milvus.connect()


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
        'dimension': _DIM
    }

    # if milvus.has_table(param['table_name']):
    #     milvus.delete_table(param['table_name'])
    #     time.sleep(2)

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
    status, schema = milvus.describe_table(table_name, timeout=1000)
    if status.OK():
        print('Describing table `{}` ... :\n'.format(table_name))
        print('    {}'.format(schema), end='\n\n')
    else:
        print("describe table failed: {}".format(status.message))


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

    status, count = milvus.get_table_row_count(table_name)
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


@timer
def search_vectors(_query_vectors):
    """
    search vectors and display results

    :param _query_vectors:
    :return: None
    """
    status, results = milvus.search_vectors(table_name=table_name, query_records=_query_vectors, top_k=top_K,
                                            nprobe=16)
    if not status.OK():
        print("search failed. {}".format(status))
        return status, None
    else:
        print('serach successfully!')
        return status, results


def create_add():
    vectors = random_vectors(nb)

    create_table()

    describe_table()

    insert_vectors(vectors)


def create_index():
    _index = {
        'index_type': IndexType.IVFLAT,
        'nlist': 4096,
        'metric_type': MetricType.L2
    }

    print("starting create index ... ")
    milvus.create_index(table_name, _index)
    time.sleep(2)


def search():
    query_vectors = random_vectors(nq)

    status, results = search_vectors(query_vectors)

    print(results[0])
    print(results[0][0])
    print("[id:{}, distance:{}]".format(results[0][0].id, results[0][0].distance))

    result0 = results[0]
    out = result0[:-1]
    print(out)

    for result in results:
        print("--- {}".format(result[0]))


if __name__ == '__main__':
    for _ in range(1):
        # create_add()
        create_index()
        search()
        time.sleep(2)
