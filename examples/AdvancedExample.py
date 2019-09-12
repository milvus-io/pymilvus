import random
import time

from milvus import Milvus, IndexType, MetricType

_DIM = 512
nb = 500000  # number of vector dataset
nq = 2000  # number of query vector
table_name = 'examples_milvus'
top_K = 1000

server_config = {
    "host": 'localhost',
    "port": '19530',
}

milvus = Milvus()
milvus.connect()


def random_vectors(num):
    # generate vectors randomly
    return [[random.random() for _ in range(_DIM)] for _ in range(num)]


def create_table():
    param = {
        'table_name': table_name,
        'dimension': _DIM
    }

    if milvus.has_table(param['table_name']):
        milvus.delete_table(param['table_name'])
        time.sleep(2)

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


def build_index():
    print("Start build index ...... ")

    index = {
        "index_type": IndexType.IVFLAT,
        "nlist": 4096
    }

    status = milvus.create_index(table_name, index)

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
    status, results = milvus.search_vectors(table_name=table_name, query_records=_query_vectors, top_k=top_K,
                                            nprobe=16)
    if not status.OK():
        print("search failed. {}".format(status))
    else:
        print('serach successfully!')
        print("\nresult:\n{}".format(results))


def create_index():
    _index = {
        'index_type': IndexType.IVFLAT,
        'nlist': 4096,
        'metric_type': MetricType.L2
    }

    print("starting create index ... ")
    milvus.create_index(table_name, _index)
    time.sleep(2)


def run():
    vectors = random_vectors(nb)

    create_table()

    describe_table()

    insert_vectors(vectors)

    # wait for data persisting
    time.sleep(2)

    # create index
    create_index()

    milvus.describe_table(table_name)

    status, index = milvus.describe_index(table_name)
    print(index)

    # generate query vectors
    query_vectors = random_vectors(nq)

    search_vectors(query_vectors)

    # delete index
    milvus.drop_index(table_name=table_name)

    time.sleep(3)

    # delete table
    delete_table()


if __name__ == '__main__':
    run()
