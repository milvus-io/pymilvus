import random
import time
from milvus import Milvus, Status, IndexType

_DIM = 512
nb = 100000  # number of vector dataset
nq = 10  # number of query vector
table_name = 'example'
top_K = 1

server_config = {
    "host": 'localhost',
    "port": '19530',
}

milvus = Milvus()
milvus.connect(**server_config)


def ramdom_vectors(num):

    # generate vectors randomly
    return [[random.random() for _ in range(_DIM)] for _ in range(num)]


def create_table():
    param = {
        'table_name': table_name,
        'dimension': _DIM,
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    }

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
        print("table {} delet failed: {}".format(table_name, status.message))


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


def insert_vectors(_vectors):
    """
    insert vectors to milvus server

    :param _vectors: list of vector to insert
    :return: None
    """
    status, ids = milvus.add_vectors(table_name=table_name, records=_vectors)

    if not status.OK():
        print("insert failed: {}".format(status.message))

    # sleep 6 seconds to wait data persisting
    time.sleep(6)

    status, count = milvus.get_table_row_count(table_name)
    if status.OK() and count == len(_vectors):
        print("insert vectors into table `{}` succesfully!".format(table_name))


def search_vectors(query_vectors):
    """
    search vectors and display results

    :param query_vectors:
    :return: None
    """

    status, results = milvus.search_vectors(table_name=table_name, query_records=query_vectors, top_k=top_K)
    if status.OK():
        print("Search successfully!")
        for result_record in results:
            print(result_record)
    else:
        print("serach failed: {}".format(status.message))


if __name__ == '__main__':

    # generate dataset vectors
    vectors = ramdom_vectors(nb)

    create_table()

    describe_table()

    insert_vectors(vectors)

    query_vectors = ramdom_vectors(nq)

    search_vectors(query_vectors)

    delete_table()
