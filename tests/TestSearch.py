import sys
from time import sleep
sys.path.append('.')
from milvus import Milvus, Prepare
import random, datetime
from pprint import pprint
import logging
from factorys import *
LOGGER = logging.getLogger(__name__)

_HOST = 'localhost'
_PORT = '19530'

dim = 256
table_id = 'test_search'


class SearchOpt:
    def __init__(self, number=100, nq=1, k=5):
        self.nb = number
        self.nq = nq
        self.k = k


class TestSearch:
    def init_data(self, connect, table, nb=10):
        vectors = gen_vectors(nb, dim)
        status, ids = connect.add_vectors(table, vectors)
        sleep(6)
        return vectors, ids

    def test_search_query_range(self, connect, table):
        nb = 10
        top_k = 2
        vectors, ids = self.init_data(connect, table, nb=nb)
        query_vecs = [vectors[0]]
        query_ranges = [(get_current_day(), get_current_day())]
        status, result = connect.search_vectors(table, top_k, query_vecs, query_ranges=query_ranges)
        assert status.OK()
        for i in range(len(query_vecs)):
            assert result[i][0].id in ids
            assert result[i][0].distance == 100.0


def main():
    milvus = Milvus()
    milvus.connect(host=_HOST, port=_PORT)
    #
    # table_name = 'test_search_in_file'
    # dimension = 256

    # vectors = Prepare.records([[random.random()for _ in range(dimension)] for _ in range(20)])
    # param = {
    #     'table_name': table_name,
    #     'file_ids': ['1'],
    #     'query_records': vectors,
    #     'top_k': 5,
    #     # 'query_ranges': []  # Not fully tested yet
    # }
    # status, result = milvus.search_vectors_in_files(**param)
    # if status.OK():
    #     pprint(result)
    #
    # _, result = milvus.get_table_row_count(table_name)
    # print('# Count: {}'.format(result))

    table_name = 'test_search'
    dimension = 256
    ranges = [['2019-07-01', datetime.datetime.now()]]

    vectors = Prepare.records([[random.random()for _ in range(dimension)] for _ in range(1)])
    LOGGER.info(ranges)
    param = {
        'table_name': table_name,
        'query_records': vectors,
        'top_k': 5,
        'query_ranges': ranges
    }
    status, result = milvus.search_vectors(**param)
    if status.OK():
        pprint(result)

    # _, tables = milvus.show_tables()
    # print(tables)

    # _, result = milvus.get_table_row_count(table_name)
    # print('# Count: {}'.format(result))
    milvus.disconnect()

if __name__ == '__main__':
    main()