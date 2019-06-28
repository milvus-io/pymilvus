import sys
sys.path.append('.')
from milvus import Milvus, Prepare
import random, datetime
from pprint import pprint
import logging
LOGGER = logging.getLogger(__name__)

_HOST = 'localhost'
_PORT = '19530'


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
    # param = {'start_date': '2019-06-24', 'end_date': '2019-06-25'}
    ranges = [['2019-06-25', '2019-06-25']]

    vectors = Prepare.records([[random.random()for _ in range(dimension)] for _ in range(1)])
    # ranges = [Prepare.range(**param)]
    LOGGER.info(ranges)
    param = {
        'table_name': table_name,
        'query_records': vectors,
        'top_k': 5,
        # 'query_ranges': ranges
    }
    status, result = milvus.search_vectors(**param)
    if status.OK():
        pprint(result)

    _, result = milvus.get_table_row_count(table_name)
    print('# Count: {}'.format(result))
    milvus.disconnect()


if __name__ == '__main__':
    main()
