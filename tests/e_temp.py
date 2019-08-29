import sys

sys.path.append(".")
from milvus import Milvus
import random
import time


def run():
    client = Milvus()
    client.connect()

    _table = {
        'table_name': 'name',
        'dimension': 16,
        'index_file_size': 1024
    }

    _status = client.create_table(_table)

    num = 10
    vectors = [[random.random() for _ in range(16)] for _ in range(num)]
    ids = [i for i in range(num)]
    _status, _ = client.add_vectors('name', vectors, ids)
    assert _status.OK()

    time.sleep(10)
    client.disconnect()


if __name__ == '__main__':
    # run()
    dic = {'a': 1, 'b': 2}

    try:
        dic['c']
    except KeyError as e:
        print(e)
        print("")
        print("")
