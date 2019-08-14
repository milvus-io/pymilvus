import sys
import time
sys.path.append('.')
from milvus import Milvus, IndexType
import random


def main():
    milvus = Milvus()
    milvus.connect()

    table_name = 'test_timeout'
    dimension = 512
    if milvus.has_table(table_name):
        milvus.delete_table(table_name)

    print(f"sleep 1 s")
    time.sleep(1)

    if not milvus.has_table(table_name):
        param = {
            'table_name': table_name,
            'dimension': dimension,
            'index_type': IndexType.IVFLAT,
            'store_raw_vector': False
        }

        milvus.create_table(param)

    vectors = [[random.random()for _ in range(dimension)] for _ in range(100000)]

    for index in range(5):
        start = time.time()
        _, ids = milvus.add_vectors(table_name=table_name, records=vectors)
        if len(ids) == 100000:
            print(f"Add vectors {index} OK!")
        r = time.time() - start
        print(f"time: {r:0.2f}")


if __name__ == '__main__':
    main()
