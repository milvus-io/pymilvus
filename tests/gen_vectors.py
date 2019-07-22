import time
import sys
sys.path.append('.')

from milvus import Milvus, IndexType
from factorys import *



def main():
    NUM = 10000
    DIM = 256

    if sys.argv[1:2] == '-n':
        NUM = int(sys.argv[2:])
    if sys.argv[3:4] == '-d':
        DIM = int(sys.argv[4:])


    table_name = 'TEST'

    mi = Milvus()
    mi.connect()
    
    if not mi.has_table(table_name) or (mi.describe_table(table_name)[1].dimension != DIM):
        mi.delete_table(table_name)
        time.sleep(2)

        mi.create_table({
            'table_name': table_name,
            'dimension': DIM,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
            })

    vectors = gen_vectors(num=NUM, dim=DIM)

    mi.add_vectors(table_name, vectors)

    time.sleep(5)

    _, n = mi.get_table_row_count(table_name)
    print(f"Add {NUM} vectors successfully, total: {n}")

    mi.disconnect()


if __name__ == "__main__":
    main()
