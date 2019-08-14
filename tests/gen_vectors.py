import sys

sys.path.append('.')

from milvus import Milvus
from factorys import *

NUM = 100000
DIM = 512
table_name = 'TEST'


def grpc_time_without_prepare():
    mi = Milvus()
    mi.connect()

    if mi.has_table(table_name):
        mi.delete_table(table_name)
        time.sleep(2)

    mi.create_table({
        'table_name': table_name,
        'dimension': DIM,
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    })

    vectors = gen_vectors(num=NUM, dim=DIM)

    before = time.perf_counter()
    insertinfo = gPrepare.insert_param(table_name, vectors)
    delt = time.perf_counter() - before

    grpc_add_vectors_without_prepare(mi, insertinfo)

    time.sleep(5)

    _, n = mi.get_table_row_count(table_name)
    print(f"[grpc] add {NUM} vectors successfully, total: {n}")
    print(f"gRPC Serializing costs: {delt}")


def grpc_time():
    mi = Milvus()
    mi.connect()

    if mi.has_table(table_name):
        mi.delete_table(table_name)
        time.sleep(2)

    mi.create_table({
        'table_name': table_name,
        'dimension': DIM,
        'index_type': IndexType.FLAT,
        'store_raw_vector': False
    })

    vectors = gen_vectors(num=NUM, dim=DIM)

    gen(mi, table_name, vectors)

    time.sleep(5)

    _, n = mi.get_table_row_count(table_name)
    print(f"[grpc] add {NUM} vectors successfully, total: {n}")


@time_it
def grpc_add_vectors_without_prepare(milvus, insertinfo):
    ids = milvus.add_vectors(None, None, flag=True, param=insertinfo)


@time_it
def gen(milvus, table_name, vectors):
    status, _ = milvus.add_vectors(table_name, vectors)
    assert status.OK()


if __name__ == "__main__":
    grpc_time()
    grpc_time_without_prepare()
