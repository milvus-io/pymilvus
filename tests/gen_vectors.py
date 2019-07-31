import time
import sys
sys.path.append('.')

from milvus import Milvus, IndexType, Prepare
from milvus.client.GrpcClient import GrpcMilvus as gMilvus
from milvus.client.GrpcClient import Prepare as gPrepare
from factorys import *


NUM = 100000
DIM = 512


table_name = 'TEST'

def thrift_time_without_prepare():

    mi = Milvus()
    mi.connect(uri='tcp://localhost:19531')
    
    if mi.has_table(table_name) or (mi.describe_table(table_name)[1].dimension != DIM):
        mi.delete_table(table_name)
        time.sleep(2)

        mi.create_table({
            'table_name': table_name,
            'dimension': DIM,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
            })

    vectors = gen_vectors(num=NUM, dim=DIM)

    records = Prepare.records(vectors)


    thrift_add_vectors(mi, table_name, records)

    time.sleep(5)

    _, n = mi.get_table_row_count(table_name)
    print(f"[thrift] add {NUM} vectors successfully, total: {n}")

    mi.disconnect()

def thrift_time():

    mi = Milvus()
    mi.connect(uri='tcp://localhost:19531')
    
    if mi.has_table(table_name) or (mi.describe_table(table_name)[1].dimension != DIM):
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
    print(f"[thrift] add {NUM} vectors successfully, total: {n}")

    mi.disconnect()

def grpc_time_without_prepare():
    mi = gMilvus()
    mi.connect()
    
    if mi.has_table(table_name) or (mi.describe_table(table_name)[1].dimension != DIM):
        mi.delete_table(table_name)
        time.sleep(2)

        mi.create_table({
            'table_name': table_name,
            'dimension': DIM,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
            })

    vectors = gen_vectors(num=NUM, dim=DIM)

    insertinfo = gPrepare.insert_infos(table_name, vectors)

    grpc_add_vectors(mi, insertinfo)

    time.sleep(5)

    _, n = mi.get_table_row_count(table_name)
    print(f"[grpc] add {NUM} vectors successfully, total: {n}")

    mi.disconnect()

def grpc_time():
    mi = gMilvus()
    mi.connect()
    
    if mi.has_table(table_name) or (mi.describe_table(table_name)[1].dimension != DIM):
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

    mi.disconnect()

@time_it
def thrift_add_vectors_without_prepare(milvus, table_name, vectors):
    ids = milvus._client.AddVector(table_name, vectors)

@time_it
def grpc_add_vectors_without_prepare(milvus, insertinfo):
    ids = milvus._stub.InsertVector(insertinfo)

@time_it
def gen(milvus, table_name, vectors):
    status, _ = milvus.add_vectors(table_name, vectors)
    assert status.OK()



if __name__ == "__main__":
    thrift_time()
    grpc_time()
    
    thrift_time_without_prepare()
    grpc_time_without_prepare()
