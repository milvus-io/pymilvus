# This program demos how to connect to Milvus vector database,
# create a vector table,
# insert 10 vectors,
# and execute a vector similarity search.
import sys
sys.path.append(".")
import random
from milvus import Milvus, IndexType, MetricType

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 128  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index

if __name__ == '__main__':
    client = Milvus()
    # server default host: 127.0.0.1. port: 19530
    client.connect()

    table_name = "example_index"
    param = {
        'table_name': table_name,
        'dimension': _DIM,
        'index_file_size': _INDEX_FILE_SIZE,  # optional
        'metric_type': MetricType.L2  # optional
    }

    client.create_table(param)

    vectors = [[random.random() for _ in range(_DIM)] for _ in range(100000)]
    client.insert(table_name, vectors)

    client.flush([table_name])

    # create index IVF_FLAT
    ivf_param = {
        "nlist": 4096
    }
    status = client.create_index(table_name, IndexType.IVF_FLAT, ivf_param)
    if status.OK():
        print("Create index IVF_FLAT: successfully")
    client.drop_index(table_name)

    # create index IVF_PQ
    pq_param = {
        "m": 12,
        "nlist": 4096
    }
    client.create_index(table_name, IndexType.IVF_PQ, pq_param)
    if status.OK():
        print("Create index IVF_PQ: successfully")
    client.drop_index(table_name)

    sq8_param = {
        "nlist": 4096
    }
    client.create_index(table_name, IndexType.IVF_SQ8, sq8_param)
    if status.OK():
        print("Create index IVF_SQ8: successfully")
    client.drop_index(table_name)

    nsg_param = {
        "search_length": 45,
        "out_degree": 50,
        "pool_size": 300,
        "knng": 100
    }
    client.create_index(table_name, IndexType.RNSG, nsg_param)
    if status.OK():
        print("Create index NSG: successfully")
    client.drop_index(table_name)

    hnsw_param = {
        "M": 16,
        "efConstruction": 500
    }
    client.create_index(table_name, IndexType.HNSW, hnsw_param)
    if status.OK():
        print("Create index HNSW: successfully")
    client.drop_index(table_name)
