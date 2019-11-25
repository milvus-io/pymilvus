# This program demos how to connect to Milvus vector database, 
# create a vector table, 
# insert 10 vectors, 
# and execute a vector similarity search.
import sys
import numpy as np
import random
sys.path.append(".")
from milvus import Milvus, IndexType, MetricType
import time

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'  # default value

# Vector parameters
_DIM = 16  # dimension of vector
_INDEX_FILE_SIZE = 32  # max file size of stored index


def main():
    milvus = Milvus()

    # Connect to Milvus server
    # You may need to change _HOST and _PORT accordingly
    param = {'host': _HOST, 'port': _PORT}
    status = milvus.connect(**param)
    if status.OK():
        print("Server connected.")
    else:
        print("Server connect fail.")
        sys.exit(1)

    # Create table demo_table if it dosen't exist.
    table_name = 'demo_table'

    status, ok = milvus.has_table(table_name)
    if not ok:
        param = {
            'table_name': table_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.L2  # optional
        }

        milvus.create_table(param)

    # Show tables in Milvus server
    _, tables = milvus.show_tables()

    # Describe demo_table
    _, table = milvus.describe_table(table_name)
    print(table)

    # 10000 vectors with 16 dimension
    # element per dimension is float32 type
    # vectors should be a 2-D array
    vectors = [[random.random() for _ in range(_DIM)] for _ in range(10000)]
    # You can also use numpy to generate random vectors:
    #     `vectors = np.random.rand(10000, 16).astype(np.float32).tolist()`

    # Insert vectors into demo_table, return status and vectors id list
    status, ids = milvus.insert(table_name=table_name, records=vectors)

    # Wait for 6 seconds, until Milvus server persist vector data.
    time.sleep(6)

    # Get demo_table row count
    status, result = milvus.count_table(table_name)

    # create index of vectors, search more rapidly
    index_param = {
        'index_type': IndexType.IVFLAT,  # choice ivflat index
        'nlist': 2048
    }

    # Create ivflat index in demo_table
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    status = milvus.create_index(table_name, index_param)

    # describe index, get information of index
    status, index = milvus.describe_index(table_name)
    print(index)

    # Use the top 10 vectors for similarity search
    query_vectors = vectors[0:10]

    # execute vector similarity search
    param = {
        'table_name': table_name,
        'query_records': query_vectors,
        'top_k': 1,
        'nprobe': 16
    }
    status, results = milvus.search_vectors(**param)

    if status.OK():
        # indicate search result
        # also use by:
        #   `results.distance_array[0][0] == 0.0 or results.id_array[0][0] == ids[0]`
        if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
            print('Query result is correct')
        else:
            print('Query result isn\'t correct')

    # print results
    print(results)

    # Delete demo_table
    status = milvus.drop_table(table_name)

    # Disconnect from Milvus
    status = milvus.disconnect()


if __name__ == '__main__':
    main()
