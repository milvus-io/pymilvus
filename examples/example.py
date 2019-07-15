# This program demos how to connect to Milvus vector database, 
# create a vector table, 
# insert 10 vectors, 
# and execute a vector similarity search.

from milvus import Milvus, Prepare, IndexType, Status
import time

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = '127.0.0.1'
_PORT = '19530'


def main():
    milvus = Milvus()

    # Connect to Milvus server
    # You may need to change _HOST and _PORT accordingly
    param = {'host': _HOST, 'port': _PORT}
    status = milvus.connect(**param)
    print('Connect to {}:{}\n'.format(_HOST, _PORT))

    # Create table demo_table if it dosen't exist.
    table_name = 'demo_table'

    if not milvus.has_table(table_name):
        param = {
            'table_name': table_name,
            'dimension': 16,
            'index_type': IndexType.FLAT,
            'store_raw_vector': False
        }

        milvus.create_table(param)
        print('Creating table {}\n'.format(table_name))

    # Show tables in Milvus server
    _, tables = milvus.show_tables()
    print('Showing tables: {}\n'.format(tables))

    # Describe demo_table
    _, table = milvus.describe_table(table_name)
    print('Describing table {}: {}\n'.format(table_name, table))

    # 10 vectors with 16 dimension
    vectors = [
        [0.66, 0.01, 0.29, 0.64, 0.75, 0.94, 0.26, 0.79, 0.61, 0.11, 0.25, 0.50, 0.74, 0.37, 0.28, 0.63],
        [0.77, 0.65, 0.57, 0.68, 0.29, 0.93, 0.17, 0.15, 0.95, 0.09, 0.78, 0.37, 0.76, 0.21, 0.42, 0.15],
        [0.61, 0.38, 0.32, 0.39, 0.54, 0.93, 0.09, 0.81, 0.52, 0.30, 0.20, 0.59, 0.15, 0.27, 0.04, 0.37],
        [0.33, 0.03, 0.87, 0.47, 0.79, 0.61, 0.46, 0.77, 0.62, 0.70, 0.85, 0.01, 0.30, 0.41, 0.74, 0.98],
        [0.19, 0.80, 0.03, 0.75, 0.22, 0.49, 0.52, 0.91, 0.40, 0.91, 0.79, 0.08, 0.27, 0.16, 0.07, 0.24],
        [0.44, 0.36, 0.16, 0.88, 0.30, 0.79, 0.45, 0.31, 0.45, 0.99, 0.15, 0.93, 0.37, 0.25, 0.78, 0.84],
        [0.33, 0.37, 0.59, 0.66, 0.76, 0.11, 0.19, 0.38, 0.14, 0.37, 0.97, 0.50, 0.08, 0.69, 0.16, 0.67],
        [0.68, 0.97, 0.20, 0.13, 0.30, 0.16, 0.85, 0.21, 0.26, 0.17, 0.81, 0.96, 0.18, 0.40, 0.13, 0.74],
        [0.11, 0.26, 0.44, 0.91, 0.89, 0.79, 0.98, 0.91, 0.09, 0.45, 0.07, 0.88, 0.71, 0.35, 0.97, 0.41],
        [0.17, 0.54, 0.61, 0.58, 0.25, 0.63, 0.65, 0.71, 0.26, 0.80, 0.28, 0.77, 0.69, 0.02, 0.63, 0.60],
    ]

    # Insert vectors into demo_table
    status, ids = milvus.add_vectors(table_name=table_name, records=vectors)
    print('Adding vectors into table {}\n'.format(table_name))
    time.sleep(1)

    # Get demo_table row count
    status, result = milvus.get_table_row_count(table_name)
    print('Row counts of table {}: {}\n'.format(table_name, result))

    # Wait for 6 seconds, since Milvus server persist vector data every 5 seconds by default.
    # You can set data persist interval in Milvus config file.
    time.sleep(6)
    print('Waiting for 6s ...\n')

    # Use the 3rd vector for similarity search
    query_vectors = [
        vectors[3]
    ]

    # execute vector similarity search
    param = {
        'table_name': table_name,
        'query_records': query_vectors,
        'top_k': 1,
    }
    status, results = milvus.search_vectors(**param)
    print('Searching vectors in table {} by the 3rd vector we just added ...\n'.format(table_name))
    
    print('Compareing result with the 3rd vector ... \n')
    if results[0][0].score == 100.0 or result[0][0].id == ids[3]:
        print('Query result is correct\n')
    else:
        print('Query result isn\'t correct\n')

    # Delete demo_table
    status = milvus.delete_table(table_name)
    print('Deleting table {} ...\n'.format(table_name))

    # Disconnect from Milvus
    status = milvus.disconnect()
    print('Disconnecting with the server ...\n')


if __name__ == '__main__':
    main()
