# This program demos how to connect Milvus server, create a table, insert 10 vectors, search vectors.

from milvus import Milvus, Prepare, IndexType, Status
import time


# Milvus server ip address and port
_HOST = '192.168.1.101'
_PORT = '33001'


def main():
    milvus = Milvus()

    # Print client version
    print('# Client version: {}'.format(milvus.client_version()))

    # Connect Milvus server
    # You may need to change _HOST and _PORT accordingly
    param = {'host': _HOST, 'port': _PORT}
    status = milvus.connect(**param)
    print('# {}'.format(status))

    # Check if connected
    print('# Is connected: {}'.format(milvus.connected))

    # Check Milvus server version
    print('# Server version: {}'.format(milvus.server_version()))

    # Describe demo_table
    table_name = 'demo_table'
    res_status, table = milvus.describe_table(table_name)
    print('# Describe table status: {}'.format(res_status).format(table))


    # Create demo_table
    # Check if `demo_table` exists, if not, create a table `demo_table`
    dimension = 16
    if not table:
        param = {
            'table_name': table_name,
            'dimension': dimension,
            'index_type': IndexType.IDMAP,
            'store_raw_vector': False
        }

        res_status = milvus.create_table(Prepare.table_schema(**param))
        print('# Create table status: {}'.format(res_status))

    # Show tables in Milvus server
    status, tables = milvus.show_tables()
    print(tables)
    print('# {}'.format(status))

    # Generate 10 vectors with 16 dimension
    vector_list = [
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
    vectors = Prepare.records(vector_list)

    # Insert vectors into table 'demo_table'
    status, ids = milvus.add_vectors(table_name=table_name, records=vectors)
    print('# Insert vector status: {}'.format(status))

    # Search vectors
    # Milvus server will take 5s(default value, can be changed in server config) to persist vector data,
    # so you have to wait for 6s.
    print('# Waiting for 6s...')
    time.sleep(6)

    # Get demo_table row count
    status, result = milvus.get_table_row_count(table_name)
    print('# demo_table row count: {}'.format(result))

    # Generate 1 vectors with 2 dimension
    query_list = [
        vector_list[3]
    ]
    query_vectors = Prepare.records(query_list)

    # Search the vector in Milvus server and expect to get same vector as the result.
    param = {
        'table_name': table_name,
        'query_records': query_vectors,
        'top_k': 1,
    }
    status, results = milvus.search_vectors(**param)
    print('# {}'.format(status))

    if results[0][0].score == 100.0 or result[0][0].id == ids[3]:
        print('# Query result is correct')
    else:
        print('# Query result isn\'t correct')

    # Delete table
    milvus.delete_table(table_name)

    # Disconnect
    milvus.disconnect()


if __name__ == '__main__':
    main()
