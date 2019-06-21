from milvus import Milvus, Prepare, IndexType, Status
import random, time
from pprint import pprint

_HOST = 'localhost'
_PORT = '33001'


def main():
    milvus = Milvus()

    # Print client version
    print('# Client version: {}'.format(milvus.client_version()))

    # Connect milvus server
    # Please change HOST and PORT to the correct one
    param = {'host': _HOST, 'port': _PORT}
    cnn_status = milvus.connect(**param)
    print('# Connect Status: {}'.format(cnn_status))

    # Check if connected
    print('# Is connected: {}'.format(milvus.connected))

    # Print milvus server version
    print('# Server version: {}'.format(milvus.server_version()))

    # Describe table
    table_name = 'table01'
    res_status, table = milvus.describe_table(table_name)
    print('# Describe table status: {}'.format(res_status))
    print('# Describe table:{}'.format(table))

    # Create table
    # Check if `table01` exists, if not, create a table `table01`
    dimension = 256
    if not table:
        param = {
            'table_name': table_name,
            'dimension': dimension,
            'index_type': IndexType.IDMAP,
            'store_raw_vector': False
        }

        res_status = milvus.create_table(Prepare.table_schema(**param))
        print('# Create table status: {}'.format(res_status))

    # Show tables
    status, tables = milvus.show_tables()
    pprint(tables)

    # Add vectors
    # Prepare vector with 256 dimension
    vectors = Prepare.records([[random.random()for _ in range(dimension)] for _ in range(20)])

    # Insert vectors into table 'table01'
    status, ids = milvus.add_vectors(table_name=table_name, records=vectors)
    print('# Add vector status: {}'.format(status))
    pprint(ids)

    # Search vectors
    # When adding vectors for the first time, server will take at least 5s to
    # persist vector data, so you have to wait for 6s after adding vectors for
    # the first time.
    print('# Waiting for 6s...')
    time.sleep(6)

    q_records = Prepare.records([[random.random()for _ in range(dimension)] for _ in range(2)])

    param = {
        'table_name': table_name,
        'query_records': q_records,
        'top_k': 10,
    }
    status, results = milvus.search_vectors(**param)
    print('# Search vectors status: {}'.format(status))
    pprint(results)

    # Get table row count
    status, result = milvus.get_table_row_count(table_name)
    print('# Status: {}'.format(status))
    print('# Count: {}'.format(result))

    # Delete table `table01`
    status = milvus.delete_table(table_name)
    print('# Delete table status: {}'.format(status))

    # Disconnect
    status = milvus.disconnect()
    print('# Disconnect Status: {}'.format(status))


if __name__ == '__main__':
    main()
