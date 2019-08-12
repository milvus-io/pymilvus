# This program demos how to create a Milvus table, insert 20 vectors and get the table row count.
from milvus import Milvus, Prepare, IndexType, Status

import random
import time

milvus = Milvus()

# Connect Milvus server.
# You may need to change HOST and PORT accordingly.
milvus.connect(host='localhost', port='19530')

# Table name is defined
table_name = 'demo_table_02'

# Create table: table name, vector dimension and index type
if not milvus.has_table(table_name):
    milvus.create_table({'table_name': table_name, 'dimension': 256, 'index_type': IndexType.FLAT})

# Insert 20 256-dim-vectors into demo_table
vectors = [[random.random()for _ in range(256)] for _ in range(20)]
milvus.add_vectors(table_name=table_name, records=vectors)
time.sleep(1)

# Get table row count
_, result = milvus.get_table_row_count(table_name=table_name)
print('Table {}, row counts: {}'.format(table_name, result))
