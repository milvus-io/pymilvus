# This program demos how to create a Milvus table, insert 20 vectors and get the table row count.

from milvus import Milvus, Prepare, IndexType, Status
import random

milvus = Milvus()

# Connect Milvus server.
# You may need to change HOST and PORT accordingly.
milvus.connect(host='localhost', port='33001')

# Table name is defined
table_name = 'demo_table'

# Create table: table name, vector dimension and index type
milvus.create_table(Prepare.table_schema(table_name, dimension=256, index_type=IndexType.IDMAP))

# Insert 20 256-dim-vectors into demo_table
vectors = Prepare.records([[random.random()for _ in range(256)] for _ in range(20)])
milvus.add_vectors(table_name=table_name, records=vectors)

# Get table row count
_, result = milvus.get_table_row_count(table_name=table_name)
print('Table {}, row counts: {}'.format(table_name, result))
