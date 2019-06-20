from milvus import Milvus, Prepare, IndexType, Status
import random, struct

milvus = Milvus()
table_name = 'table_'+str(random.randint(0,100))

# Connect
milvus.connect(host='localhost', port='33001')

# Create table
milvus.create_table(Prepare.table_schema(table_name, dimension=256, index_type=IndexType.IDMAP))

# Add 20 256-dim-vectors to table
vectors = Prepare.records([[random.random()for _ in range(256)] for _ in range(20)])
milvus.add_vectors(table_name=table_name, records=vectors)

# Get table row count
_, result = milvus.get_table_row_count(table_name=table_name)
print('Table {}, row counts: {}'.format(table_name, result))
