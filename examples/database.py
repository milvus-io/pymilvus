import numpy as np
from pymilvus import (
    MilvusClient,
    DataType
)

milvus_client = MilvusClient("http://localhost:19530")

db1Name = "db1"
# create db1
if db1Name not in milvus_client.list_databases():
    print("\ncreate database: db1")
    milvus_client.create_database(db_name=db1Name, properties={"key1":"value1"})
    db_info = milvus_client.describe_database(db_name=db1Name)
    print(db_info)


# alter_database_properties of db1
db_info = milvus_client.describe_database(db_name=db1Name)
print(db_info)
print("\nalter database properties of db1:")
milvus_client.alter_database_properties(db_name=db1Name, properties={"key": "value"})
db_info = milvus_client.describe_database(db_name=db1Name)
print(db_info)

print("\ndrop database properties of db1")
milvus_client.drop_database_properties(db_name=db1Name, property_keys=["key"])
db_info = milvus_client.describe_database(db_name=db1Name)
print(db_info)

# list database
print("\nlist databases:")
print(milvus_client.list_databases())