from pymilvus import CollectionSchema, FieldSchema, Collection, connections, DataType, Partition, utility
import random
import numpy as np
import secrets


def generate_random_hex_string(length):
    return secrets.token_hex(length // 2)


IP = "localhost"
connections.connect("default", host=IP, port="19530")

dim = 128
clean_exist = False
prepare_data = True

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="int64", dtype=DataType.INT64),
    FieldSchema(name="float", dtype=DataType.FLOAT),
    FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="bool", dtype=DataType.BOOL),
    FieldSchema(name="string", dtype=DataType.VARCHAR, max_length=512)
]
schema = CollectionSchema(fields=fields)
collection_name = 'test_group_by_' + generate_random_hex_string(24)
if clean_exist and utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(collection_name, schema=schema)
nb = 1500
batch_num = 3
vectors = [[random.random() for _ in range(dim)] for _ in range(nb)]
# insert data
if prepare_data:
    for i in range(batch_num):
        data = [
            [i for i in range(nb * i, nb * (i + 1))],
            [i % 33 for i in range(nb)],
            [np.float32(i) for i in range(nb)],
            vectors,
            [bool(random.randrange(2)) for i in range(nb)],
            [str(i % 44) for i in range(nb * i, nb * (i + 1))],
        ]
        collection.insert(data)
        print("insert data done")
        collection.flush()
    collection.create_index("float_vector", {"metric_type": "COSINE"})

# create collection and load
collection.load()
batch_size = 100
search_params = {"metric_type": "COSINE"}
result = collection.search(vectors[:3], "float_vector", search_params, limit=batch_size, timeout=600,
                           output_fields=["int64"], group_by_field="string") #set up group_by_field

for i in range(len(result)):
    resultI = result[i]
    print(f"---result{i}_size:{len(result[i])}-------------------------")
    for j in range(len(resultI)):
        print(resultI[j])
    print("----------------------------")
