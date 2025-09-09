from pymilvus import CollectionSchema, FieldSchema, Collection, connections, DataType, Partition, utility
import numpy as np
import random
import pandas as pd
connections.connect()

dim = 128
collection_name = "test_array"
arr_len = 100
nb = 10
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
# create collection
pk_field = FieldSchema(name="int64", dtype=DataType.INT64, is_primary=True, description='pk')
vector_field = FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
int8_array = FieldSchema(name="int8_array", dtype=DataType.ARRAY, element_type=DataType.INT8, max_capacity=arr_len)
int16_array = FieldSchema(name="int16_array", dtype=DataType.ARRAY, element_type=DataType.INT16, max_capacity=arr_len)
int32_array = FieldSchema(name="int32_array", dtype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=arr_len)
int64_array = FieldSchema(name="int64_array", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=arr_len)
bool_array = FieldSchema(name="bool_array", dtype=DataType.ARRAY, element_type=DataType.BOOL, max_capacity=arr_len)
float_array = FieldSchema(name="float_array", dtype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=arr_len)
double_array = FieldSchema(name="double_array", dtype=DataType.ARRAY, element_type=DataType.DOUBLE, max_capacity=arr_len)
string_array = FieldSchema(name="string_array", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=arr_len,
                           max_length=100)

fields = [pk_field, vector_field, int8_array, int16_array, int32_array, int64_array,
           bool_array, float_array, double_array, string_array]

schema = CollectionSchema(fields=fields)
collection = Collection(collection_name, schema=schema)

# insert data
pk_value = [i for i in range(nb)]
vector_value = [[random.random() for _ in range(dim)] for i in range(nb)]
int8_value = [[np.int8(j) for j in range(arr_len)] for i in range(nb)]
int16_value = [[np.int16(j) for j in range(arr_len)] for i in range(nb)]
int32_value = [[np.int32(j) for j in range(arr_len)] for i in range(nb)]
int64_value = [[np.int64(j) for j in range(arr_len)] for i in range(nb)]
bool_value = [[np.bool_(j) for j in range(arr_len)] for i in range(nb)]
float_value = [[np.float32(j) for j in range(arr_len)] for i in range(nb)]
double_value = [[np.double(j) for j in range(arr_len)] for i in range(nb)]
string_value = [[str(j) for j in range(arr_len)] for i in range(nb)]

data = [pk_value, vector_value,
        int8_value,int16_value, int32_value, int64_value,
        bool_value,
        float_value,
        double_value,
        string_value
        ]

#collection.insert(data)

data = pd.DataFrame({
    'int64': pk_value,
    'float_vector': vector_value,
    "int8_array": int8_value,
    "int16_array": int16_value,
    "int32_array": int32_value,
    "int64_array": int64_value,
    "bool_array": bool_value,
    "float_array": float_value,
    "double_array": double_value,
    "string_array": string_value
})
collection.insert(data)

index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

collection.create_index("float_vector", index)
collection.load()

res = collection.query("int64 >= 0", output_fields=["int8_array"])
for hits in res:
    print(hits)
