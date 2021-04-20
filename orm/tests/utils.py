from milvus import DataType
from pymilvus_orm.schema import CollectionSchema, FieldSchema
import random
from sklearn import preprocessing

default_dim = 128
default_nb = 1200
default_float_vec_field_name = "float_vector"

def gen_collection_name():
    return f'ut-collection-' + str(random.randint(100000, 999999))

def gen_partition_name():
    return f'ut-partition-' + str(random.randint(100000, 999999))

def gen_index_name():
    return f'ut-index-' + str(random.randint(100000, 999999))

def gen_field_name():
    return f'ut-field-' + str(random.randint(100000, 999999))

def gen_schema():
    fields = [
        FieldSchema(gen_field_name(), DataType.INT64),
        FieldSchema(gen_field_name(), DataType.FLOAT_VECTOR)
    ]
    collection_schema = CollectionSchema(fields)
    return collection_schema

def gen_vectors(num, dim, is_normal=True):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()

def gen_int_attr(row_num):
    return [random.randint(0, 255) for _ in range(row_num)]

def gen_data(nb, is_normal=False):
    vectors = gen_vectors(nb, default_dim, is_normal)
    datas = [
        {"name": "int64", "type": DataType.INT64, "values": [i for i in range(nb)]},
        {"name": "float", "type": DataType.FLOAT, "values": [float(i) for i in range(nb)]},
        {"name": default_float_vec_field_name, "type": DataType.FLOAT_VECTOR, "values": vectors}
    ]
    return datas