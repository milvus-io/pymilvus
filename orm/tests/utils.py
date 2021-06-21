import random
import pandas
from sklearn import preprocessing
from pymilvus_orm.types import DataType

default_dim = 128
default_nb = 1200
default_nq = 10
default_float_vec_field_name = "float_vector"

all_index_types = [
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_SQ8_HYBRID",
    "IVF_PQ",
    "HNSW",
    # "NSG",
    "ANNOY",
    "RHNSW_PQ",
    "RHNSW_SQ",
    "BIN_FLAT",
    "BIN_IVF_FLAT"
]

default_index_params = [
    {"nlist": 128},
    {"nlist": 128},
    {"nlist": 128},
    {"nlist": 128},
    {"nlist": 128, "m": 16, "nbits": 8},
    {"M": 48, "efConstruction": 500},
    # {"search_length": 50, "out_degree": 40, "candidate_pool_size": 100, "knng": 50},
    {"n_trees": 50},
    {"M": 48, "efConstruction": 500, "PQM": 64},
    {"M": 48, "efConstruction": 500},
    {"nlist": 128},
    {"nlist": 128}
]

def binary_support():
    return ["BIN_FLAT", "BIN_IVF_FLAT"]

def gen_collection_name():
    return f'ut-collection-' + str(random.randint(100000, 999999))


def gen_partition_name():
    return f'ut-partition-' + str(random.randint(100000, 999999))


def gen_index_name():
    return f'ut-index-' + str(random.randint(100000, 999999))


def gen_field_name():
    return f'ut-field-' + str(random.randint(100000, 999999))


def gen_schema():
    from pymilvus_orm.schema import CollectionSchema, FieldSchema
    fields = [
        FieldSchema(gen_field_name(), DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(gen_field_name(), DataType.FLOAT),
        FieldSchema(gen_field_name(), DataType.FLOAT_VECTOR, dim=default_dim)
    ]
    collection_schema = CollectionSchema(fields)
    return collection_schema


def gen_vectors(num, dim, is_normal=True):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num)]
    vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
    return vectors.tolist()


def gen_int_attr(row_num):
    return [random.randint(0, 255) for _ in range(row_num)]


# pandas.DataFrame
def gen_pd_data(nb, is_normal=False):
    import numpy
    vectors = gen_vectors(nb, default_dim, is_normal)
    datas = {
        "int64": [i for i in range(nb)],
        "float": numpy.array([i for i in range(nb)], dtype=numpy.float32),
        default_float_vec_field_name: vectors
    }
    data = pandas.DataFrame(datas)
    return data


# list or tuple data
def gen_list_data(nb, is_normal=False):
    vectors = gen_vectors(nb, default_dim, is_normal)
    datas = [[i for i in range(nb)], [float(i) for i in range(nb)], vectors]
    return datas


def gen_index():
    nlists = [1, 1024, 16384]
    pq_ms = [128, 64, 32, 16, 8, 4]
    Ms = [5, 24, 48]
    efConstructions = [100, 300, 500]
    search_lengths = [10, 100, 300]
    out_degrees = [5, 40, 300]
    candidate_pool_sizes = [50, 100, 300]
    knngs = [5, 100, 300]

    index_params = []
    for index_type in all_index_types:
        if index_type in ["FLAT", "BIN_FLAT", "BIN_IVF_FLAT"]:
            index_params.append({"index_type": index_type, "index_param": {"nlist": 1024}})
        elif index_type in ["IVF_FLAT", "IVF_SQ8", "IVF_SQ8_HYBRID"]:
            ivf_params = [{"index_type": index_type, "index_param": {"nlist": nlist}} \
                          for nlist in nlists]
            index_params.extend(ivf_params)
        elif index_type == "IVF_PQ":
            IVFPQ_params = [{"index_type": index_type, "index_param": {"nlist": nlist, "m": m}} \
                            for nlist in nlists \
                            for m in pq_ms]
            index_params.extend(IVFPQ_params)
        elif index_type in ["HNSW", "RHNSW_SQ", "RHNSW_PQ"]:
            hnsw_params = [{"index_type": index_type, "index_param": {"M": M, "efConstruction": efConstruction}} \
                           for M in Ms \
                           for efConstruction in efConstructions]
            index_params.extend(hnsw_params)
        elif index_type == "NSG":
            nsg_params = [{"index_type": index_type,
                           "index_param": {"search_length": search_length, "out_degree": out_degree,
                                           "candidate_pool_size": candidate_pool_size, "knng": knng}} \
                          for search_length in search_lengths \
                          for out_degree in out_degrees \
                          for candidate_pool_size in candidate_pool_sizes \
                          for knng in knngs]
            index_params.extend(nsg_params)

    return index_params

def gen_simple_index():
    index_params = []
    for i in range(len(all_index_types)):
        if all_index_types[i] in binary_support():
            continue
        dic = {"index_type": all_index_types[i], "metric_type": "L2"}
        dic.update({"params": default_index_params[i]})
        index_params.append(dic)
    return index_params
