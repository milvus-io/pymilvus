# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. query on entities
# 6. drop collection
import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
dim = 8

#################################################################################
# 1. connect to Milvus
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_milvus")
print(f"Does collection hello_milvus exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 4 fields.
# +-+---------------------+------------+---------------------+--------------------------------------------------+
# | |     field name      | field type |   other attributes  |                  field description               |
# +-+---------------------+------------+---------------------+--------------------------------------------------+
# |1|         "pk"        |   VarChar  |    is_primary=True  |                "primary field"                   |
# | |                     |            |     auto_id=False   |                                                  |
# +-+---------------------+------------+---------------------+--------------------------------------------------+
# |2|    "nullable_fid"   |    Double  |    nullable=True    |        "a double field can insert null"          |
# +-+---------------------+------------+---------------------+--------------------------------------------------+
# |3| "default_value_fid" |    Int64   |   default_value=1   |  "a int64 field can insert with default value"   |
# +-+---------------------+------------+---------------------+--------------------------------------------------+
# |4|     "embeddings"    | FloatVector|        dim=8        |              "float vector with dim 8"           |
# +-+---------------------+------------+---------------------+--------------------------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="nullable_fid", dtype=DataType.DOUBLE,nullable=True),
    FieldSchema(name="default_value_fid", dtype=DataType.INT64,default_value=1),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the demo to introduce the nullable and default value functions")

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# For fields marked as nullable=True: 
# you can skip the field when inserting data, or set it directly to a null value, and the system will treat it as null 
# For fields marked default_value: 
# the system will automatically apply this value if no data is specified for the field during insertion

print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)

data_rows = [
    {
        # skip the field when inserting data
        "pk": "19530",
        "embeddings": rng.random((1, dim), np.float32)[0]
    },
    {
        # set it directly to a null value
        "pk": "19531",
        "nullable_fid": None,
        "default_value_fid": None,
        "embeddings": rng.random((1, dim), np.float32)[0]
    },
]

hello_milvus.insert(data_rows)

hello_milvus.flush()
print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entities

################################################################################
# 4. create index
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)

################################################################################
# 5. query on entities
print(fmt.format("Start loading"))
hello_milvus.load()

print(fmt.format("Start querying"))
start_time = time.time()
result = hello_milvus.query(expr='pk in ["19530","19531"]', output_fields=["nullable_fid", "default_value_fid","embeddings"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. drop collection
# Finally, drop the hello_milvus collection
print(fmt.format("Drop collection `hello_milvus`"))
utility.drop_collection("hello_milvus")
