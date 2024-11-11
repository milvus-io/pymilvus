# hello_milvus.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
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
num_entities, dim = 10, 8

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

collection_name = "hello_cost"
has = utility.has_collection(collection_name)
print(f"Does collection {collection_name} exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "random"  |    Double  |                  |      "a double field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, f"{collection_name} is the simplest demo to introduce the APIs")

print(fmt.format(f"Create collection `{collection_name}`"))
hello_milvus = Collection(collection_name, schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `hello_milvus`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),  # field random, only supports list
    rng.random((num_entities, dim)),    # field embeddings, supports numpy.ndarray and list
]

insert_result = hello_milvus.insert(entities)
# OUTPUT:
# insert result: (insert count: 10, delete count: 0, upsert count: 0, timestamp: 449296288881311748, success count: 10, err count: 0, cost: 1);
# insert cost: 1
print(f"insert result: {insert_result};\ninsert cost: {insert_result.cost}")

hello_milvus.flush()
print(f"Number of entities in Milvus: {hello_milvus.num_entities}")  # check the num_entities

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for hello_milvus collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

hello_milvus.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
hello_milvus.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
vectors_to_search = entities[-1][-2:]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["random"])
end_time = time.time()

# OUTPUT:
# search result: data: ['["id: 8, distance: 0.0, entity: {\'random\': 0.9007387227368949}", "id: 0, distance: 0.49515748023986816, entity: {\'random\': 0.6378742006852851}", "id: 2, distance: 0.5305156707763672, entity: {\'random\': 0.1321158395732429}"]', '["id: 9, distance: 0.0, entity: {\'random\': 0.4494463384561439}", "id: 8, distance: 0.558194100856781, entity: {\'random\': 0.9007387227368949}", "id: 2, distance: 0.7718868255615234, entity: {\'random\': 0.1321158395732429}"]'], cost: 21;
# search cost: 21
print(f"search result: {result};\nsearch cost: {result.cost}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
print(fmt.format("Start querying with `random > 0.5`"))

start_time = time.time()
result = hello_milvus.query(expr="random > 0.5", output_fields=["random", "embeddings"])
end_time = time.time()

# OUTPUT:
# query result: data: ["{'random': 0.6378742006852851, 'embeddings': [0.18477614, 0.42930314, 0.40345728, 0.3957196, 0.6963897, 0.24356908, 0.42512414, 0.5724385], 'pk': '0'}", "{'random': 0.744296470467782, 'embeddings': [0.8349225, 0.6614872, 0.98359716, 0.15854438, 0.30939594, 0.23553558, 0.1950739, 0.80361205], 'pk': '4'}", "{'random': 0.6025374094941409, 'embeddings': [0.36677808, 0.218786, 0.25240582, 0.82230526, 0.21011819, 0.16813536, 0.8129038, 0.74800706], 'pk': '7'}", "{'random': 0.9007387227368949, 'embeddings': [0.27464902, 0.07500089, 0.57728964, 0.6654878, 0.8698446, 0.3814792, 0.8825416, 0.58730817], 'pk': '8'}"], extra_info: {'cost': '21'};
# query cost: 21
print(f"query result: {result};\nquery cost: {result.extra['cost']}")
print(search_latency_fmt.format(end_time - start_time))


# -----------------------------------------------------------------------------
# hybrid search
print(fmt.format("Start hybrid searching with `random > 0.5`"))

start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5", output_fields=["random"])
end_time = time.time()

# OUTPUT:
# search result: data: ['["id: 8, distance: 0.0, entity: {\'random\': 0.9007387227368949}", "id: 0, distance: 0.49515748023986816, entity: {\'random\': 0.6378742006852851}", "id: 7, distance: 0.670731246471405, entity: {\'random\': 0.6025374094941409}"]', '["id: 8, distance: 0.558194100856781, entity: {\'random\': 0.9007387227368949}", "id: 0, distance: 1.0780366659164429, entity: {\'random\': 0.6378742006852851}", "id: 7, distance: 1.1083570718765259, entity: {\'random\': 0.6025374094941409}"]'], cost: 21;
# search cost: 21
print(f"search result: {result};\nsearch cost: {result.cost}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by PK
# You can delete entities by their PK values using boolean expressions.
ids = insert_result.primary_keys

expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
print(fmt.format(f"Start deleting with expr `{expr}`"))

result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

delete_result = hello_milvus.delete(expr)
# OUTPUT:
# delete result: (insert count: 0, delete count: 2, upsert count: 0, timestamp: 0, success count: 0, err count: 0, cost: 2);
# delete cost: 2
print(f"delete result: {delete_result};\ndelete cost: {delete_result.cost}")

result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the hello_milvus collection
print(fmt.format(f"Drop collection `{collection_name}`"))
utility.drop_collection(collection_name)
