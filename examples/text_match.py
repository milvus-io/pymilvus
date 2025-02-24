# hello_text_match.py demonstrates how to insert raw data only into Milvus and perform
# document retrieval based on specific terms by text match expression.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. search, query, and filtering search on entities
# 5. drop collection
import time
import numpy as np

from pymilvus import (
    MilvusClient,
    Function,
    FunctionType,
    DataType,
)

fmt = "\n=== {:30} ===\n"
collection_name = "text_match"
dim = 8

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
print(fmt.format("start connecting to Milvus"))
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
print(f"Does collection hello_text_match exist in Milvus: {has_collection}")
if has_collection:
    milvus_client.drop_collection(collection_name)

#################################################################################
# 2. create collection
# We're going to create a collection with 3 explicit fields.
# +-+------------+------------+----------------------+------------------------------+
# | | field name | field type |   other attributes   |       field description      |
# +-+------------+------------+----------------------+------------------------------+
# |1|    "id"    |   INT64    |    is_primary=True   |      "primary field"         |
# | |            |            |     auto_id=False    |                              |
# +-+------------+------------+----------------------+------------------------------+
# |2| "document" | VarChar    | enable_analyzer=True |     "raw text document"      |
# | |            |            |   enable_match=True  |                              |
# +-+------------+------------+----------------------+------------------------------+
# |3|"embeddings"| FloatVector|        dim=8         |  "float vector with dim 8"   |
# +-+------------+------------+----------------------+------------------------------+

schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
# set analyzer params in document field for more situations
# default as analyzer_params = {"type": "standard"}
schema.add_field("document", DataType.VARCHAR, max_length=1000, enable_analyzer=True, enable_match=True),
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=dim)

print(fmt.format("Create collection `hello_text_match`"))

index_params = milvus_client.prepare_index_params()
index_params.add_index(
    "embeddings",
    index_type= "AUTOINDEX",
    metric_type= "IP"
)

milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 6 rows of data into `hello_text_match`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))

rng = np.random.default_rng(seed=19530)
num_entities = 6
keywords = ["milvus", "match", "search", "query", "analyzer", "tokenizer"]
embeddings = rng.random((num_entities, dim), np.float32)

entities = [{
        "id": i,
        "document":f"This is a test document {i} with keywords: {keywords[i]}",
        "embeddings": embeddings[i]
    } for i in range(num_entities)
]

insert_result = milvus_client.insert(collection_name, entities)
print(f"Number of insert entities in Milvus: {insert_result['insert_count']}")  # check the num_entities
milvus_client.flush(collection_name)

# ###############################################################################
# 4. query and scalar filtering search with text match
# After data were inserted into Milvus and indexed, you can perform:
# - query with text match expression
# - search data with text match filter

# -----------------------------------------------------------------------------
# query based text match with single keyword filter
filter = f"TEXT_MATCH(document, '{keywords[0]}')"
print(fmt.format(f"Start querying with `{filter}`"))

result = milvus_client.query(collection_name, filter, output_fields=["document"])
print(f"query result:\n-{result}")

# query based text match with mutiple keywords
filter = f"TEXT_MATCH(document, '{keywords[0]} {keywords[1]} {keywords[2]}')"
print(fmt.format(f"Start querying with `{filter}`"))

result = milvus_client.query(collection_name, filter, output_fields=["document"])
print(f"query result:\n-{result}")

# -----------------------------------------------------------------------------
# scalar filtering search with text match
search_params = {
    "metric_type": "IP",
    "params": {},
}
filter = f"TEXT_MATCH(document, '{keywords[0]} {keywords[1]} {keywords[2]}')"
print(fmt.format(f"Start filtered searching with `{filter}`"))

vector_to_search = rng.random((1, dim), np.float32)
result = milvus_client.search(collection_name ,vector_to_search, filter, anns_field="embeddings", search_params=search_params, limit=3, output_fields=["document"])

print(result)

###############################################################################
# 6. delete entities by text match filter
# You can delete entities by their PK values using boolean expressions.

filter = f"TEXT_MATCH(document, '{keywords[4]}')"
print(fmt.format(f"Start deleting with expr `{filter}`"))

result = milvus_client.query(collection_name, filter, output_fields=["document"])
print(f"query before delete by expr=`{filter}` -> result: \n- {result}\n")

milvus_client.delete(collection_name, filter=filter)

result = milvus_client.query(collection_name, filter, output_fields=["document"])
print(f"query after delete by expr=`{filter}` -> result: {result}\n")


###############################################################################
# 5. drop collection
# Finally, drop the hello_text_match collection
print(fmt.format(f"Drop collection `{collection_name}`"))
milvus_client.drop_collection(collection_name)
