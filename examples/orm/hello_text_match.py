# hello_text_match.py demonstrates how to insert raw data only into Milvus and perform
# document retrieval based on specific terms by text match expression.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and filtering search on entities
# 7. drop collection
import time
import numpy as np

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, Function, DataType, FunctionType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
dim = 8

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("hello_text_match")
print(f"Does collection hello_text_match exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 2 explicit fields and a function.
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
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    # set analyzer params in document field for more situations
    # default as analyzer_params = {"type": "standard"}
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=1000, enable_analyzer=True, enable_match=True), 
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]


schema = CollectionSchema(fields, "hello_text_match demo")

print(fmt.format("Create collection `hello_text_match`"))
hello_text_match = Collection("hello_text_match", schema, consistency_level="Strong")

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

entities = [
    [f"This is a test document {i + hello_text_match.num_entities} with keywords: {keywords[i]}" for i in range(num_entities)],
    rng.random((num_entities, dim), np.float32)
]

insert_result = hello_text_match.insert(entities)
ids = insert_result.primary_keys

hello_text_match.flush()
print(f"Number of entities in Milvus: {hello_text_match.num_entities}")  # check the num_entities

################################################################################
# 4. create index
# We are going to create an vector index for hello_text_match collection
print(fmt.format("Start Creating index AUTOINDEX"))
index = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP",
}

hello_text_match.create_index("embeddings", index)
################################################################################
# 5. query and scalar filtering search with text match
# After data were inserted into Milvus and indexed, you can perform:
# - query with text match expression
# - search data with text match filter

# Before conducting a search or a query, you need to load the data in `hello_text_match` into memory.
print(fmt.format("Start loading"))
hello_text_match.load()

# -----------------------------------------------------------------------------
# query based text match with single keyword
expr = f"TEXT_MATCH(document, '{keywords[0]}')"
print(fmt.format(f"Start querying with `{expr}`"))

start_time = time.time()
result = hello_text_match.query(expr=expr, output_fields=["document"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

# query based text match with mutiple keywords
expr = f"TEXT_MATCH(document, '{keywords[0]} {keywords[1]} {keywords[2]}')"
print(fmt.format(f"Start querying with `{expr}`"))

start_time = time.time()
result = hello_text_match.query(expr=expr, output_fields=["document"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# scalar filtering search with text match
search_params = {
    "metric_type": "IP",
    "params": {},
}
expr = f"TEXT_MATCH(document, '{keywords[0]} {keywords[1]} {keywords[2]}')"
print(fmt.format(f"Start filtered searching with `{expr}`"))

start_time = time.time()
vector_to_search = rng.random((1, dim), np.float32)
result = hello_text_match.search(vector_to_search, "embeddings", search_params, limit=3, expr=expr, output_fields=["document"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"\thit: {hit}, document field: {hit.entity.get('document')}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by text match
# You can delete entities by their PK values using boolean expressions.

expr = f"TEXT_MATCH(document, '{keywords[4]}')"
print(fmt.format(f"Start deleting with expr `{expr}`"))

result = hello_text_match.query(expr=expr, output_fields=["document"])
print(f"query before delete by expr=`{expr}` -> result: \n- {result[0]}\n")

hello_text_match.delete(expr)

result = hello_text_match.query(expr=expr, output_fields=["document"])
print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the hello_text_match collection
print(fmt.format("Drop collection `hello_text_match`"))
utility.drop_collection("hello_text_match")
