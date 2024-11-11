# hello_text_embedding.py demonstrates how to insert raw data only into Milvus and perform
# dense vector based ANN search using TextEmbedding.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search
# 6. drop collection
import time

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, Function, DataType, FunctionType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

collection_name = "text_embedding"

has = utility.has_collection(collection_name)
print(f"Does collection {collection_name} exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 2 explicit fields and a function.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "id"    |   INT64    |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2| "document" | VarChar    |                  |     "raw text document"      |
# +-+------------+------------+------------------+------------------------------+
#
# Function 'text embedding' is used to convert raw text document to a dense vector representation
# and store it in the 'dense' field.
# +-+------------+-------------------+-----------+------------------------------+
# | | field name | field type        | other attr|       field description      |
# +-+------------+-------------------+-----------+------------------------------+
# |3|  "dense"   |FLOAT_VECTOR       |  dim=1536 |                              |
# +-+------------+-------------------+-----------+------------------------------+
#
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=1000),
]

text_embedding_function = Function(
    name="openai",
    function_type=FunctionType.TEXTEMBEDDING,
    input_field_names=["document"],
    output_field_names="dense",
    params={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
    }
)

schema = CollectionSchema(fields, "hello_text_embedding demo")
schema.add_function(text_embedding_function)

hello_text_embedding = Collection(collection_name, schema, consistency_level="Strong")

print(fmt.format("Start inserting documents"))
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

insert_result = hello_text_embedding.insert([docs])
ids = insert_result.primary_keys

################################################################################
# 4. create index
# We are going to create an index for collection, here we simply
# uses AUTOINDEX so Milvus can use the default parameters.
print(fmt.format("Start Creating index AUTOINDEX"))
index = {
    "index_type": "AUTOINDEX",
    "metric_type": "IP",
}

hello_text_embedding.create_index("dense", index)

################################################################################
# 5. search, query, and scalar filtering search
# After data were inserted into Milvus and indexed, you can perform:
# - search texts relevance by TextEmbedding using dense vector ANN search

# Before conducting a search or a query, you need to load the data into memory.
print(fmt.format("Start loading"))
hello_text_embedding.load()

# -----------------------------------------------------------------------------
search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 10},
}
queries = ["When was artificial intelligence founded", 
           "Where was Alan Turing born?"]

start_time = time.time()
result = hello_text_embedding.search(queries, "dense", search_params, limit=3, output_fields=["document"], consistency_level="Strong")
end_time = time.time()

for hits, text in zip(result, queries):
    print(f"result of text: {text}")
    for hit in hits:
        print(f"\thit: {hit}, document field: {hit.entity.get('document')}")
print(search_latency_fmt.format(end_time - start_time))

# Finally, drop the collection
utility.drop_collection(collection_name)
