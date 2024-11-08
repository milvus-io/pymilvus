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
    MilvusClient,
    utility,
    FieldSchema, CollectionSchema, Function, DataType, FunctionType,
    Collection,
)

collection_name = "text_embedding"

milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("document", DataType.VARCHAR, max_length=9000)
schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1536)

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

schema.add_function(text_embedding_function)

index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="AUTOINDEX",
    metric_type="IP",
)

ret = milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")

rows = [
        {"id": 1, "document": "Artificial intelligence was founded as an academic discipline in 1956."},
        {"id": 2, "document": "Alan Turing was the first person to conduct substantial research in AI."},
        {"id": 3, "document": "Born in Maida Vale, London, Turing was raised in southern England."},
]

insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)


# -----------------------------------------------------------------------------
search_params = {
    "params": {"nprobe": 10},
}
queries = ["When was artificial intelligence founded", 
           "Where was Alan Turing born?"]

start_time = time.time()
result = milvus_client.search(collection_name, data=queries, anns_field="dense", search_params=search_params, limit=3, output_fields=["document"], consistency_level="Strong")           
end_time = time.time()

for hits, text in zip(result, queries):
    print(f"result of text: {text}")
    for hit in hits:
        print(f"\thit: {hit}, document field: {hit.get('document')}")

# Finally, drop the collection
milvus_client.drop_collection(collection_name)
