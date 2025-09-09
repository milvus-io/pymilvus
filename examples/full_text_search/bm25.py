from pymilvus import (
    MilvusClient,
    Function,
    FunctionType,
    DataType,
)

fmt = "\n=== {:30} ===\n"
collection_name = "doc_in_doc_out"
milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("document_content", DataType.VARCHAR, max_length=9000, enable_analyzer=True)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

bm25_function = Function(
    name="bm25_fn",
    input_field_names=["document_content"],
    output_field_names="sparse_vector",
    function_type=FunctionType.BM25,
)
schema.add_function(bm25_function)

index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={"bm25_k1": 1.2, "bm25_b": 0.75},
)

ret = milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")
print(ret)

print(fmt.format("    all collections    "))
print(milvus_client.list_collections())

print(fmt.format(f"schema of collection {collection_name}"))
print(milvus_client.describe_collection(collection_name))

rows = [
        {"id": 1, "document_content": "hello world"},
        {"id": 2, "document_content": "hello milvus"},
        {"id": 3, "document_content": "hello zilliz"},
]

print(fmt.format("Start inserting entities"))
insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)
print(fmt.format("Inserting entities done"))
print(insert_result)

texts_to_search = ["zilliz"]
search_params = {
    "metric_type": "BM25",
    "params": {}
}
print(fmt.format(f"Start search with retrieve several fields."))
result = milvus_client.search(collection_name, texts_to_search, limit=3, output_fields=["document_content"], search_params=search_params)
for hits in result:
    for hit in hits:
        print(f"hit: {hit}")

print(fmt.format("Start query by specifying primary keys"))
query_results = milvus_client.query(collection_name, ids=[3])
print(query_results[0])

upsert_ret = milvus_client.upsert(collection_name, {"id": 2 , "document_content": "hello milvus again"})
print(upsert_ret)

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter="document_content == 'hello milvus again'")
for ret in query_results:
    print(ret)

print(f"start to delete by specifying filter in collection {collection_name}")
delete_result = milvus_client.delete(collection_name, ids=[3])
print(delete_result)

print(fmt.format("Start query by specifying filtering expression"))
query_results = milvus_client.query(collection_name, filter="document_content == 'hello zilliz'")
print(f"Query results after deletion: {query_results}")

milvus_client.drop_collection(collection_name)
