from pymilvus import (
    MilvusClient,
    Function, DataType, FunctionType,
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

ret = milvus_client.describe_collection(collection_name)
print(ret["functions"][0])

text_embedding_function.params["user"] = "user123"

milvus_client.alter_collection_function(collection_name, "openai", text_embedding_function)

ret = milvus_client.describe_collection(collection_name)
print(ret["functions"][0])

milvus_client.drop_collection_function(collection_name, "openai")

ret = milvus_client.describe_collection(collection_name)
print(ret["functions"])

text_embedding_function.params["user"] = "user1234"

milvus_client.add_collection_function(collection_name, text_embedding_function)

ret = milvus_client.describe_collection(collection_name)
print(ret["functions"][0])
