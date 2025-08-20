from pymilvus import MilvusClient, DataType

milvus_host = "http://localhost:19530"
def test_createcollection():
    milvus_client = MilvusClient(uri=milvus_host)
    collection_name = "test_timestamptz_collection"

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)
        print(f"Dropped existing collection: {collection_name}")

    schema = milvus_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("timestamp", DataType.TIMESTAMPTZ)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=128)

    milvus_client.create_collection(collection_name, schema=schema)
    print(milvus_client.describe_collection(collection_name))
    hit = False
    actual_fields = milvus_client.describe_collection(collection_name)['fields']
    for field in actual_fields:
      if field['name'] == 'timestamp' and field['type'] == DataType.TIMESTAMPTZ:
        hit = True
        break
    assert hit, "TIMESTAMPTZ field was not created successfully."

    assert milvus_client.has_collection(collection_name)

    milvus_client.drop_collection(collection_name)

if __name__ == "__main__":
    test_createcollection()
    print("Test completed successfully.")