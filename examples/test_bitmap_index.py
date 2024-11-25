# test_bitmap_index.py demonstrates how to create bitmap index and perform query
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create bitmap index
# 5. search
# 6. drop collection
import time

from pymilvus import (
    MilvusClient,
    utility,
    FieldSchema, CollectionSchema, Function, DataType, FunctionType,
    Collection,
)

collection_name = "text_bitmap_book"

milvus_client = MilvusClient("http://localhost:19530")

has_collection = milvus_client.has_collection(collection_name, timeout=5)
if has_collection:
    milvus_client.drop_collection(collection_name)

schema = milvus_client.create_schema()
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
schema.add_field("title", DataType.VARCHAR, max_length=200)
schema.add_field("type", DataType.VARCHAR, max_length=200)
schema.add_field("embeddings", DataType.FLOAT_VECTOR, dim=8)

index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="embeddings",
    index_name="vec_index",
    index_type="FLAT",
    metric_type="L2",
)
index_params.add_index(
    field_name="type",
    index_name="type_index",
    index_type="BITMAP",
)

ret = milvus_client.create_collection(collection_name, schema=schema, index_params=index_params, consistency_level="Strong")

rows = [
    {
        "id": 1,
        "title": "Lord of the Flies",
        "type": "novel",
        "embeddings": [0.64, 0.44, 0.13, 0.47, 0.74, 0.03, 0.32, 0.6],
    },
    {
        "id": 2,
        "title": "Chinese-English dictionary",
        "type": "reference",
        "embeddings": [0.9, 0.45, 0.18, 0.43, 0.4, 0.4, 0.7, 0.24],
    },
    {
        "id": 3,
        "title": "My Childhood",
        "type": "autobiographical", 
        "embeddings": [0.43, 0.57, 0.43, 0.88, 0.84, 0.69, 0.27, 0.98],
    },
]

insert_result = milvus_client.insert(collection_name, rows, progress_bar=True)

result = milvus_client.query(collection_name, filter='type in ["reference"]', output_fields=["type"], consistency_level="Strong")           
print(result)

# Finally, drop the collection
milvus_client.drop_collection(collection_name)
