from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

DIMENSION = 8
COLLECTION_NAME = "books"
connections.connect("default", host="localhost", port="19530")

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='type', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
collection = Collection(name=COLLECTION_NAME, schema=schema)

data_rows = [
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

collection.insert(data_rows)
collection.create_index(
    "embeddings", {"index_type": "FLAT", "metric_type": "L2"})

# create bitmap index for scalar fields.
collection.create_index(
    "type", {"index_type": "BITMAP"})

collection.load()

res = collection.query(expr='type in ["reference"]', output_fields=["id", "type"])
print(res)
