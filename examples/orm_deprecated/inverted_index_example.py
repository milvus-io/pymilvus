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
    FieldSchema(name='release_year', dtype=DataType.INT64),
    FieldSchema(name='embeddings', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
]
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
collection = Collection(name=COLLECTION_NAME, schema=schema)

data_rows = [
    {
        "id": 1,
        "title": "Lord of the Flies",
        "release_year": 1954,
        "embeddings": [0.64, 0.44, 0.13, 0.47, 0.74, 0.03, 0.32, 0.6],
    },
    {
        "id": 2,
        "title": "The Great Gatsby",
        "release_year": 1925,
        "embeddings": [0.9, 0.45, 0.18, 0.43, 0.4, 0.4, 0.7, 0.24],
    },
    {
        "id": 3,
        "title": "The Catcher in the Rye",
        "release_year": 1951,
        "embeddings": [0.43, 0.57, 0.43, 0.88, 0.84, 0.69, 0.27, 0.98],
    },
]

collection.insert(data_rows)
collection.create_index(
    "embeddings", {"index_type": "FLAT", "metric_type": "L2"})

# create inverted index for scalar fields.
collection.create_index(
    "release_year", {"index_type": "INVERTED"})
collection.create_index(
    "title", {"index_type": "INVERTED"})

collection.load()

res = collection.query(expr='release_year <= 1951', output_fields=["id"])
print(res)
res = collection.query(
    expr='release_year in [1951, 1954] and title like "The%"', output_fields=["id", "title"])
print(res)
