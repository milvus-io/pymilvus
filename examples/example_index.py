from pymilvus import (
    connections,
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Index
)

# configure milvus hostname and port
print(f"\nCreate connection...")
connections.connect()

# List all collection names
print(f"\nList collections...")
print(list_collections())

# Create a collection named 'demo_film_tutorial'
print(f"\nCreate collection...")
field1 = FieldSchema(name="release_year", dtype=DataType.INT64, description="int64", is_primary=True)
field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector", dim=128, is_primary=False)
schema = CollectionSchema(fields=[field1, field2], description="collection description")
collection = Collection(name='demo_film_tutorial', data=None, schema=schema)

print(f"\nCreate index...")
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}}
index = Index(collection, "embedding", index_params)
print(index.params)

print([index.params for index in collection.indexes])

print(f"\nDrop index...")
index.drop()

print([index.params for index in collection.indexes])
collection.drop()
