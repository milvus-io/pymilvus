import random

from pymilvus import (
    connections,
    list_collections,
    FieldSchema, CollectionSchema, DataType,
    Collection, Partition,
    utility
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
field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector", dim=8, is_primary=False)
schema = CollectionSchema(fields=[field1, field2], description="collection description")
collection = Collection(name='demo_film_tutorial', data=None, schema=schema)

# List all collection names
print(f"\nList collections...")
print(list_collections())

print(f"\nGet collection name, schema and description...")
print(collection.name)
print(collection.schema)
print(collection.description)

# List all partition names in demo collection
print(f"\nList partitions...")
print(collection.partitions)

# Create a partition named 'American'
print(f"\nCreate partition...")
partition_name = "American"
partition = Partition(collection, partition_name)
print(collection.partition(partition_name='American'))

# List all partition names in demo collection
print(f"\nList partitions...")
print(collection.partitions)

# Construct some entities
The_Lord_of_the_Rings = [
    {
        "id": 1,
        "title": "The_Fellowship_of_the_Ring",
        "release_year": 2001,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "id": 2,
        "title": "The_Two_Towers",
        "release_year": 2002,
        "embedding": [random.random() for _ in range(8)]
    },
    {
        "id": 3,
        "title": "The_Return_of_the_King",
        "release_year": 2003,
        "embedding": [random.random() for _ in range(8)]
    }
]

# Transform
ids = [k.get("id") for k in The_Lord_of_the_Rings]
release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

data = [release_years, embeddings]

# Insert into milvus
print(f"\nInsert data...")
partition.insert(data)

# Count entities
print(f"\nCount entities...")
print(collection.num_entities)

# TODO(wxyu): search

# Drop a partition
print(f"\nDrop partition...")
partition.drop()

# List all partition names in demo collection
print(f"\nList partitions...")
print(collection.partitions)

# List all collection names
print(f"\nList collections...")
print(list_collections())

# Drop a collection
print(f"\nDrop collection...")
collection.drop()

# List all collection names
print(f"\nList collections...")
print(list_collections())

# Calculate distance between vectors
vectors_l = [[random.random() for _ in range(64)] for _ in range(3)]
vectors_r = [[random.random() for _ in range(64)] for _ in range(5)]
op_l = {"float_vectors": vectors_l}
op_r = {"float_vectors": vectors_r}
params = {"metric_type": "L2", "sqrt": True}
results = utility.calc_distance(vectors_left=op_l, vectors_right=op_r, params=params)
for i in range(len(results)):
    print(results[i])
