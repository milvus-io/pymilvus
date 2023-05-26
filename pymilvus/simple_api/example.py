from pprint import pprint
from pymilvus import (
    SimpleAPI,
)

fmt = "\n=== {:30} ===\n"
dim = 3
collection_name = "hello_milvus"
vector_field_name = "vector"
primary_key_name = "id"
api = SimpleAPI()
api.drop_collection(collection_name)

api.create_collection(
    collection_name=collection_name,
    dimension=dim,
    vector_field=vector_field_name,
    primary_key_name=primary_key_name,
    metric_type="L2",
    partition_field={"name": "a", "type": "int"},
    overwrite=True,
)

print("collections:", api.list_collections())

# print(f"{collection_name} :", api.describe_collection(collection_name))

test_data = [
    {"vector": [1, 2, 3], "a": 1, "b": 3},
    {"vector": [2, 3, 4], "a": 2, "b": 2.1},
    {"vector": [3, 4, 5], "a": 3, "c": -1},
    {"vector": [4, 5, 6], "a": 4, "d": {"m": 3}},
    {"vector": [7, 8, 9], "a": 5, "f": [3, 2, 1]},
    {"vector": [8, 9, 10], "a": 6, "g": "laq"},
    {"vector": [7, 10, 11], "a": 7, "z": -1},
]

print(fmt.format("Start inserting entities"))
pks = api.insert(collection_name, test_data, progress_bar=True)
print(fmt.format("Start searching based on vector similarity"))

print("len of pks:", len(pks), "first pk is :", pks[0])

print(f"get rows with `a` values that are 3 or 4 from {collection_name}")

values = api.fetch(collection_name, field_name="a", values=[3, 4], include_vectors=True)

print("values are:")
pprint(values)
print()

print(
    f"get rows where `b` < 3 from partiton `a` in [1,2,3] from {collection_name} but only the vector."
)

values = api.query(
    collection_name,
    filter_expression="b < 3",
    partition_keys=[1, 2, 3],
    output_fields=["vector"],
)

print("values are:")
pprint(values)
print()

print(f"search for [3,3,3] in {collection_name} and include the vector result.")

values = api.search(
    collection_name=collection_name, data=[3, 3, 3], include_vectors=True, top_k=1
)

print("values are:")
pprint(values)
print()

print(f"Delete vectors where b = 3 in partitions a in [1, 2, 3] from {collection_name}")

api.delete(
    collection_name=collection_name,
    field_name="a",
    values=[3],
    partition_keys=[1, 2, 3],
)
