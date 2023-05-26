from pymilvus import (
    connections,
    FieldSchema,
    Collection,
    CollectionSchema,
    DataType,
    SimpleAPI,
    utility,
)

connections.connect()

for x in utility.list_collections():
    utility.drop_collection(x)

z = SimpleAPI()

test_data = [
    {"vector": [1, 2, 3], "e": 3, "a": 1},
    {"vector": [2, 3, 4], "b": 2, "a": 2},
    {"vector": [3, 4, 5], "c": 1, "a": 3},
    {"vector": [4, 5, 6], "d": {"m": 3}, "a": 4},
    {"vector": [7, 8, 9], "f": [3, 2, 1], "a": 5},
    {"vector": [8, 9, 10], "g": 1.3, "a": 6},
    {"vector": [7, 10, 11], "z": -1, "a": 7},
]

# # # Test with AutoID

z.create_collection(
    "test",
    3,
    vector_field="vector",
    overwrite=True,
    partition_field={"name": "a", "type": "int", "default": 2},
    consistency_level="Strong",
    metric_type="L2",
)

z.insert("test", test_data)

# # Test List Collections
assert z.list_collections() == ["test"]
# # Test Num Entities
assert z.num_entities("test") == 7

# # Test search with vectors
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, include_vectors=True)[0][
        0
    ]["data"].keys()
) == {"id", "vector", "e", "a"}
# # # Test search without vectors
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, include_vectors=False)[0][
        0
    ]["data"].keys()
) == {"id", "e", "a"}
# # Test search with only vectors in output_field
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, output_fields=["vector"])[
        0
    ][0]["data"].keys()
) == {"id", "vector"}
# # # Teset search with only attribute in output
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, output_fields=["a"])[0][
        0
    ]["data"].keys()
) == {"id", "a"}
# # # Test search with partition keys with result
assert (
    z.search(collection_name="test", data=[2, 2, 2], partition_keys=[3], top_k=1)[0][0][
        "data"
    ]["a"]
    == 3
)
# # # Test search with partition keys with expression no result
assert z.search(
    collection_name="test",
    data=[2, 2, 2],
    filter_expression="e==3",
    partition_keys=[2],
    top_k=1,
) == [[]]


# # Query with vectors
assert set(
    z.query(collection_name="test", filter_expression="e==3", include_vectors=True)[
        0
    ].keys()
) == {"e", "vector", "id", "a"}
# # Query without vectors
assert set(
    z.query(collection_name="test", filter_expression="e==3", include_vectors=False)[
        0
    ].keys()
) == {"e", "id", "a"}
# # Query with only vector in output
assert set(
    z.query(collection_name="test", filter_expression="e==3", output_fields=["vector"])[
        0
    ].keys()
) == {"id", "vector"}
# # Query with only attribute in output
assert set(
    z.query(collection_name="test", filter_expression="e==3", output_fields=["a"])[
        0
    ].keys()
) == {"id", "a"}
# # Query with partition no result
assert (
    z.query(collection_name="test", filter_expression="e==3", partition_keys=[2]) == []
)
# # Query with partition with result
assert (
    z.query(collection_name="test", filter_expression="e==3", partition_keys=[1])[0][
        "e"
    ]
    == 3
)
# # Query with partition filter
assert len(z.query(collection_name="test", filter_expression="a<3")) == 2

# # Fetch with pk
assert {
    x["a"]
    for x in z.fetch(
        collection_name="test", field_name="a", values=[4, 3], include_vectors=True
    )
} == {3, 4}
# # Fetch with attribute
assert (
    z.fetch(collection_name="test", field_name="e", values=[3], include_vectors=True)[
        0
    ]["e"]
    == 3
)


# Delete with Attribute:
z.delete(collection_name="test", field_name="a", values=[1, 2])
assert (
    len(z.query(collection_name="test", filter_expression="a<4", include_vectors=False))
    == 1
)

# Delete with PK
vals = z.query(collection_name="test", filter_expression="a<5", include_vectors=False)
id = [x["id"] for x in vals]
z.delete(collection_name="test", field_name="id", values=id)
assert (
    len(z.query(collection_name="test", filter_expression="a<6", include_vectors=False))
    == 1
)

z.drop_collection("test")


# # # Test with auto_id off

z.create_collection(
    "test",
    3,
    vector_field="vector",
    primary_field="a",
    primary_type="int",
    primary_auto_id=False,
    overwrite=True,
    consistency_level="Strong",
    metric_type="L2",
)

z.insert("test", test_data)

# # Test List Collections
assert z.list_collections() == ["test"]
# # Test Num Entities
assert z.num_entities("test") == 7
# # Test search with vectors
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, include_vectors=True)[0][
        0
    ]["data"].keys()
) == {"vector", "e", "a"}
# # # Test search without vectors
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, include_vectors=False)[0][
        0
    ]["data"].keys()
) == {"e", "a"}
# # Test search with only vectors in output_field
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, output_fields=["vector"])[
        0
    ][0]["data"].keys()
) == {"a", "vector"}
# # # Teset search with only attribute in output
assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, output_fields=["a"])[0][
        0
    ]["data"].keys()
) == {"a"}

# # Query with vectors
assert set(
    z.query(collection_name="test", filter_expression="e==3", include_vectors=True)[
        0
    ].keys()
) == {"e", "vector", "a"}
# # Query without vectors
assert set(
    z.query(collection_name="test", filter_expression="e==3", include_vectors=False)[
        0
    ].keys()
) == {"e", "a"}
# # Query with only vector in output
assert set(
    z.query(collection_name="test", filter_expression="e==3", output_fields=["vector"])[
        0
    ].keys()
) == {"a", "vector"}
# # Query with only attribute in output
assert set(
    z.query(collection_name="test", filter_expression="e==3", output_fields=["a"])[
        0
    ].keys()
) == {"a"}

# # Fetch with pk
assert {
    x["a"]
    for x in z.fetch(
        collection_name="test", field_name="a", values=[4, 3], include_vectors=True
    )
} == {3, 4}
# # Fetch with attribute
assert (
    z.fetch(collection_name="test", field_name="e", values=[3], include_vectors=True)[
        0
    ]["e"]
    == 3
)

z.drop_collection("test")

z.close()

# # # Test with existing original

s = [
    FieldSchema("a", DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema("b", DataType.FLOAT_VECTOR, dim=3),
]
test_data_created = [
    {"b": [1, 2, 3], "a": 3, "v": 3},
    {"b": [2, 3, 4], "a": 2, "g": 3},
    {"b": [3, 4, 5], "a": 1, "h": "3"},
    {"b": [4, 5, 6], "a": 4, "p": 3},
]

schema = CollectionSchema(s, enable_dynamic_field=True, num_partitions=10)
c = Collection("test", schema, consistency_level="Session")
index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 400}}
c.create_index("b", index_params=index_params)
c.load()

z = SimpleAPI()

z.insert("test", test_data_created)

assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, include_vectors=True)[0][
        0
    ]["data"].keys()
) == {"a", "v", "b"}


z.drop_collection("test")

z.create_collection_from_schema(collection_name="test", schema=schema, metric_type="L2")

z.insert("test", test_data_created)

assert set(
    z.search(collection_name="test", data=[2, 2, 2], top_k=1, include_vectors=True)[0][
        0
    ]["data"].keys()
) == {"a", "v", "b"}

z.drop_collection("test")

z.close()
