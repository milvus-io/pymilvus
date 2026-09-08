"""Store integer and string arrays inside an array of structs.

Requires a Milvus server that supports array sub-fields in structs.
Run from the repository root: PYTHONPATH=. python examples/nested_array.py
"""

from pprint import pprint

from pymilvus import DataType, MilvusClient

URI = "http://localhost:19530"
COLLECTION_NAME = "nested_array_example"


def main() -> None:
    client = MilvusClient(URI)
    try:
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)

        schema = client.create_schema(auto_id=False)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)

        # Each Struct element contains Array<Int32> and Array<VarChar> fields.
        group_schema = client.create_struct_field_schema()
        group_schema.add_field(
            "values", DataType.ARRAY, element_type=DataType.INT32, max_capacity=4
        )
        group_schema.add_field(
            "labels",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=4,
            max_length=64,
        )
        schema.add_field(
            "groups",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=group_schema,
            max_capacity=8,  # Up to eight Struct elements per row.
            nullable=True,  # Nullability applies to the whole Struct array.
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

        # Describe returns the same logical field types used above.
        print("Describe collection:")
        pprint(client.describe_collection(COLLECTION_NAME), sort_dicts=False)

        rows = [
            {
                "pk": 0,
                "vector": [1.0, 0.0, 0.0, 0.0],
                "groups": [
                    {"values": [1, 2], "labels": ["alpha", "shared"]},
                    {"values": [-3, 4], "labels": ["beta", "shared"]},
                ],
            },
            {
                "pk": 1,
                "vector": [0.0, 1.0, 0.0, 0.0],
                "groups": None,  # A null Struct array.
            },
            {
                "pk": 2,
                "vector": [0.0, 0.0, 1.0, 0.0],
                "groups": [{"values": [], "labels": []}],  # Empty array sub-fields.
            },
            {
                "pk": 3,
                "vector": [0.0, 0.0, 0.0, 1.0],
                "groups": [],  # An empty Struct array.
            },
        ]
        result = client.insert(collection_name=COLLECTION_NAME, data=rows)
        print(f"Inserted {result['insert_count']} rows")

        print("Query inserted rows:")
        result = client.query(
            collection_name=COLLECTION_NAME,
            filter="pk in [0, 1, 2, 3]",
            output_fields=["pk", "groups"],
        )
        pprint(sorted(result, key=lambda row: row["pk"]), sort_dicts=False)

        # Match rows with at least one Struct element whose labels contain "alpha".
        print("Rows containing the label 'alpha' (pk=0):")
        result = client.query(
            collection_name=COLLECTION_NAME,
            filter='MATCH_ANY(groups, array_contains($[labels], "alpha"))',
            output_fields=["pk", "groups"],
        )
        pprint(result, sort_dicts=False)

        client.drop_collection(COLLECTION_NAME)
    finally:
        client.close()


if __name__ == "__main__":
    main()
