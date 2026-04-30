"""Demo: ARRAY_APPEND / ARRAY_REMOVE partial-update operators.

Run a local standalone Milvus (v2.7+ with FieldPartialUpdateOp support),
then:

    python examples/array_partial_update.py

Exercises:
  1. Create a collection with an Array<Int64> field (max_capacity=16).
  2. Insert two rows with initial tag arrays.
  3. ARRAY_APPEND new tags without resending the whole array.
  4. ARRAY_REMOVE selected tags (removes every matching occurrence).
  5. Query back and print the merged state.
"""

import numpy as np

from pymilvus import DataType, FieldOp, MilvusClient

URI = "http://localhost:19530"
COLLECTION = "array_partial_update_demo"
DIM = 4
MAX_CAPACITY = 16

fmt = "\n=== {:40} ===\n"


def main() -> None:
    client = MilvusClient(URI)

    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    schema = client.create_schema(enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(
        "tags",
        DataType.ARRAY,
        element_type=DataType.INT64,
        max_capacity=MAX_CAPACITY,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="L2")
    client.create_collection(
        COLLECTION, schema=schema, index_params=index_params, consistency_level="Strong"
    )

    rng = np.random.default_rng(seed=19530)
    rows = [
        {"id": 1, "vector": rng.random(DIM).tolist(), "tags": [1, 2]},
        {"id": 2, "vector": rng.random(DIM).tolist(), "tags": [10, 20, 30]},
    ]

    print(fmt.format("Insert seed rows"))
    client.insert(COLLECTION, rows)
    client.load_collection(COLLECTION)

    print("seed state:")
    for row in client.query(COLLECTION, filter="id >= 0", output_fields=["id", "tags"]):
        print(" ", row)

    # --------------------------------------------------------------
    # ARRAY_APPEND — add [3, 4] to row 1's tags; add [40] to row 2's
    # --------------------------------------------------------------
    print(fmt.format("ARRAY_APPEND tags"))
    client.upsert(
        COLLECTION,
        data=[
            {"id": 1, "vector": rng.random(DIM).tolist(), "tags": [3, 4]},
            {"id": 2, "vector": rng.random(DIM).tolist(), "tags": [40]},
        ],
        field_ops={"tags": FieldOp.array_append()},
    )
    for row in client.query(COLLECTION, filter="id >= 0", output_fields=["id", "tags"]):
        print(" ", row)
    # Row 1 -> [1, 2, 3, 4]; row 2 -> [10, 20, 30, 40]. Bandwidth and
    # concurrent-writer safety both improve vs. read-modify-write.

    # --------------------------------------------------------------
    # ARRAY_REMOVE — drop every occurrence of the payload values
    # --------------------------------------------------------------
    print(fmt.format("ARRAY_REMOVE tags"))
    client.upsert(
        COLLECTION,
        data=[
            {"id": 1, "vector": rng.random(DIM).tolist(), "tags": [2]},
            # Payload [999] matches nothing in row 2 -> no-op for that row.
            {"id": 2, "vector": rng.random(DIM).tolist(), "tags": [999]},
        ],
        field_ops={"tags": "array_remove"},
    )
    for row in client.query(COLLECTION, filter="id >= 0", output_fields=["id", "tags"]):
        print(" ", row)
    # Row 1 -> [1, 3, 4]; row 2 -> [10, 20, 30, 40] (unchanged).

    client.drop_collection(COLLECTION)


if __name__ == "__main__":
    main()
