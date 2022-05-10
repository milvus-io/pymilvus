# Problem: https://github.com/milvus-io/milvus/issues/16835
# [Bug]: collection.search() timeout parameter not working
# 1. Connect to Milvus
# 2. Stop the Milvus Service
# 3. Execute collection.search(xxx, ..., timeout=1)
# 4. Wait
# The process hangs forever

# pymilvus==2.0.2           hang forever
# pymilvus==2.1.0.dev53     fail for not affected by timeout

from pymilvus import connections, Collection
from pymilvus import DataType, FieldSchema, CollectionSchema
from pymilvus import utility
import numpy as np
import time

collection_name = "_test_issue_16835"


def get_workable_collcetion(num_entites=1000, dim=8):
    schema = get_schema()
    print(f"get schema: {schema}")

    issue_16853 = Collection(collection_name, schema)

    rng = np.random.default_rng(seed=16835)
    entities = [
        [i for i in range(num_entites)],
        rng.random((num_entites, dim)),
    ]
    insert_result = issue_16853.insert(entities)
    issue_16853.load()

    print(f"collection {collection_name}: inserted {num_entites} entites and loaded")
    return issue_16853, insert_result


def get_schema():
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
    ]

    schema = CollectionSchema(fields)
    return schema


def main():
    connections.connect("default", host="localhost", port="19530")

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    issue_16853, insert_result = get_workable_collcetion()
    issue_16853.search(
        [[1., 2., 3., 4., 5., 6., 7., 8.]],
        "embeddings",
        {},
        limit=10, timeout=1)

    y = ""
    while y != "y":
        y = input("Please turn-off the server [y/N]: ")

    # should fail
    try:
        start = time.time()
        issue_16853.search(
            [[1., 2., 3., 4., 5., 6., 7., 8.]],
            "embeddings",
            {},
            limit=10, timeout=20)
    except BaseException:
        end = time.time()
        if end - start >= 10:
            print(f"=== test PASS: raise error by the timeout ===")
        else:
            print(f"=== test FAIL: Not influenced by the timeout ===")
        return

    print(f"=== test FAIL ===")


if __name__ == "__main__":
    main()
