import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import threading
import concurrent

fmt = "\n=== {:30} ==="
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

if utility.has_collection("hello_milvus"):
    utility.drop_collection("hello_milvus")
    print(f"Dropping existing collection hello_milvus")

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")

print(fmt.format("Create collection `hello_milvus`"))
hello_milvus = Collection("hello_milvus", schema, consistency_level="Strong")


class MilvusMultiThreadingInsert:
    def __init__(self, collection_name: str, number_of_batch: int):

        self.thread_local = threading.local()
        self.collection_name = collection_name
        self.batchs = [i for i in range(number_of_batch)]

    def get_thread_local_collection(self):
        if not hasattr(self.thread_local, "collection"):
            self.thread_local.collection = Collection(self.collection_name)
        return self.thread_local.collection

    def insert_data(self, number: int):
        print(fmt.format(f"No.{number:2}: Start inserting entities"))
        rng = np.random.default_rng(seed=number)
        entities = [
            [i for i in range(num_entities)],
            rng.random(num_entities).tolist(),
            rng.random((num_entities, dim)),
        ]

        insert_result = hello_milvus.insert(entities)
        assert len(insert_result.primary_keys) == num_entities
        print(fmt.format(f"No.{number:2}: Finish inserting entities"))

    def insert_all_batches(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            executor.map(self.insert_data, self.batchs)

    def run(self):
        start_time = time.time()
        self.insert_all_batches()
        duration = time.time() - start_time
        print(f'Inserted {len(self.batchs)} batches of {num_entities} entities in {duration} seconds')
        print(f"Expected num_entities: {len(self.batchs)*num_entities}. \
                Acutal num_entites: {self.get_thread_local_collection().num_entities}")


if __name__ == "__main__":
    multithreading_insert = MilvusMultiThreadingInsert("hello_milvus", 10)
    multithreading_insert.run()
