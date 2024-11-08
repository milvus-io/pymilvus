import numpy as np
import random
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import logging

HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "test_iterator"
USER_ID = "id"
MAX_LENGTH = 65535
AGE = "age"
DEPOSIT = "deposit"
PICTURE = "picture"
CONSISTENCY_LEVEL = "Eventually"
LIMIT = 5
NUM_ENTITIES = 1000
DIM = 8
CLEAR_EXIST = True

# Create a logger for the main script
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # Set the log level to INFO

# Create a console handler and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter for the console output
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the handler
console_handler.setFormatter(formatter)

# Add the handler to the logger (this will apply globally)
log.addHandler(console_handler)

# Now, configure the root logger to apply to the entire app (including your package)
logging.getLogger().setLevel(logging.INFO)  # Set the root logger level to INFO
logging.getLogger().addHandler(console_handler)  # Attach the handler to the root logger


def re_create_collection(prepare_new_data: bool):
    if prepare_new_data:
        if utility.has_collection(COLLECTION_NAME) and CLEAR_EXIST:
            utility.drop_collection(COLLECTION_NAME)
            print(f"dropped existed collection{COLLECTION_NAME}")

        fields = [
            FieldSchema(name=USER_ID, dtype=DataType.VARCHAR, is_primary=True,
                        auto_id=False, max_length=MAX_LENGTH),
            FieldSchema(name=AGE, dtype=DataType.INT64),
            FieldSchema(name=DEPOSIT, dtype=DataType.DOUBLE),
            FieldSchema(name=PICTURE, dtype=DataType.FLOAT_VECTOR, dim=DIM)
        ]

        schema = CollectionSchema(fields)
        print(f"Create collection {COLLECTION_NAME}")
        collection = Collection(COLLECTION_NAME, schema, consistency_level=CONSISTENCY_LEVEL, num_shards=2)
    else:
        collection = Collection(COLLECTION_NAME)
    return collection


def random_pk(filter_set: set, lower_bound: int, upper_bound: int) -> str:
    ret: str = ""
    while True:
        candidate = str(random.randint(lower_bound, upper_bound))
        if candidate in filter_set:
            continue
        ret = candidate
        break
    return ret


def insert_data(collection):
    rng = np.random.default_rng(seed=19530)
    batch_count = 5
    filter_set: set = {}
    for i in range(batch_count):
        entities = [
            [random_pk(filter_set, 0, batch_count * NUM_ENTITIES) for _ in range(NUM_ENTITIES)],
            [int(ni % 100) for ni in range(NUM_ENTITIES)],
            [float(ni) for ni in range(NUM_ENTITIES)],
            rng.random((NUM_ENTITIES, DIM)),
        ]
        collection.insert(entities)
        collection.flush()
        print(f"Finish insert batch{i}, number of entities in Milvus: {collection.num_entities}")


def prepare_index(collection):
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    collection.create_index(PICTURE, index)
    print("Finish Creating index IVF_FLAT")
    collection.load()
    print("Finish Loading index IVF_FLAT")


def prepare_data(collection):
    insert_data(collection)
    prepare_index(collection)
    return collection


def query_iterate_collection_no_offset(collection):
    expr = f"10 <= {AGE} <= 25"

    query_iterator = collection.query_iterator(expr=expr, output_fields=[USER_ID, AGE],
                                               offset=0, batch_size=5, consistency_level=CONSISTENCY_LEVEL,
                                               reduce_stop_for_best="false", print_iterator_cursor=False,
                                               iterator_cp_file="/tmp/it_cp")
    no_best_ids: set = set({})
    page_idx = 0
    while True:
        res = query_iterator.next()
        if len(res) == 0:
            print("query iteration finished, close")
            query_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
            no_best_ids.add(res[i]['id'])
        page_idx += 1
        print(f"page{page_idx}-------------------------")

    print("best---------------------------")
    query_iterator = collection.query_iterator(expr=expr, output_fields=[USER_ID, AGE],
                                               offset=0, batch_size=5, consistency_level=CONSISTENCY_LEVEL,
                                                   reduce_stop_for_best="true", print_iterator_cursor=False, iterator_cp_file="/tmp/it_cp")
    
    best_ids: set = set({})
    page_idx = 0
    while True:
        res = query_iterator.next()
        if len(res) == 0:
            print("query iteration finished, close")
            query_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
            best_ids.add(res[i]['id'])
        page_idx += 1
        print(f"page{page_idx}-------------------------")

    diff = best_ids.difference(no_best_ids)
    for id in diff:
        print(f"diff id:{id}")


def query_iterate_collection_with_offset(collection):
    expr = f"10 <= {AGE} <= 14"
    query_iterator = collection.query_iterator(expr=expr, output_fields=[USER_ID, AGE],
                                               offset=10, batch_size=50, consistency_level=CONSISTENCY_LEVEL, print_iterator_cursor=True)
    page_idx = 0
    while True:
        res = query_iterator.next()
        if len(res) == 0:
            print("query iteration finished, close")
            query_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")


def query_iterate_collection_with_limit(collection):
    expr = f"10 <= {AGE} <= 44"
    query_iterator = collection.query_iterator(expr=expr, output_fields=[USER_ID, AGE],
                                               batch_size=80, limit=530, consistency_level=CONSISTENCY_LEVEL, print_iterator_cursor=True)
    page_idx = 0
    while True:
        res = query_iterator.next()
        if len(res) == 0:
            print("query iteration finished, close")
            query_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")


def search_iterator_collection(collection):
    SEARCH_NQ = 1
    DIM = 8
    rng = np.random.default_rng(seed=19530)
    vectors_to_search = rng.random((SEARCH_NQ, DIM))
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10, "radius": 1.0},
    }
    search_iterator = collection.search_iterator(vectors_to_search, PICTURE, search_params, batch_size=500,
                                                 output_fields=[USER_ID], print_iterator_cursor=True)
    page_idx = 0
    while True:
        res = search_iterator.next()
        if len(res) == 0:
            print("query iteration finished, close")
            search_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")


def search_iterator_collection_with_limit(collection):
    SEARCH_NQ = 1
    DIM = 8
    rng = np.random.default_rng(seed=19530)
    vectors_to_search = rng.random((SEARCH_NQ, DIM))
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10, "radius": 1.0},
    }
    search_iterator = collection.search_iterator(vectors_to_search, PICTURE, search_params, batch_size=200, limit=755,
                                                 output_fields=[USER_ID], print_iterator_cursor=True)
    page_idx = 0
    while True:
        res = search_iterator.next()
        if len(res) == 0:
            print("query iteration finished, close")
            search_iterator.close()
            break
        for i in range(len(res)):
            print(res[i])
        page_idx += 1
        print(f"page{page_idx}-------------------------")


def main():
    prepare_new_data = True
    connections.connect("default", host=HOST, port=PORT)
    collection = re_create_collection(prepare_new_data)
    if prepare_new_data:
        collection = prepare_data(collection)
    query_iterate_collection_no_offset(collection)
    query_iterate_collection_with_offset(collection)
    query_iterate_collection_with_limit(collection)
    search_iterator_collection(collection)
    search_iterator_collection_with_limit(collection)


if __name__ == '__main__':
    main()
