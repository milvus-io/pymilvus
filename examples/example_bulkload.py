import random
import json
import time
import os

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    BulkLoadState,
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. create some json files for bulkload operation
#   4. do bulkload
#   5. search

# To run this example, start a standalone(local storage) milvus with the following configurations, in the milvus.yml:
# localStorage:
#   path: /tmp/milvus/data/
# rocksmq:
#   path: /tmp/milvus/rdb_data
# storageType: local
MILVUS_DATA_PATH = "/tmp/milvus/data/"

_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulkload'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_STR_FIELD_NAME = "str_field"
_MAX_LENGTH = 65535

# Vector parameters
_DIM = 8


# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# Create a collection named 'demo'
def create_collection(name, id_field, vector_field, str_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=str_field, dtype=DataType.VARCHAR, description="string",
                         max_length=_MAX_LENGTH, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=name, data=None, schema=schema)
    print("\ncollection created:", name)
    return collection


def has_collection(name):
    return utility.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())


def gen_json_rowbased(num, path):
    data = []
    for i in range(num):
        data.append({
            _STR_FIELD_NAME: "row-based_" + str(i),
            _VECTOR_FIELD_NAME: [round(random.random(), 6) for _ in range(_DIM)],
        })

    dict = {
        "rows": data,
    }
    with open(path, "w") as json_file:
        json.dump(dict, json_file)


def bulkload_row_based(collection):
    # make sure the data path is exist
    exist = os.path.exists(MILVUS_DATA_PATH)
    if not exist:
        os.mkdir(MILVUS_DATA_PATH)

    file_names = ["rows_1.json", "rows_2.json", "rows_3.json"]
    for filename in file_names:
        gen_json_rowbased(100, MILVUS_DATA_PATH + filename)

    # row-based files bulkload, each file converted to a task
    # the rootcoord maintains a task list, each idle datanode will receive a task
    # once a task is finished, the datanode become idle and will receive another task, until all tasks finished
    task_ids = utility.bulk_load(collection_name=_COLLECTION_NAME, is_row_based=True, files=file_names)
    return wait_tasks_persisted(task_ids)


def wait_tasks_persisted(task_ids):
    print("=========================================================================================================\n")
    while True:
        time.sleep(2)
        all_persisted = True
        persist_count = 0
        states = []
        for id in task_ids:
            state = utility.get_bulk_load_state(task_id=id)
            if state.state == BulkLoadState.ImportFailed:
                print(state)
                print("The task", state.task_id, "failed, reason:", state.failed_reason)
                return states

            states.append(state)
            if state.state != BulkLoadState.ImportPersisted:
                all_persisted = False
            else:
                persist_count = persist_count + 1

        print(persist_count, "of", len(task_ids), " files has been parsed and persisted")
        if all_persisted:
            print(len(task_ids), " files has been parsed and persisted")
            for state in states:
                print(state)
            return states


def gen_json_columnbased(num, path):
    str_column = []
    vector_column = []
    for i in range(num):
        str_column.append("column-based_" + str(i))
        vector_column.append([round(random.random(), 6) for _ in range(_DIM)])

    dict = {
        _STR_FIELD_NAME: str_column,
        _VECTOR_FIELD_NAME: vector_column,
    }
    with open(path, "w") as json_file:
        json.dump(dict, json_file)


def bulkload_column_based(collection):
    # make sure the data path is exist
    exist = os.path.exists(MILVUS_DATA_PATH)
    if not exist:
        os.mkdir(MILVUS_DATA_PATH)

    file_names = ["columns_1.json"]
    gen_json_columnbased(1000, MILVUS_DATA_PATH + file_names[0])

    # column-based files bulkload, each file converted to a task
    # the rootcoord maintains a task list, each idle datanode will receive a task
    # once a task is finished, the datanode become idle and will receive another task, until all tasks finished
    task_ids = utility.bulk_load(collection_name=_COLLECTION_NAME, is_row_based=False, files=file_names)
    return wait_tasks_persisted(task_ids)


def wait_tasks_queryable(tasks):
    while True:
        time.sleep(1)
        queryable_count = 0
        for task in tasks:
            if task.data_queryable:
                queryable_count = queryable_count + 1
            print(queryable_count, "of", len(tasks), "tasks data is queryable")
        if queryable_count == len(tasks):
            print("All data imported by bulkload is queryable!")
            break


def list_all_bulkload_tasks():
    tasks = utility.list_bulk_load_tasks()
    print("=========================================================================================================\n")
    for task in tasks:
        print(task)
    print("Totally there are", len(tasks), "bulkload tasks are pending or processed")


def get_entity_num(collection):
    print("=========================================================================================================\n")
    print("The number of entity:", collection.num_entities)


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()


def search(collection, vector_field, id_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": "L2", "params": {}},
        "limit": 10,
        "output_fields": [_STR_FIELD_NAME],
    }
    results = collection.search(**search_param)
    print("=========================================================================================================\n")
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print(f"\ttop{j}: {res}, {_STR_FIELD_NAME}: {res.entity.get(_STR_FIELD_NAME)}")
        print("\thits count:", len(result))


def main():
    # create a connection
    create_connection()

    # drop collection if the collection exists
    if has_collection(_COLLECTION_NAME):
        drop_collection(_COLLECTION_NAME)

    # create collection
    collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME, _STR_FIELD_NAME)

    # load data to memory
    load_collection(collection)

    # show collections
    list_collections()

    # do bulkload
    tasks = bulkload_row_based(collection)
    wait_tasks_queryable(tasks)
    tasks = bulkload_column_based(collection)
    wait_tasks_queryable(tasks)
    list_all_bulkload_tasks()

    # get the number of entities
    get_entity_num(collection)

    # search
    vector = [round(random.random(), 6) for _ in range(_DIM)]
    vectors = [vector]
    search(collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, vectors)

    # release memory
    release_collection(collection)

    # drop collection
    drop_collection(_COLLECTION_NAME)


if __name__ == '__main__':
    main()