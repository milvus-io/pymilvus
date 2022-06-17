import random
import json
import time
import os
import numpy as np

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

# Milvus service address
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulkload'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_STR_FIELD_NAME = "str_field"

# String field parameter
_MAX_LENGTH = 65535

# Vector field parameter
_DIM = 8


# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


# Create a collection
def create_collection():
    field1 = FieldSchema(name=_ID_FIELD_NAME, dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
    field2 = FieldSchema(name=_VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=_STR_FIELD_NAME, dtype=DataType.VARCHAR, description="string",
                         max_length=_MAX_LENGTH, is_primary=False)
    schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
    collection = Collection(name=_COLLECTION_NAME, data=None, schema=schema)
    print("\nCollection created:", _COLLECTION_NAME)
    return collection


# Test existence of a collection
def has_collection():
    return utility.has_collection(_COLLECTION_NAME)


# Drop a collection in Milvus
def drop_collection():
    collection = Collection(_COLLECTION_NAME)
    collection.drop()
    print("\nDrop collection:", _COLLECTION_NAME)


# List all collections in Milvus
def list_collections():
    print("\nList collections:")
    print(utility.list_collections())


# Generate a json file with row-based data.
# The json file must contain a root key "rows", its value is a list, each row must contain a value of each field.
# No need to provide the auto-id field "id_field" since milvus will generate it.
# The row-based json file looks like this:
# {"rows": [
# 	  {"str_field": "row-based_0", "float_vector_field": [0.190, 0.046, 0.143, 0.972, 0.592, 0.238, 0.266, 0.995]},
# 	  {"str_field": "row-based_1", "float_vector_field": [0.149, 0.586, 0.012, 0.673, 0.588, 0.917, 0.949, 0.944]},
#     ......
#   ]
# }
def gen_json_rowbased(num, path):
    rows = []
    for i in range(num):
        rows.append({
            _STR_FIELD_NAME: "row-based_" + str(i),
            _VECTOR_FIELD_NAME: [round(random.random(), 6) for _ in range(_DIM)],
        })

    data = {
        "rows": rows,
    }
    with open(path, "w") as json_file:
        json.dump(data, json_file)


# Bulkload for row-based files, each file is converted to a task.
# The rootcoord maintains a task list, each idle datanode will receive a task. If no datanode available, the task will
# be put into pending list to wait, the max size of pending list is 32. If new tasks count exceed spare quantity of
# pending list, the bulkload() method will return error.
# Once a task is finished, the datanode become idle and will receive another task.
#
# The max size of each file is 1GB, if a file size is larger than 1GB, the task will failed and you will get error
# from the "failed_reason" of the task state.
#
# Then, how many segments generated? Let's say the collection's shard number is 2, typically each row-based file
# will be split into 2 segments. So, basically, each task generates segment count is equal to shard number.
# But if the segment.maxSize of milvus.yml is set to a small value, there could be shardNum*2, shardNum*3 segments
# generated, or even more.
def bulkload_rowbased(collection):
    # make sure the data path is exist
    exist = os.path.exists(MILVUS_DATA_PATH)
    if not exist:
        os.mkdir(MILVUS_DATA_PATH)

    file_names = ["rows_1.json", "rows_2.json", "rows_3.json"]
    for filename in file_names:
        gen_json_rowbased(100, MILVUS_DATA_PATH + filename)

    print("Bulkload row-based files:", file_names)
    task_ids = utility.bulk_load(collection_name=_COLLECTION_NAME, is_row_based=True, files=file_names)
    return wait_tasks_persisted(task_ids)


# Get bulkload task state to check whether the data file has been parsed and persisted successfully.
# Persisted state doesn't mean the data is queryable, to query the data, you need to wait until the segment is
# loaded into memory.
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
                # if you want to get the auto-generated primary keys, use state.ids to fetch
                print("Auto-generated ids:", state.ids)
            return states


# Generate a json file with column-based data.
# Each field has its field name as key and following by a list of values, all these lists length must be equal.
# No need to provide the auto-id field "id_field" since milvus will generate it.
# The column-based json file looks like this:
# {
# 	"str_field": ["column-based_0", "column-based_1", ......],
# 	"float_vector_field": [
#     [0.650735, 0.73155, 0.130244, 0.435798, 0.944411, 0.156331, 0.278817, 0.945728],
#     [0.251747, 0.069492, 0.868035, 0.740365, 0.117564, 0.60355, 0.309295, 0.274155],
#     ......
#   ]
# }
def gen_json_columnbased(num, path, gen_vectors=True):
    data = {}
    str_column = []
    for i in range(num):
        str_column.append("column-based_" + str(i))
    data[_STR_FIELD_NAME] = str_column

    if gen_vectors:
        vector_column = []
        for i in range(num):
            vector_column.append([round(random.random(), 6) for _ in range(_DIM)])
        data[_VECTOR_FIELD_NAME] = vector_column

    with open(path, "w") as json_file:
        json.dump(data, json_file)


# Bulkload for column-based files, each call to bulkload(is_row_based=False) is processed as a single task.
# The rootcoord maintains a task list, wait an idle datanode and send the task to it.
# If no datanode available, the task will be put into pending list to wait, the max size of pending list is 32.
# If new tasks count exceed spare quantity of pending list, the bulkload() method will return error.
#
# The max size of each file is 1GB, if a file size is larger than 1GB, the task will failed and you will get error
# from the "failed_reason" of the task state.
def bulkload_columnbased_json(collection):
    # make sure the data path is exist
    exist = os.path.exists(MILVUS_DATA_PATH)
    if not exist:
        os.mkdir(MILVUS_DATA_PATH)

    file_names = ["columns_1.json"]
    gen_json_columnbased(1000, MILVUS_DATA_PATH + file_names[0])

    print("Bulkload column-based files:", file_names)
    task_ids = utility.bulk_load(collection_name=_COLLECTION_NAME, is_row_based=False, files=file_names)
    return wait_tasks_persisted(task_ids)


# Generate a numpy binary file for vector field
def gen_numpy_vectors(num, path):
    arr = np.array([[random.random() for _ in range(_DIM)] for _ in range(num)])
    np.save(path, arr)


# Bulkload for column-based files, each call to bulkload(is_row_based=False) is processed as a single task.
# The rootcoord maintains a task list, wait an idle datanode and send the task to it.
# If no datanode available, the task will be put into pending list to wait, the max size of pending list is 32.
# If new tasks count exceed spare quantity of pending list, the bulkload() method will return error.
#
# The max size of each file is 1GB, if a file size is larger than 1GB, the task will failed and you will get error
# from the "failed_reason" of the task state.
#
# The bulkload() can support json/numpy format files for column-based data, here we use a numpy to store the vector
# field, use a json file to store the string field.
#
# Note: for numpy file, the file name must be equal to the field name. Milvus use the file name to mapping to a field
def bulkload_columnbased_numpy(collection):
    # make sure the data path is exist
    exist = os.path.exists(MILVUS_DATA_PATH)
    if not exist:
        os.mkdir(MILVUS_DATA_PATH)

    row_count = 1000
    file_names = ["str_field.json", _VECTOR_FIELD_NAME + ".npy"]
    gen_json_columnbased(row_count, MILVUS_DATA_PATH + file_names[0], False)
    gen_numpy_vectors(row_count, MILVUS_DATA_PATH + file_names[1])

    print("Bulkload column-based files:", file_names)
    task_ids = utility.bulk_load(collection_name=_COLLECTION_NAME, is_row_based=False, files=file_names)
    return wait_tasks_persisted(task_ids)


# Get bulkload task state to check whether the imported data is queryable.
# If collection.load() has been called, the imported data will be loaded into memory after a few seconds.
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


# List all bulkload tasks, including pending tasks, working tasks and finished tasks.
def list_all_bulkload_tasks():
    tasks = utility.list_bulk_load_tasks()
    print("=========================================================================================================\n")
    for task in tasks:
        print(task)
    print("Totally there are", len(tasks), "bulkload tasks are pending, working or processed")


# Get collection row count.
# The collection.num_entities will trigger a flush() operation, flush data from buffer into storage, generate
# some new segments.
def get_entity_num(collection):
    print("=========================================================================================================\n")
    print("The number of entity:", collection.num_entities)


# Load collection data into memory. If collection is not loaded, the search() and query() methods will return error.
def load_collection(collection):
    collection.load()


# Release collection data to free memory.
def release_collection(collection):
    collection.release()


# ANN search
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
    if has_collection():
        drop_collection()

    # create collection
    collection = create_collection()

    # load data to memory
    load_collection(collection)

    # show collections
    list_collections()

    # do bulkload
    tasks = bulkload_rowbased(collection)
    wait_tasks_queryable(tasks)
    tasks = bulkload_columnbased_json(collection)
    wait_tasks_queryable(tasks)
    tasks = bulkload_columnbased_numpy(collection)
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
    drop_collection()


if __name__ == '__main__':
    main()