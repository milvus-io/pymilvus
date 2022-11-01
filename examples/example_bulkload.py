import random
import json
import time
import os

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    BulkInsertState,
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. create some json files for do_bulk_insert operation
#   4. call do_bulk_insert
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
_COLLECTION_NAME = 'demo_bulk_insert'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_STR_FIELD_NAME = "str_field"

# String field parameter
_MAX_LENGTH = 65535

# Vector field parameter
_DIM = 8


# Create a Milvus connection
def create_connection():
    retry = True
    while retry:
        try:
            print(f"\nCreate connection...")
            connections.connect(host=_HOST, port=_PORT)
            retry = False
        except Exception as e:
            print("Cannot connect to Milvus. Error: " + str(e))
            print(f"Cannot connect to Milvus. Trying to connect Again. Sleeping for: 1")
            time.sleep(1)

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

# Create a partition
def create_partition(collection, partition_name):
    collection.create_partition(partition_name=partition_name)
    print("\nPartition created:", partition_name)
    return collection.partition(partition_name)

# Generate a json file with row-based data.
# The json file must contain a root key "rows", its value is a list, each row must contain a value of each field.
# No need to provide the auto-id field "id_field" since milvus will generate it.
# The row-based json file looks like:
# {"rows": [
# 	  {"str_field": "row-based_0", "float_vector_field": [0.190, 0.046, 0.143, 0.972, 0.592, 0.238, 0.266, 0.995]},
# 	  {"str_field": "row-based_1", "float_vector_field": [0.149, 0.586, 0.012, 0.673, 0.588, 0.917, 0.949, 0.944]},
#     ......
#   ]
# }
def gen_json_rowbased(num, path, tag):
    rows = []
    for i in range(num):
        rows.append({
            _STR_FIELD_NAME: tag + str(i),
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
# pending list, the do_bulk_insert() method will return error.
# Once a task is finished, the datanode become idle and will receive another task.
#
# The max size of each file is 1GB, if a file size is larger than 1GB, the task will failed and you will get error
# from the "failed_reason" of the task state.
#
# Then, how many segments generated? Let's say the collection's shard number is 2, typically each row-based file
# will be split into 2 segments. So, basically, each task generates segment count is equal to shard number.
# But if the segment.maxSize of milvus.yml is set to a small value, there could be shardNum*2, shardNum*3 segments
# generated, or even more.
def bulk_insert_rowbased(row_count_each_file, file_count, tag, partition_name = None):
    # make sure the data path is exist
    exist = os.path.exists(MILVUS_DATA_PATH)
    if not exist:
        os.mkdir(MILVUS_DATA_PATH)

    file_names = []
    for i in range(file_count):
        file_names.append("rows_" + str(i) + ".json")

    task_ids = []
    for filename in file_names:
        print("Generate row-based file:", MILVUS_DATA_PATH + filename)
        gen_json_rowbased(row_count_each_file, MILVUS_DATA_PATH + filename, tag)
        print("Import row-based file:", filename)
        task_id = utility.do_bulk_insert(collection_name=_COLLECTION_NAME,
                                     partition_name=partition_name,
                                     files=[filename])
        task_ids.append(task_id)
    return wait_tasks_persisted(task_ids)

# wait all bulk insert tasks to be a certain state
# return the states of all the tasks, including failed task
def wait_tasks_to_state(task_ids, state_code):
    wait_ids = task_ids
    states = []
    while True:
        time.sleep(2)
        temp_ids = []
        for id in wait_ids:
            state = utility.get_bulk_insert_state(task_id=id)
            if state.state == BulkInsertState.ImportFailed or state.state == BulkInsertState.ImportFailedAndCleaned:
                print(state)
                print("The task", state.task_id, "failed, reason:", state.failed_reason)
                continue

            if state.state >= state_code:
                states.append(state)
                continue

            temp_ids.append(id)

        wait_ids = temp_ids
        if len(wait_ids) == 0:
            break;
        print(len(wait_ids), "tasks not reach state:", BulkInsertState.state_2_name.get(state_code, "unknown"), ", next round check")

    return states


# Get bulk insert task state to check whether the data file has been parsed and persisted successfully.
# Persisted state doesn't mean the data is queryable, to query the data, you need to wait until the segment is
# loaded into memory.
def wait_tasks_persisted(task_ids):
    print("=========================================================================================================")
    states = wait_tasks_to_state(task_ids, BulkInsertState.ImportPersisted)
    persist_count = 0
    for state in states:
        if state.state == BulkInsertState.ImportPersisted or state.state == BulkInsertState.ImportCompleted:
            persist_count = persist_count + 1
        # print(state)
        # if you want to get the auto-generated primary keys, use state.ids to fetch
        # print("Auto-generated ids:", state.ids)

    print(persist_count, "of", len(task_ids), " tasks have successfully parsed all data files and data already persisted")
    print("=========================================================================================================\n")
    return states

# Get bulk insert task state to check whether the data file has been indexed successfully.
# If the state of bulk insert task is BulkInsertState.ImportCompleted, that means the data is queryable.
def wait_tasks_competed(task_ids):
    print("=========================================================================================================")
    states = wait_tasks_to_state(task_ids, BulkInsertState.ImportCompleted)
    complete_count = 0
    for state in states:
        if state.state == BulkInsertState.ImportCompleted:
            complete_count = complete_count + 1
        # print(state)
        # if you want to get the auto-generated primary keys, use state.ids to fetch
        # print("Auto-generated ids:", state.ids)

    print(complete_count, "of", len(task_ids), " tasks have successfully generated segments and these segments have been indexed, able to be compacted as normal")
    print("=========================================================================================================\n")
    return states

# List all bulk insert tasks, including pending tasks, working tasks and finished tasks.
# the parameter 'limit' is: how many latest tasks should be returned, if the limit<=0, all the tasks will be returned
def list_all_bulk_insert_tasks(limit):
    tasks = utility.list_bulk_insert_tasks(limit)
    print("=========================================================================================================")
    print("list bulk insert tasks with limit", limit)
    pending = 0
    started = 0
    persisted = 0
    completed = 0
    failed = 0
    for task in tasks:
        print(task)
        if task.state == BulkInsertState.ImportPending:
            pending = pending + 1
        elif task.state == BulkInsertState.ImportStarted:
            started = started + 1
        elif task.state == BulkInsertState.ImportPersisted:
            persisted = persisted + 1
        elif task.state == BulkInsertState.ImportCompleted:
            completed = completed + 1
        elif task.state == BulkInsertState.ImportFailed:
            failed = failed + 1
    print("There are", len(tasks), "bulk insert tasks.", pending, "pending,", started, "started,", persisted, "persisted,", completed, "completed,", failed, "failed")
    print("=========================================================================================================\n")

# Get collection row count.
# The collection.num_entities will trigger a flush() operation, flush data from buffer into storage, generate
# some new segments.
def get_entity_num(collection):
    print("=========================================================================================================")
    print("The number of entity:", collection.num_entities)


def create_index(collection):
    print("Start Creating index IVF_FLAT")
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index(_VECTOR_FIELD_NAME, index)


# Load collection data into memory. If collection is not loaded, the search() and query() methods will return error.
def load_collection(collection):
    collection.load()


# Release collection data to free memory.
def release_collection(collection):
    collection.release()


# ANN search
def search(collection, vector_field, search_vectors, partition_name = None, consistency_level = "Eventually"):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 10,
        "output_fields": [_STR_FIELD_NAME],
        "consistency_level": consistency_level,
    }
    if partition_name != None:
        search_param["partition_names"] = [partition_name]

    results = collection.search(**search_param)
    print("=========================================================================================================")
    for i, result in enumerate(results):
        if partition_name != None:
            print("Search result for {}th vector in partition '{}': ".format(i, partition_name))
        else:
            print("Search result for {}th vector: ".format(i))

        for j, res in enumerate(result):
            print(f"\ttop{j}: {res}, {_STR_FIELD_NAME}: {res.entity.get(_STR_FIELD_NAME)}")
        print("\thits count:", len(result))
    print("=========================================================================================================\n")

# delete entities
def delete(collection, ids):
    print("Delete these entities:", ids)
    expr = _ID_FIELD_NAME + " in " + str(ids)
    collection.delete(expr=expr)

# retrieve entities
def retrieve(collection, ids):
    print("Retrieve these entities:", ids)
    expr = _ID_FIELD_NAME + " in " + str(ids)
    result = collection.query(expr=expr, output_fields=[_VECTOR_FIELD_NAME])
    # the result is like [{'id_field': 0, 'float_vector_field': [...]}, {'id_field': 1, 'float_vector_field': [...]}]
    return result

def main():
    # create a connection
    create_connection()

    # drop collection if the collection exists
    if has_collection():
        drop_collection()

    # create collection
    collection = create_collection()

    # create a partition
    a_partition = "part_1"
    partition = create_partition(collection, a_partition)

    # specify an index type
    create_index(collection)


    # load data to memory
    load_collection(collection)

    # show collections
    list_collections()

    # do bulk_insert, wait all tasks finish persisting
    task_ids = []
    tasks = bulk_insert_rowbased(row_count_each_file=1000, file_count=1, tag="to_default_")
    for task in tasks:
        task_ids.append(task.task_id)
    tasks = bulk_insert_rowbased(row_count_each_file=1000, file_count=3, tag="to_partition_", partition_name=a_partition)
    for task in tasks:
        task_ids.append(task.task_id)

    # wai until all tasks completed(completed means queryable)
    wait_tasks_competed(task_ids)

    # list all tasks
    list_all_bulk_insert_tasks(len(task_ids))

    # get the number of entities
    get_entity_num(collection)

    # bulk insert task complete state doesn't mean the data can be searched, wait seconds to load the data
    print("wait 5 seconds to load the data")
    time.sleep(5)

    # search in entire collection
    vector = [round(random.random(), 6) for _ in range(_DIM)]
    vectors = [vector]
    print("Use a random vector to search in entire collection")
    search(collection, _VECTOR_FIELD_NAME, vectors)

    # search in a partition
    print("Use a random vector to search in partition:", a_partition)
    search(collection, _VECTOR_FIELD_NAME, vectors, partition_name=a_partition)

    # pick some entities to delete
    delete_ids = []
    for task in tasks:
        delete_ids.append(task.ids[5])
    id_vectors = retrieve(collection, delete_ids)
    delete(collection, delete_ids)

    # search the delete entities to check existence, check the top0 of the search result
    for id_vector in id_vectors:
        id = id_vector[_ID_FIELD_NAME]
        vector = id_vector[_VECTOR_FIELD_NAME]
        print("Search id:", id, ", compare this id to the top0 of search result, the entity with the id has been deleted")
        # here we use Stong consistency level to do search, because we need to make sure the delete operation is applied
        search(collection, _VECTOR_FIELD_NAME, [vector], partition_name=None, consistency_level="Strong")

    # release memory
    release_collection(collection)

    # drop collection
    drop_collection()


if __name__ == '__main__':
    main()