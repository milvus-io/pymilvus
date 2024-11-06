import random
import json
import csv
import time
import os

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    BulkInsertState,
    Function, FunctionType,
)


LOCAL_FILES_PATH = "/tmp/milvus_bulkinsert"

# Milvus service address
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulk_insert_csv'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_JSON_FIELD_NAME = "json_field"
_VARCHAR_FIELD_NAME = "varchar_field"
_DYNAMIC_FIELD_NAME = "$meta"     # dynamic field, the internal name is "$meta", enable_dynamic_field=True


# Vector field parameter
_DIM = 1536

# to generate increment ID
id_start = 1

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
def create_collection(has_partition_key: bool):
    field1 = FieldSchema(name=_ID_FIELD_NAME, dtype=DataType.INT64, description="int64", is_primary=True, auto_id=False)
    field2 = FieldSchema(name=_VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=_JSON_FIELD_NAME, dtype=DataType.JSON)
    # if has partition key, we use this varchar field as partition key field
    field4 = FieldSchema(name=_VARCHAR_FIELD_NAME, dtype=DataType.VARCHAR, max_length=256, is_partition_key=has_partition_key)
    schema = CollectionSchema(fields=[field1, field2, field3, field4], enable_dynamic_field=True)
    text_embedding_function = Function(
        name="openai",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=[_VARCHAR_FIELD_NAME],
        output_field_names=_VECTOR_FIELD_NAME,
        params={
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        }
    )
    schema.add_function(text_embedding_function)
    if has_partition_key:
        collection = Collection(name=_COLLECTION_NAME, schema=schema, num_partitions=10)
    else:
        collection = Collection(name=_COLLECTION_NAME, schema=schema)
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

def gen_csv_rowbased(num, path, partition_name, sep=","):
    global id_start
    header = [_ID_FIELD_NAME, _JSON_FIELD_NAME, _VARCHAR_FIELD_NAME, _DYNAMIC_FIELD_NAME]
    rows = []
    for i in range(num):
        rows.append([
            id_start, # id field
            json.dumps({"Number": id_start, "Name": "book_"+str(id_start)}), # json field
            "{}_{}".format(partition_name, id_start) if partition_name is not None else "description_{}".format(id_start), # varchar field
            json.dumps({"dynamic_field": id_start}), # no field matches this value, this value will be put into dynamic field
        ])
        id_start = id_start + 1
    data = [header] + rows
    with open(path, "w") as f:
        writer = csv.writer(f, delimiter=sep)
        for row in data:
            writer.writerow(row)


def bulk_insert_rowbased(row_count_per_file, file_count, partition_name = None):
    # make sure the files folder is created
    os.makedirs(name=LOCAL_FILES_PATH, exist_ok=True)

    task_ids = []
    for i in range(file_count):
        data_folder = os.path.join(LOCAL_FILES_PATH, "csv_{}".format(i))
        os.makedirs(name=data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, "csv_{}.csv".format(i))
        print("Generate csv file:", file_path)
        sep ="\t"
        gen_csv_rowbased(row_count_per_file, file_path, partition_name, sep)

        print("Import csv file:", file_path)
        task_id = utility.do_bulk_insert(collection_name=_COLLECTION_NAME,
                                         partition_name=partition_name,
                                         files=[file_path],
                                         sep=sep)
        task_ids.append(task_id)

    return wait_tasks_competed(task_ids)

# Wait all bulkinsert tasks to be a certain state
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
            break
        print("Wait {} tasks to be state: {}. Next round check".format(len(wait_ids), BulkInsertState.state_2_name.get(state_code, "unknown")))

    return states

# If the state of bulkinsert task is BulkInsertState.ImportCompleted, that means the data file has been parsed and data has been persisted,
# some segments have been created and waiting for index.
# ImportCompleted state doesn't mean the data is queryable, to query the data, you need to wait until the segment is
# indexed successfully and loaded into memory.
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

    print("{} of {} tasks have successfully generated segments, able to be compacted and indexed as normal".format(complete_count, len(task_ids)))
    print("=========================================================================================================\n")
    return states

# List all bulkinsert tasks, including pending tasks, working tasks and finished tasks.
# the parameter 'limit' is: how many latest tasks should be returned, if the limit<=0, all the tasks will be returned
def list_all_bulk_insert_tasks(collection_name=_COLLECTION_NAME, limit=0):
    tasks = utility.list_bulk_insert_tasks(limit=limit, collection_name=collection_name)
    print("=========================================================================================================")
    print("List bulkinsert tasks with limit", limit)
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
    print("There are {} bulkinsert tasks: {} pending, {} started, {} persisted, {} completed, {} failed"
          .format(len(tasks), pending, started, persisted, completed, failed))
    print("=========================================================================================================\n")

# Get collection row count.
def get_entity_num(collection):
    print("=========================================================================================================")
    print("The number of entity:", collection.num_entities)

# Specify an index type
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
def search(collection, search_vector, expr = None, consistency_level = "Eventually"):
    search_param = {
        "expr": expr,
        "data": [search_vector],
        "anns_field": _VECTOR_FIELD_NAME,
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 5,
        "output_fields": [_JSON_FIELD_NAME, _VARCHAR_FIELD_NAME, _DYNAMIC_FIELD_NAME],
        "consistency_level": consistency_level,
    }
    print("search..." if expr is None else "hybrid search...")
    results = collection.search(**search_param)
    print("=========================================================================================================")
    result = results[0]
    for j, res in enumerate(result):
        print(f"\ttop{j}: {res}")
    print("\thits count:", len(result))
    print("=========================================================================================================\n")

# Delete entities
def delete(collection, ids):
    print("=========================================================================================================\n")
    print("Delete these entities:", ids)
    expr = _ID_FIELD_NAME + " in " + str(ids)
    collection.delete(expr=expr)
    print("=========================================================================================================\n")

# Retrieve entities
def retrieve(collection, ids):
    print("=========================================================================================================")
    print("Retrieve these entities:", ids)
    expr = _ID_FIELD_NAME + " in " + str(ids)
    result = collection.query(expr=expr, output_fields=[_JSON_FIELD_NAME, _VARCHAR_FIELD_NAME, _VECTOR_FIELD_NAME, _DYNAMIC_FIELD_NAME])
    for item in result:
        print(item)
    print("=========================================================================================================\n")
    return result


def main(has_partition_key: bool):
    # create a connection
    create_connection()

    # drop collection if the collection exists
    if has_collection():
        drop_collection()

    # create collection
    collection = create_collection(has_partition_key)

    # specify an index type
    create_index(collection)

    print("Load data to memory")
    # load data to memory
    load_collection(collection)
    print("Load data to memory completed")
    # show collections
    print("Show collections")
    list_collections()

    # do bulk_insert, wait all tasks finish persisting
    row_count_per_file = 10
    if has_partition_key:
        # automatically partitioning
        bulk_insert_rowbased(row_count_per_file=row_count_per_file, file_count=2)
    else:
        # bulklinsert into default partition
        bulk_insert_rowbased(row_count_per_file=row_count_per_file, file_count=1)
    print("Bulk insert completed")

    # list all tasks
    list_all_bulk_insert_tasks()

    # get the number of entities
    get_entity_num(collection)

    print("Waiting index complete and refresh segments list to load...")
    utility.wait_for_index_building_complete(_COLLECTION_NAME)
    collection.load(_refresh = True)

    # pick some entities
    pick_ids = [50, row_count_per_file + 99]
    id_vectors = retrieve(collection, pick_ids)

    # search the picked entities, they are in result at the top0
    for id_vector in id_vectors:
        id = id_vector[_ID_FIELD_NAME]
        vector = id_vector[_VECTOR_FIELD_NAME]
        print("Search id:", id, ", compare this id to the top0 of search result, they are equal")
        search(collection, vector)

    # delete the picked entities
    delete(collection, pick_ids)

    # search the deleted entities, they are not in result anymore
    for id_vector in id_vectors:
        id = id_vector[_ID_FIELD_NAME]
        vector = id_vector[_VECTOR_FIELD_NAME]
        print("Search id:", id, ", compare this id to the top0 result, they are not equal since the id has been deleted")
        # here we use Strong consistency level to do search, because we need to make sure the delete operation is applied
        search(collection, vector, consistency_level="Strong")

    # search by filtering the varchar field
    vector = [round(random.random(), 6) for _ in range(_DIM)]
    ret = search(collection, vector)
    print(ret)
    # release memory
    # release_collection(collection)

    # drop collection
    # drop_collection()


if __name__ == '__main__':
    # change this value if you want to test bulkinert with partition key
    # Note: bulkinsert supports partition key from Milvus v2.2.12
    has_partition_key = False
    main(has_partition_key)
