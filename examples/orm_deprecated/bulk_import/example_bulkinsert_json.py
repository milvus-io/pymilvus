import random
import json
import time
import os

from minio import Minio
from minio.error import S3Error

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
#   3. create some json files for bulkinsert operation
#   4. call do_bulk_insert()
#   5. wait data to be consumed and indexed
#   6. search

# To run this example
# 1. start a standalone milvus(version >= v2.2.9) instance locally
#    make sure the docker-compose.yml has exposed the minio console:
#         minio:
#           ......
#           ports:
#             - "9000:9000"
#             - "9001:9001"
#           command: minio server /minio_data --console-address ":9001"
#
# 2. pip3 install minio

# Local path to generate JSON files
LOCAL_FILES_PATH = "/tmp/milvus_bulkinsert"

# Milvus service address
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulk_insert_json'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_JSON_FIELD_NAME = "json_field"
_VARCHAR_FIELD_NAME = "varchar_field"
_DYNAMIC_FIELD_NAME = "$meta"     # dynamic field, the internal name is "$meta", enable_dynamic_field=True

# minio
DEFAULT_BUCKET_NAME = "a-bucket"
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

# Vector field parameter
_DIM = 128

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
def gen_json_rowbased(num, path, partition_name):
    global id_start
    rows = []
    for i in range(num):
        rows.append({
            _ID_FIELD_NAME: id_start, # id field
            _JSON_FIELD_NAME: json.dumps({"Number": id_start, "Name": "book_"+str(id_start)}), # json field
            _VECTOR_FIELD_NAME: [round(random.random(), 6) for _ in range(_DIM)], # vector field
            _VARCHAR_FIELD_NAME: "{}_{}".format(partition_name, id_start) if partition_name is not None else "description_{}".format(id_start), # varchar field
            "dynamic_{}".format(id_start): id_start, # no field matches this value, this value will be put into dynamic field
        })
        id_start = id_start + 1

    data = {
        "rows": rows,
    }
    with open(path, "w") as json_file:
        json.dump(data, json_file)


# For row-based files, each file is converted to a task. Each time you can call do_bulk_insert() to insert one file.
# The rootcoord maintains a task list, each idle datanode will receive a task. If no datanode available, the task will
# be put into pending list to wait, the max size of pending list is 32. If new tasks count exceed spare quantity of
# pending list, the do_bulk_insert() method will return error.
# Once a task is finished, the datanode become idle and will receive another task.
#
# By default, the max size of each file is 16GB, this limit is configurable in the milvus.yaml (common.ImportMaxFileSize)
# If a file size is larger than 16GB, the task will fail and you will get error from the "failed_reason" of the task state.
#
# Then, how many segments generated? Let's say the collection's shard number is 2, typically each row-based file
# will be split into 2 segments. So, basically, each task generates segment count is equal to shard number.
# But if a file's data size exceed the segment.maxSize of milvus.yaml, there could be shardNum*2, shardNum*3 segments
# generated, or even more.
def bulk_insert_rowbased(row_count_per_file, file_count, partition_name = None):
    # make sure the files folder is created
    os.makedirs(name=LOCAL_FILES_PATH, exist_ok=True)

    task_ids = []
    for i in range(file_count):
        data_folder = os.path.join(LOCAL_FILES_PATH, "rows_{}".format(i))
        os.makedirs(name=data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, "rows_{}.json".format(i))
        print("Generate row-based file:", file_path)
        gen_json_rowbased(row_count_per_file, file_path, partition_name)

        ok, remote_files = upload(data_folder=data_folder)
        if ok:
            print("Import row-based file:", remote_files)
            task_id = utility.do_bulk_insert(collection_name=_COLLECTION_NAME,
                                         partition_name=partition_name,
                                         files=remote_files)
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
            break;
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

# Upload data files to minio
def upload(data_folder: str,
           bucket_name: str=DEFAULT_BUCKET_NAME)->(bool, list):
    if not os.path.exists(data_folder):
        print("Data path '{}' doesn't exist".format(data_folder))
        return False, []

    remote_files = []
    try:
        print("Prepare upload files")
        minio_client = Minio(endpoint=MINIO_ADDRESS, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        found = minio_client.bucket_exists(bucket_name)
        if not found:
            print("MinIO bucket '{}' doesn't exist".format(bucket_name))
            return False, []

        remote_data_path = "milvus_bulkinsert"
        def upload_files(folder:str):
            for parent, dirnames, filenames in os.walk(folder):
                if parent is folder:
                    for filename in filenames:
                        ext = os.path.splitext(filename)
                        if len(ext) != 2 or (ext[1] != ".json" and ext[1] != ".npy"):
                            continue
                        local_full_path = os.path.join(parent, filename)
                        minio_file_path = os.path.join(remote_data_path, os.path.basename(folder), filename)
                        minio_client.fput_object(bucket_name, minio_file_path, local_full_path)
                        print("Upload file '{}' to '{}'".format(local_full_path, minio_file_path))
                        remote_files.append(minio_file_path)
                    for dir in dirnames:
                        upload_files(os.path.join(parent, dir))

        upload_files(data_folder)

    except S3Error as e:
        print("Failed to connect MinIO server {}, error: {}".format(MINIO_ADDRESS, e))
        return False, []

    print("Successfully upload files: {}".format(remote_files))
    return True, remote_files

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


    # load data to memory
    load_collection(collection)

    # show collections
    list_collections()

    # do bulk_insert, wait all tasks finish persisting
    row_count_per_file = 100000
    if has_partition_key:
        # automatically partitioning
        bulk_insert_rowbased(row_count_per_file=row_count_per_file, file_count=2)
    else:
        # bulklinsert into default partition
        bulk_insert_rowbased(row_count_per_file=row_count_per_file, file_count=1)

        # create a partition, bulkinsert into the partition
        a_partition = "part_1"
        create_partition(collection, a_partition)
        bulk_insert_rowbased(row_count_per_file=row_count_per_file, file_count=1, partition_name=a_partition)

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
    search(collection, vector, expr="{} like \"description_33%\"".format(_VARCHAR_FIELD_NAME))

    # release memory
    release_collection(collection)

    # drop collection
    drop_collection()


if __name__ == '__main__':
    # change this value if you want to test bulkinert with partition key
    # Note: bulkinsert supports partition key from Milvus v2.2.12
    has_partition_key = False
    main(has_partition_key)
