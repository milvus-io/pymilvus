import random
import json
import time
import os
import numpy as np

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
#   3. create some numpy files for bulkinsert operation
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

# Local path to generate Numpy files
LOCAL_FILES_PATH = "/tmp/milvus_bulkinsert/"

# Milvus service address
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulk_insert_npy'
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

# Generate a column-based data numpy file for each field.
# If primary key is not auto-id, bulkinsert requires a numpy file for the primary key field. For auto-id primary key, you don't need to provide this file.
# For dynamic field, provide a numpy file with name "$meta.npy" to store dynamic values. No need to provide this file if you have no dynamic value to import.
# For JSON type field, the numpy file must be a list of strings, each string is in JSON format. For example: ["{\"a\": 1}", "{\"b\": 2}", "{\"c\": 3}"]
# The dynamic field is also a JSON type field, the "$meta.npy" must be a list of strings in JSON format. For example: ["{}", "{\"x\": 100, \"y\": true}", "{\"z\": 3.14}"]
# The row count of each numpy file must be equal.
def gen_npy_columnbased(num: int, path: str):
    # make sure the files folder is created
    os.makedirs(name=path, exist_ok=True)

    global id_start
    id_array = [id_start + i for i in range(num)]
    vec_array = [[round(random.random(), 6) for _ in range(_DIM)] for _ in range(num)]
    json_array = [json.dumps({"Number": id_start + i, "Name": "book_"+str(id_start + i)}) for i in range(num)]
    varchar_array = ["description_{}".format(id_start + i) for i in range(num)]
    dynamic_array = [json.dumps({"dynamic_{}".format(id_start + i): True}) for i in range(num)]
    id_start = id_start + num

    file_list = []
    # numpy file for _ID_FIELD_NAME
    file_path = os.path.join(path, _ID_FIELD_NAME+".npy")
    file_list.append(file_path)
    np.save(file_path, id_array)
    print("Generate column-based numpy file:", file_path)

    # numpy file for _VECTOR_FIELD_NAME
    file_path = os.path.join(path, _VECTOR_FIELD_NAME+".npy")
    file_list.append(file_path)
    np.save(file_path, vec_array)
    print("Generate column-based numpy file:", file_path)

    # numpy file for _JSON_FIELD_NAME
    file_path = os.path.join(path, _JSON_FIELD_NAME+".npy")
    file_list.append(file_path)
    np.save(file_path, json_array)
    print("Generate column-based numpy file:", file_path)

    # numpy file for _VARCHAR_FIELD_NAME
    file_path = os.path.join(path, _VARCHAR_FIELD_NAME + ".npy")
    file_list.append(file_path)
    np.save(file_path, varchar_array)
    print("Generate column-based numpy file:", file_path)

    # numpy file for dynamic field
    file_path = os.path.join(path, _DYNAMIC_FIELD_NAME + ".npy")
    file_list.append(file_path)
    np.save(file_path, dynamic_array)
    print("Generate column-based numpy file:", file_path)

    return file_list

# Generate numpy files and upload files to minio, then call the do_bulk_insert()
def bulk_insert_columnbased(row_count_per_file, file_count, partition_name = None):
    # make sure the files folder is created
    os.makedirs(name=LOCAL_FILES_PATH, exist_ok=True)

    task_ids = []
    for i in range(file_count):
        data_folder = os.path.join(LOCAL_FILES_PATH, "columns_{}".format(i))
        os.makedirs(name=data_folder, exist_ok=True)
        file_list = gen_npy_columnbased(row_count_per_file, data_folder)

        ok, remote_files = upload(data_folder=data_folder)
        if ok:
            print("Import column-based files:", remote_files)
            task_id = utility.do_bulk_insert(collection_name=_COLLECTION_NAME,
                                             partition_name=partition_name,
                                             files=remote_files)
            task_ids.append(task_id)

    return wait_tasks_competed(task_ids)

# Wait all bulk insert tasks to be a certain state
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

# List all bulk insert tasks, including pending tasks, working tasks and finished tasks.
# the parameter 'limit' is: how many latest tasks should be returned, if the limit<=0, all the tasks will be returned
def list_all_bulk_insert_tasks(collection_name=_COLLECTION_NAME, limit=0):
    tasks = utility.list_bulk_insert_tasks(limit=limit, collection_name=collection_name)
    print("=========================================================================================================")
    print("List bulk insert tasks with limit", limit)
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
    bulk_insert_columnbased(row_count_per_file=100000, file_count=2)

    # list all tasks
    list_all_bulk_insert_tasks()

    # get the number of entities
    get_entity_num(collection)

    # print("Waiting index complete and refresh segments list to load...")
    utility.wait_for_index_building_complete(_COLLECTION_NAME)
    collection.load(_refresh = True)

    # pick some entities
    delete_ids = [50, 100]
    id_vectors = retrieve(collection, delete_ids)

    # search in entire collection
    for id_vector in id_vectors:
        id = id_vector[_ID_FIELD_NAME]
        vector = id_vector[_VECTOR_FIELD_NAME]
        print("Search id:", id, ", compare this id to the top0 of search result, they are equal")
        search(collection, vector)

    # delete the picked entities
    delete(collection, delete_ids)

    # search the deleted entities to check existence
    for id_vector in id_vectors:
        id = id_vector[_ID_FIELD_NAME]
        vector = id_vector[_VECTOR_FIELD_NAME]
        print("Search id:", id, ", compare this id to the top0 result, they are not equal since the id has been deleted")
        # here we use Stong consistency level to do search, because we need to make sure the delete operation is applied
        search(collection, vector, consistency_level="Strong")

    # search by filtering the varchar field
    vector = [round(random.random(), 6) for _ in range(_DIM)]
    search(collection, vector, expr="{} like \"description_8%\"".format(_VARCHAR_FIELD_NAME))

    # release memory
    release_collection(collection)

    # drop collection
    drop_collection()


if __name__ == '__main__':
    # change this value if you want to test bulkinert with partition key
    # Note: bulkinsert supports partition key from Milvus v2.2.12
    has_partition_key = False
    main(has_partition_key)
