import random
import json
import time
import os
from typing import List

from minio import Minio
from minio.error import S3Error

from pymilvus import (
    DataType,
    utility,
    BulkInsertState,
    MilvusClient,
)

from pymilvus.bulk_writer import (
    bulk_import,
    get_import_progress,
)

# Local path to generate JSON files
LOCAL_FILES_PATH = "/tmp/milvus_bulkinsert"

# Milvus service address
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulk_insert_json'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'
_VARCHAR_FIELD_NAME = "varchar_field"

_STRUCT_NAME = "struct_field"
_STRUCT_SUB_STR  = "struct_str"
_STRUCT_SUB_FLOAT = "struct_float_vec"
_CONCAT_STRUCT_SUB_FLOAT = "struct_field[struct_float_vec]"

# minio
DEFAULT_BUCKET_NAME = "a-bucket"
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"
REMOTE_DATA_PATH = "bulkinsert_data"

# Vector field parameter
_DIM = 128

# to generate increment ID
id_start = 1

client = MilvusClient(uri="http://localhost:19530")
print(client.get_server_version())


# Create a collection
def create_collection():
    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name=_ID_FIELD_NAME, datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name=_VECTOR_FIELD_NAME, datatype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM)
    schema.add_field(field_name=_VARCHAR_FIELD_NAME, datatype=DataType.VARCHAR, max_length=256)
    struct_schema = MilvusClient.create_struct_field_schema()
    struct_schema.add_field("struct_str", DataType.VARCHAR, max_length=65535)
    struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=_DIM)
    schema.add_field(_STRUCT_NAME, datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema,
                     max_capacity=1000)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=_VECTOR_FIELD_NAME,
        index_type="IVF_FLAT",
        metric_type="L2",
    )
    index_params.add_index(
        field_name=_CONCAT_STRUCT_SUB_FLOAT,
        index_type="HNSW",
        metric_type="MAX_SIM",
    )
    client.drop_collection(collection_name=_COLLECTION_NAME)
    client.create_collection(
        collection_name=_COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(_COLLECTION_NAME, "created")


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
        # struct field - generate array of struct objects
        arr_len = random.randint(1, 5)  # Random number of struct elements
        struct_list = []
        for _ in range(arr_len):
            struct_obj = {
                _STRUCT_SUB_STR: f"struct_str_{i}_{random.randint(0, 100)}",
                _STRUCT_SUB_FLOAT: [random.random() for _ in range(_DIM)]
            }
            struct_list.append(struct_obj)

        rows.append({
            _ID_FIELD_NAME: id_start, # id field
            _VECTOR_FIELD_NAME: [round(random.random(), 6) for _ in range(_DIM)], # vector field
            _VARCHAR_FIELD_NAME: "{}_{}".format(partition_name, id_start) if partition_name is not None else "description_{}".format(id_start), # varchar field
            _STRUCT_NAME: struct_list, # struct field
            "dynamic_{}".format(id_start): id_start, # no field matches this value, this value will be put into dynamic field
        })
        id_start = id_start + 1

    data = {
        "rows": rows,
    }
    with open(path, "w") as json_file:
        json.dump(data, json_file)
    return data


def bulk_insert_rowbased(row_count_per_file, file_id, partition_name = None):
    os.makedirs(name=LOCAL_FILES_PATH, exist_ok=True)
    data_folder = os.path.join(LOCAL_FILES_PATH, "rows_{}".format(file_id))
    os.makedirs(name=data_folder, exist_ok=True)
    file_path = os.path.join(data_folder, "rows_{}.json".format(file_id))
    print("Generate row-based file:", file_path)
    data = gen_json_rowbased(row_count_per_file, file_path, partition_name)

    ok, remote_files = upload(file_path)
    if not ok:
        raise Exception("failed to upload file")

    call_bulkinsert(remote_files)

    return data

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

# Upload data files to minio
def upload(local_file_path: str,
           bucket_name: str=DEFAULT_BUCKET_NAME)->(bool, list):
    if not os.path.exists(local_file_path):
        print(f"Local file '{local_file_path}' doesn't exist")
        return False, []

    remote_files = []
    try:
        print("Prepare upload files")
        minio_client = Minio(endpoint=MINIO_ADDRESS, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        found = minio_client.bucket_exists(bucket_name)
        if not found:
            print("MinIO bucket '{}' doesn't exist".format(bucket_name))
            return False, []

        filename = os.path.basename(local_file_path)
        minio_file_path = os.path.join(REMOTE_DATA_PATH, filename)
        minio_client.fput_object(bucket_name, minio_file_path, local_file_path)
        print(f"Upload file '{local_file_path}' to '{minio_file_path}'")
        remote_files.append([minio_file_path])


    except S3Error as e:
        print(f"Failed to connect MinIO server {MINIO_ADDRESS}, error: {e}")
        return False, []

    print(f"Successfully upload files: {remote_files}")
    return True, remote_files

def call_bulkinsert(batch_files: List[List[str]]):
    url = f"http://{_HOST}:{_PORT}"
    print(f"\n===================== Import files to milvus ====================")
    resp = bulk_import(
        url=url,
        collection_name=_COLLECTION_NAME,
        files=batch_files,
    )
    print(resp.json())
    job_id = resp.json()['data']['jobId']
    print(f"Create a bulkinsert job, job id: {job_id}")

    while True:
        print("Wait 5 second to check bulkinsert job state...")
        time.sleep(5)

        print(f"\n===================== Get import job progress ====================")
        resp = get_import_progress(
            url=url,
            job_id=job_id,
        )

        state = resp.json()['data']['state']
        progress = resp.json()['data']['progress']
        if state == "Importing":
            print(f"The job {job_id} is importing... {progress}%")
            continue
        if state == "Failed":
            reason = resp.json()['data']['reason']
            print(f"The job {job_id} failed, reason: {reason}")
            break
        if state == "Completed" and progress == 100:
            print(f"The job {job_id} completed")
            break

    client.refresh_load(collection_name=_COLLECTION_NAME)
    print(f"Import done")

def verify(data):
    # For JSON row-based format, data structure is: {"rows": [row1, row2, ...]}
    rows = data["rows"]
    indices = [1, 10, 30]
    print("============= original data ==============")
    for i in indices:
        if i >= len(rows):
            print(f"Index {i} out of range (total rows: {len(rows)})")
            continue
        row = rows[i]
        # Extract dynamic field - it has a variable name pattern "dynamic_{id}"
        dynamic_value = None
        for key in row.keys():
            if key.startswith("dynamic_"):
                dynamic_value = row[key]
                break
        print(f"{_ID_FIELD_NAME}:{row.get(_ID_FIELD_NAME)}, "
              f"{_VECTOR_FIELD_NAME}:{row.get(_VECTOR_FIELD_NAME)}, "
              f"{_VARCHAR_FIELD_NAME}:{row.get(_VARCHAR_FIELD_NAME)}, "
              f"dynamic:{dynamic_value}, "
              f"{_STRUCT_NAME}:{row.get(_STRUCT_NAME)}"
              )

    # Extract IDs from the rows
    ids = [rows[i][_ID_FIELD_NAME] for i in indices if i < len(rows)]
    results = client.query(collection_name=_COLLECTION_NAME,
                           filter=f"{_ID_FIELD_NAME} in {ids}",
                           output_fields=["*"])
    print("============= query data ==============")

    # Build a map from ID to result for easier comparison
    result_map = {res[_ID_FIELD_NAME]: res for res in results}

    # Verify each row
    all_match = True
    for i in indices:
        if i >= len(rows):
            continue

        row = rows[i]
        row_id = row[_ID_FIELD_NAME]

        if row_id not in result_map:
            print(f"❌ ID {row_id} not found in query results")
            all_match = False
            continue

        res = result_map[row_id]
        print(f"\n--- Verifying ID: {row_id} ---")

        # Compare each field
        mismatches = []

        # ID field
        if row[_ID_FIELD_NAME] != res[_ID_FIELD_NAME]:
            mismatches.append(f"ID mismatch: {row[_ID_FIELD_NAME]} != {res[_ID_FIELD_NAME]}")

        # Vector field - compare with tolerance for floating point
        orig_vector = row[_VECTOR_FIELD_NAME]
        query_vector = res[_VECTOR_FIELD_NAME]
        if len(orig_vector) != len(query_vector):
            mismatches.append(f"Vector length mismatch: {len(orig_vector)} != {len(query_vector)}")
        else:
            max_diff = max(abs(a - b) for a, b in zip(orig_vector, query_vector))
            if max_diff > 1e-5:
                mismatches.append(f"Vector values differ (max diff: {max_diff})")

        # VARCHAR field
        if row[_VARCHAR_FIELD_NAME] != res[_VARCHAR_FIELD_NAME]:
            mismatches.append(f"VARCHAR mismatch: {row[_VARCHAR_FIELD_NAME]} != {res[_VARCHAR_FIELD_NAME]}")

        # Struct field - compare as lists
        orig_struct = row[_STRUCT_NAME]
        query_struct = res[_STRUCT_NAME]
        if len(orig_struct) != len(query_struct):
            mismatches.append(f"Struct array length mismatch: {len(orig_struct)} != {len(query_struct)}")
        else:
            for idx, (orig_item, query_item) in enumerate(zip(orig_struct, query_struct)):
                if orig_item[_STRUCT_SUB_STR] != query_item[_STRUCT_SUB_STR]:
                    mismatches.append(f"Struct[{idx}].{_STRUCT_SUB_STR} mismatch: {orig_item[_STRUCT_SUB_STR]} != {query_item[_STRUCT_SUB_STR]}")

                # Compare struct float vectors with tolerance
                orig_vec = orig_item[_STRUCT_SUB_FLOAT]
                query_vec = query_item[_STRUCT_SUB_FLOAT]
                if len(orig_vec) != len(query_vec):
                    mismatches.append(f"Struct[{idx}].{_STRUCT_SUB_FLOAT} length mismatch")
                else:
                    max_diff = max(abs(a - b) for a, b in zip(orig_vec, query_vec))
                    if max_diff > 1e-5:
                        mismatches.append(f"Struct[{idx}].{_STRUCT_SUB_FLOAT} values differ (max diff: {max_diff})")

        if mismatches:
            all_match = False
            print(f"❌ Verification failed for ID {row_id}:")
            for mismatch in mismatches:
                print(f"   - {mismatch}")
        else:
            print(f"✓ ID {row_id} matches perfectly")

    print("\n============= Verification Summary ==============")
    if all_match:
        print("✓ All data verified successfully!")
    else:
        print("❌ Some data mismatches found")

    return all_match

if __name__ == '__main__':
    # create a connection
    create_collection()

    # do bulk_insert, wait all tasks finish persisting
    row_count_per_file = 1000
    data = bulk_insert_rowbased(row_count_per_file, 1)
    verify(data)
