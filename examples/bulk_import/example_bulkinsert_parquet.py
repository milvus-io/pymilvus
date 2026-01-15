import random
import json
import time
import os
from pathlib import Path
from typing import List

from minio import Minio
from minio.error import S3Error

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pymilvus import (
    MilvusClient, DataType,
)

from pymilvus.bulk_writer import (
    bulk_import,
    get_import_progress,
)


# Local path to generate files
LOCAL_FILES_PATH = "/tmp/milvus_bulkinsert/"
Path(LOCAL_FILES_PATH).mkdir(exist_ok=True)

# Milvus service address
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'demo_bulk_insert_parquet'
_ID_FIELD_NAME = 'id'
_BIN_VECTOR_FIELD_NAME = 'binary_vector'
_FLOAT_VECTOR_FIELD_NAME = 'float_vector'
_SPARSE_VECTOR_FIELD_NAME = 'sparse_vector'
_VARCHAR_FIELD_NAME = "varchar"
_JSON_FIELD_NAME = "json"
_INT16_FIELD_NAME = "int16"
_ARRAY_FIELD_NAME = "array"
_STRUCT_NAME = "struct_field"
_STRUCT_SUB_STR  = "struct_str"
_STRUCT_SUB_FLOAT = "struct_float_vec"
_CONCAT_STRUCT_SUB_FLOAT = "struct_field[struct_float_vec]"
_GEOMETRY_FIELD_NAME = "geometry"

# minio
DEFAULT_BUCKET_NAME = "a-bucket"
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"
REMOTE_DATA_PATH = "bulkinsert_data"

# Vector field parameter
_BIN_DIM = 64
_FLOAT_DIM = 4

client = MilvusClient(uri="http://localhost:19530")
print(client.get_server_version())


# Create a collection
def create_collection():
    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name=_ID_FIELD_NAME, datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name=_BIN_VECTOR_FIELD_NAME, datatype=DataType.BINARY_VECTOR, dim=_BIN_DIM)
    schema.add_field(field_name=_FLOAT_VECTOR_FIELD_NAME, datatype=DataType.FLOAT_VECTOR, dim=_FLOAT_DIM)
    schema.add_field(field_name=_SPARSE_VECTOR_FIELD_NAME, datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name=_VARCHAR_FIELD_NAME, datatype=DataType.VARCHAR, max_length=256, nullable=True)
    schema.add_field(field_name=_JSON_FIELD_NAME, datatype=DataType.JSON, nullable=True)
    schema.add_field(field_name=_INT16_FIELD_NAME, datatype=DataType.INT16, nullable=True)
    schema.add_field(field_name=_ARRAY_FIELD_NAME, datatype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=100, nullable=True)
    schema.add_field(field_name=_GEOMETRY_FIELD_NAME, datatype=DataType.GEOMETRY, nullable=True)

    struct_schema = MilvusClient.create_struct_field_schema()
    struct_schema.add_field("struct_str", DataType.VARCHAR, max_length=65535)
    struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=_FLOAT_DIM)
    schema.add_field(_STRUCT_NAME, datatype=DataType.ARRAY, element_type=DataType.STRUCT, struct_schema=struct_schema,
                     max_capacity=1000)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=_BIN_VECTOR_FIELD_NAME,
        index_type="BIN_FLAT",
        metric_type="HAMMING",
    )
    index_params.add_index(
        field_name=_FLOAT_VECTOR_FIELD_NAME,
        index_type="FLAT",
        metric_type="L2",
    )
    index_params.add_index(
        field_name=_SPARSE_VECTOR_FIELD_NAME,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
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


def gen_parquet_file(num, file_path):
    id_arr = []
    bin_vector_arr = []
    float_vector_arr = []
    sparse_vector_arr = []
    varchar_arr = []
    json_arr = []
    int16_arr = []
    array_arr = []
    struct_arr = []
    k_arr = []
    geometry_arr = []
    for i in range(num):
        # id field
        id_arr.append(np.dtype("int64").type(i))

        # binary vector field
        raw_vector = [random.randint(0, 1) for _ in range(_BIN_DIM)]
        uint8_arr = np.packbits(raw_vector, axis=-1).tolist()
        bin_vector_arr.append(np.array(uint8_arr, np.dtype("uint8")))

        # float vector field
        raw_vector = [random.random() for _ in range(_FLOAT_DIM)]
        float_vector_arr.append(np.array(raw_vector, np.dtype("float32")))

        # sparse vector field
        raw_vector = {}
        for k in range(i % 10 + 1):
            raw_vector[k] = random.random()
        sparse_vector_arr.append(json.dumps(raw_vector))

        # varchar field
        if i%2 == 0:
            varchar_arr.append(np.dtype("str").type(f"this is varchar {i}"))
            geometry_arr.append(np.dtype("str").type(f"POINT({i} {i+1})"))
        else:
            varchar_arr.append(None)
            geometry_arr.append(None)

        # json field
        if i % 3 != 0:
            obj = {"K": i}
            json_arr.append(np.dtype("str").type(json.dumps(obj)))
        else:
            json_arr.append(None)

        # int16 field
        if i % 5 != 0:
            int16_arr.append(np.dtype("int16").type(i))
        else:
            int16_arr.append(None)

        # array field
        if i % 5 != 0:
            arr = []
            for k in range(i % 5 + 1):
                arr.append(k)
            array_arr.append(np.array(arr, dtype=np.dtype("int64")))
        else:
            array_arr.append(None)

        # struct field - generate array of struct objects
        arr_len = random.randint(1, 5)  # Random number of struct elements
        struct_list = []
        for _ in range(arr_len):
            struct_obj = {
                _STRUCT_SUB_STR: f"struct_str_{i}_{random.randint(0, 100)}",
                _STRUCT_SUB_FLOAT: [random.random() for _ in range(_FLOAT_DIM)]
            }
            struct_list.append(struct_obj)
        # Store as Python list for Parquet native struct type
        struct_arr.append(struct_list)

        # dynamic field
        a = {"K": i} if i % 3 == 0 else {}
        k_arr.append(np.dtype("str").type(json.dumps(a)))

    data = {}
    data[_ID_FIELD_NAME] = id_arr
    data[_BIN_VECTOR_FIELD_NAME] = bin_vector_arr
    data[_FLOAT_VECTOR_FIELD_NAME] = float_vector_arr
    data[_SPARSE_VECTOR_FIELD_NAME] = sparse_vector_arr
    data[_VARCHAR_FIELD_NAME] = varchar_arr
    data[_JSON_FIELD_NAME] = json_arr
    data[_INT16_FIELD_NAME] = int16_arr
    data[_ARRAY_FIELD_NAME] = array_arr
    data[_GEOMETRY_FIELD_NAME] = geometry_arr
    data[_STRUCT_NAME] = struct_arr
    data["$meta"] = k_arr

    # write to Parquet file using PyArrow to preserve struct schema
    # Define PyArrow schema for struct field
    struct_type = pa.struct([
        pa.field(_STRUCT_SUB_STR, pa.string()),
        pa.field(_STRUCT_SUB_FLOAT, pa.list_(pa.float32()))
    ])

    # Build PyArrow arrays with explicit types
    pa_arrays = {
        _ID_FIELD_NAME: pa.array(id_arr, type=pa.int64()),
        _BIN_VECTOR_FIELD_NAME: pa.array([np.array(v, dtype=np.uint8) for v in bin_vector_arr],
                                         type=pa.list_(pa.uint8())),
        _FLOAT_VECTOR_FIELD_NAME: pa.array([np.array(v, dtype=np.float32) for v in float_vector_arr],
                                           type=pa.list_(pa.float32())),
        _SPARSE_VECTOR_FIELD_NAME: pa.array(sparse_vector_arr, type=pa.string()),
        _VARCHAR_FIELD_NAME: pa.array(varchar_arr, type=pa.string()),
        _JSON_FIELD_NAME: pa.array(json_arr, type=pa.string()),
        _INT16_FIELD_NAME: pa.array(int16_arr, type=pa.int16()),
        _ARRAY_FIELD_NAME: pa.array(array_arr, type=pa.list_(pa.int64())),
        _GEOMETRY_FIELD_NAME: pa.array(geometry_arr, type=pa.string()),
        _STRUCT_NAME: pa.array(struct_arr, type=pa.list_(struct_type)),
        "$meta": pa.array(k_arr, type=pa.string())
    }

    # Create PyArrow table and write to Parquet
    table = pa.table(pa_arrays)
    pq.write_table(table, file_path, row_group_size=10000)

    return data


# Upload data files to minio
def upload(local_file_path: str,
           bucket_name: str=DEFAULT_BUCKET_NAME)->(bool, list):
    if not os.path.exists(local_file_path):
        print(f"Local file '{local_file_path}' doesn't exist")
        return False, []

    ext = os.path.splitext(local_file_path)
    if len(ext) != 2 or ext[1] != ".parquet":
        print(f"Local file '{local_file_path}' is not parquet file")
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
    indices = [1, 10, 30]
    print("============= original data ==============")
    for i in indices:
        print(f"{_ID_FIELD_NAME}:{data[_ID_FIELD_NAME][i]}, "
              f"{_BIN_VECTOR_FIELD_NAME}:{data[_BIN_VECTOR_FIELD_NAME][i]}, "
              f"{_FLOAT_VECTOR_FIELD_NAME}:{data[_FLOAT_VECTOR_FIELD_NAME][i]}, "
              f"{_SPARSE_VECTOR_FIELD_NAME}:{data[_SPARSE_VECTOR_FIELD_NAME][i]}, "
              f"{_VARCHAR_FIELD_NAME}:{data[_VARCHAR_FIELD_NAME][i]}, "
              f"{_JSON_FIELD_NAME}:{data[_JSON_FIELD_NAME][i]}, "
              f"{_INT16_FIELD_NAME}:{data[_INT16_FIELD_NAME][i]}, "
              f"{_ARRAY_FIELD_NAME}:{data[_ARRAY_FIELD_NAME][i]}, "
              f"{_GEOMETRY_FIELD_NAME}:{data[_GEOMETRY_FIELD_NAME][i]}, "
              f"{_STRUCT_NAME}:{data[_STRUCT_NAME][i]}"
              )

    # Extract IDs from the data
    ids = [int(data[_ID_FIELD_NAME][k]) for k in indices]
    ids = [int(val) if isinstance(val, np.int64) else val for val in ids]
    results = client.query(collection_name=_COLLECTION_NAME,
                           filter=f"{_ID_FIELD_NAME} in {ids}",
                           output_fields=["*"])
    print("============= query data ==============")

    # Build a map from ID to result for easier comparison
    result_map = {res[_ID_FIELD_NAME]: res for res in results}

    # Verify each row
    all_match = True
    for i in indices:
        row_id = int(data[_ID_FIELD_NAME][i])

        if row_id not in result_map:
            print(f"❌ ID {row_id} not found in query results")
            all_match = False
            continue

        res = result_map[row_id]
        print(f"\n--- Verifying ID: {row_id} ---")

        # Compare each field
        mismatches = []

        # ID field
        if data[_ID_FIELD_NAME][i] != res[_ID_FIELD_NAME]:
            mismatches.append(f"ID mismatch: {data[_ID_FIELD_NAME][i]} != {res[_ID_FIELD_NAME]}")

        # Binary vector field
        orig_bin_vector = np.array(data[_BIN_VECTOR_FIELD_NAME][i], dtype=np.uint8)
        query_bin_vector = np.frombuffer(res[_BIN_VECTOR_FIELD_NAME][0], dtype=np.uint8)
        if not np.array_equal(orig_bin_vector, query_bin_vector):
            mismatches.append(f"Binary vector mismatch")

        # Float vector field
        orig_float_vector = np.array(data[_FLOAT_VECTOR_FIELD_NAME][i], dtype=np.float32)
        query_float_vector = res[_FLOAT_VECTOR_FIELD_NAME]
        if len(orig_float_vector) != len(query_float_vector):
            mismatches.append(f"Float vector length mismatch: {len(orig_float_vector)} != {len(query_float_vector)}")
        else:
            max_diff = max(abs(a - b) for a, b in zip(orig_float_vector, query_float_vector))
            if max_diff > 1e-5:
                mismatches.append(f"Float vector values differ (max diff: {max_diff})")

        # Sparse vector field
        orig_sparse = json.loads(data[_SPARSE_VECTOR_FIELD_NAME][i])
        query_sparse = res[_SPARSE_VECTOR_FIELD_NAME]
        # Normalize keys to integers for comparison (JSON keys are strings, query returns int keys)
        orig_sparse_normalized = {int(k): v for k, v in orig_sparse.items()}
        if set(orig_sparse_normalized.keys()) != set(query_sparse.keys()):
            mismatches.append(f"Sparse vector keys mismatch: {set(orig_sparse_normalized.keys())} != {set(query_sparse.keys())}")
        else:
            # Compare values with tolerance for floating point
            max_sparse_diff = max(abs(orig_sparse_normalized[k] - query_sparse[k]) for k in orig_sparse_normalized.keys())
            if max_sparse_diff > 1e-5:
                mismatches.append(f"Sparse vector values differ (max diff: {max_sparse_diff})")

        # VARCHAR field (nullable)
        orig_varchar = data[_VARCHAR_FIELD_NAME][i]
        query_varchar = res[_VARCHAR_FIELD_NAME]
        if orig_varchar != query_varchar:
            mismatches.append(f"VARCHAR mismatch: {orig_varchar} != {query_varchar}")

        # JSON field (nullable)
        orig_json = data[_JSON_FIELD_NAME][i]
        query_json = res[_JSON_FIELD_NAME]
        # Compare parsed JSON objects
        if orig_json is not None:
            orig_json_obj = json.loads(orig_json)
        else:
            orig_json_obj = None
        if orig_json_obj != query_json:
            mismatches.append(f"JSON mismatch: {orig_json_obj} != {query_json}")

        # INT16 field (nullable)
        orig_int16 = data[_INT16_FIELD_NAME][i]
        query_int16 = res[_INT16_FIELD_NAME]
        if orig_int16 != query_int16:
            mismatches.append(f"INT16 mismatch: {orig_int16} != {query_int16}")

        # Array field (nullable)
        orig_array = data[_ARRAY_FIELD_NAME][i]
        query_array = res[_ARRAY_FIELD_NAME]
        if orig_array is None and query_array is None:
            pass  # Both are None, OK
        elif orig_array is None or query_array is None:
            mismatches.append(f"Array mismatch: one is None, other is not")
        elif not np.array_equal(orig_array, query_array):
            mismatches.append(f"Array mismatch: {orig_array.tolist()} != {query_array}")

        # Struct field - compare as lists
        orig_struct = data[_STRUCT_NAME][i]
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
    create_collection()

    local_file_path = LOCAL_FILES_PATH + "test.parquet"
    data = gen_parquet_file(1000, local_file_path)

    ok, remote_files = upload(local_file_path)
    if not ok:
        raise Exception("failed to upload file")

    call_bulkinsert(remote_files)

    verify(data)



