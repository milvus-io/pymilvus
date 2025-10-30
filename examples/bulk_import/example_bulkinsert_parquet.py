import random
import json
import time
import os
from typing import List

from minio import Minio
from minio.error import S3Error

import numpy as np
import pandas as pd

from pymilvus import (
    MilvusClient, DataType,
)

from pymilvus.bulk_writer import (
    bulk_import,
    get_import_progress,
)


# Local path to generate files
LOCAL_FILES_PATH = "/tmp/milvus_bulkinsert/"

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
    data["$meta"] = k_arr

    # write to Parquet file
    parquet_data = {}
    parquet_data[_ID_FIELD_NAME] = id_arr
    parquet_data[_BIN_VECTOR_FIELD_NAME] = bin_vector_arr
    parquet_data[_FLOAT_VECTOR_FIELD_NAME] = float_vector_arr
    parquet_data[_SPARSE_VECTOR_FIELD_NAME] = sparse_vector_arr
    parquet_data[_VARCHAR_FIELD_NAME] = varchar_arr
    parquet_data[_JSON_FIELD_NAME] = json_arr
    parquet_data[_INT16_FIELD_NAME] = np.array(int16_arr)
    parquet_data[_ARRAY_FIELD_NAME] = array_arr
    parquet_data[_GEOMETRY_FIELD_NAME] = geometry_arr
    parquet_data["$meta"] = k_arr

    data_frame = pd.DataFrame(data=parquet_data)
    data_frame.to_parquet(
        file_path, row_group_size=10000, engine="pyarrow"
    )  # don't use fastparquet

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
    def getValue(field_name, i):
        return f"{field_name}:{data[field_name][i] if len(data[field_name]) > 0 else None}"
    for i in indices:
        print(f"{getValue(_ID_FIELD_NAME, i)}, "
              f"{getValue(_BIN_VECTOR_FIELD_NAME, i)}, "
              f"{getValue(_FLOAT_VECTOR_FIELD_NAME, i)}, "
              f"{getValue(_SPARSE_VECTOR_FIELD_NAME, i)}, "
              f"{getValue(_VARCHAR_FIELD_NAME, i)}, "
              f"{getValue(_JSON_FIELD_NAME, i)}, "
              f"{getValue(_INT16_FIELD_NAME, i)}, "
              f"{getValue(_ARRAY_FIELD_NAME, i)}, "
              f"{getValue(_GEOMETRY_FIELD_NAME, i)}"
              )
    ids = [data[_ID_FIELD_NAME][k] for k in indices]
    results = client.query(collection_name=_COLLECTION_NAME,
                           filter=f"{_ID_FIELD_NAME} in {ids}",
                           output_fields=["*"])
    print("============= query data ==============")
    def getRes(result, field_name):
        if field_name == _BIN_VECTOR_FIELD_NAME:
            return f"{field_name}:{np.frombuffer(result[field_name][0], dtype=np.uint8)}"
        else:
            return f"{field_name}:{result[field_name]}"
    for res in results:
        print(f"{getRes(res, _ID_FIELD_NAME)}, "
              f"{getRes(res, _BIN_VECTOR_FIELD_NAME)}, "
              f"{getRes(res, _FLOAT_VECTOR_FIELD_NAME)}, "
              f"{getRes(res, _SPARSE_VECTOR_FIELD_NAME)}, "
              f"{getRes(res, _VARCHAR_FIELD_NAME)}, "
              f"{getRes(res, _JSON_FIELD_NAME)}, "
              f"{getRes(res, _INT16_FIELD_NAME)}, "
              f"{getRes(res, _ARRAY_FIELD_NAME)}, "
              f"{getRes(res, _GEOMETRY_FIELD_NAME)}, "
              )


if __name__ == '__main__':
    create_collection()

    local_file_path = LOCAL_FILES_PATH + "test.parquet"
    data = gen_parquet_file(1000, local_file_path)

    ok, remote_files = upload(local_file_path)
    if not ok:
        raise Exception("failed to upload file")

    call_bulkinsert(remote_files)

    verify(data)



