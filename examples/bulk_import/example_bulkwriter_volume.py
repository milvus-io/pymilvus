# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

import json
import logging
import time
import numpy as np

from examples.bulk_import.data_gengerator import *
from pymilvus.bulk_writer.volume_bulk_writer import VolumeBulkWriter
from pymilvus.orm import utility

logging.basicConfig(level=logging.INFO)

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

from pymilvus.bulk_writer import (
    BulkFileType,
    list_import_jobs,
    bulk_import,
    get_import_progress,
)

# zilliz cluster
PUBLIC_ENDPOINT = "cluster.endpoint"
USER_NAME = "user.name"
PASSWORD = "password"

# The value of the URL is fixed.
# For overseas regions, it is: https://api.cloud.zilliz.com
# For regions in China, it is: https://api.cloud.zilliz.com.cn
CLOUD_ENDPOINT = "https://api.cloud.zilliz.com"
API_KEY = "_api_key_for_cluster_org_"

# This is currently a private preview feature. If you need to use it, please submit a request and contact us.
VOLUME_NAME = "_volume_name_for_project_"

CLUSTER_ID = "_your_cloud_cluster_id_"
DB_NAME = ""  # If db_name is not specified, use ""
COLLECTION_NAME = "_collection_name_on_the_db_"
PARTITION_NAME = ""  # If partition_name is not specified, use ""
DIM = 512


def create_connection():
    print(f"\nCreate connection...")
    connections.connect(uri=PUBLIC_ENDPOINT, user=USER_NAME, password=PASSWORD)
    print(f"\nConnected")


def build_all_type_schema():
    print(f"\n===================== build all types schema ====================")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="bool", dtype=DataType.BOOL),
        FieldSchema(name="int8", dtype=DataType.INT8),
        FieldSchema(name="int16", dtype=DataType.INT16),
        FieldSchema(name="int32", dtype=DataType.INT32),
        FieldSchema(name="int64", dtype=DataType.INT64),
        FieldSchema(name="float", dtype=DataType.FLOAT),
        FieldSchema(name="double", dtype=DataType.DOUBLE),
        FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="json", dtype=DataType.JSON),
        # from 2.4.0, milvus supports multiple vector fields in one collection
        # FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=DIM),
        FieldSchema(name="float16_vector", dtype=DataType.FLOAT16_VECTOR, dim=DIM),
        FieldSchema(name="bfloat16_vector", dtype=DataType.BFLOAT16_VECTOR, dim=DIM),
    ]

    # milvus doesn't support parsing array/sparse_vector from numpy file
    fields.append(
        FieldSchema(name="array_str", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.VARCHAR,
                    max_length=128))
    fields.append(
        FieldSchema(name="array_int", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64))
    fields.append(FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR))

    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    return schema


def example_collection_remote_volume(file_type: BulkFileType):
    schema = build_all_type_schema()
    print(f"\n===================== all field types ({file_type.name}) ====================")
    create_collection(schema, False)
    volume_upload_result = volume_remote_writer(file_type, schema)
    call_volume_import(volume_upload_result['volume_name'], volume_upload_result['path'])
    retrieve_imported_data()


def create_collection(schema: CollectionSchema, drop_if_exist: bool):
    if utility.has_collection(COLLECTION_NAME):
        if drop_if_exist:
            utility.drop_collection(COLLECTION_NAME)
    else:
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"Collection '{collection.name}' created")


def volume_remote_writer(file_type, schema):
    with VolumeBulkWriter(
            schema=schema,
            remote_path="bulk_data",
            file_type=file_type,
            chunk_size=512 * 1024 * 1024,
            cloud_endpoint=CLOUD_ENDPOINT,
            api_key=API_KEY,
            volume_name=VOLUME_NAME,
    ) as volume_bulk_writer:
        print("Append rows")
        batch_count = 10000
        for i in range(batch_count):
            row = {
                "id": i,
                "bool": True if i % 5 == 0 else False,
                "int8": i % 128,
                "int16": i % 1000,
                "int32": i % 100000,
                "int64": i,
                "float": i / 3,
                "double": i / 7,
                "varchar": f"varchar_{i}",
                "json": {"dummy": i, "ok": f"name_{i}"},
                # "float_vector": gen_float_vector(False),
                "binary_vector": gen_binary_vector(False, DIM),
                "float16_vector": gen_fp16_vector(False, DIM),
                "bfloat16_vector": gen_bf16_vector(False, DIM),
                f"dynamic_{i}": i,
                # bulkinsert doesn't support import npy with array field and sparse vector,
                # if file_type is numpy, the below values will be stored into dynamic field
                "array_str": [f"str_{k}" for k in range(5)],
                "array_int": [k for k in range(10)],
                "sparse_vector": gen_sparse_vector(False),
            }
            volume_bulk_writer.append_row(row)

        # append rows by numpy type
        for i in range(batch_count):
            id = i + batch_count
            volume_bulk_writer.append_row({
                "id": np.int64(id),
                "bool": True if i % 3 == 0 else False,
                "int8": np.int8(id % 128),
                "int16": np.int16(id % 1000),
                "int32": np.int32(id % 100000),
                "int64": np.int64(id),
                "float": np.float32(id / 3),
                "double": np.float64(id / 7),
                "varchar": f"varchar_{id}",
                "json": json.dumps({"dummy": id, "ok": f"name_{id}"}),
                # "float_vector": gen_float_vector(True),
                "binary_vector": gen_binary_vector(True, DIM),
                "float16_vector": gen_fp16_vector(True, DIM),
                "bfloat16_vector": gen_bf16_vector(True, DIM),
                f"dynamic_{id}": id,
                # bulkinsert doesn't support import npy with array field and sparse vector,
                # if file_type is numpy, the below values will be stored into dynamic field
                "array_str": np.array([f"str_{k}" for k in range(5)], np.dtype("str")),
                "array_int": np.array([k for k in range(10)], np.dtype("int64")),
                "sparse_vector": gen_sparse_vector(True),
            })

        print(f"{volume_bulk_writer.total_row_count} rows appends")
        print(f"{volume_bulk_writer.buffer_row_count} rows in buffer not flushed")
        print("Generate data files...")
        volume_bulk_writer.commit()
        print(f"Data files have been uploaded: {volume_bulk_writer.batch_files}")
        return volume_bulk_writer.get_volume_upload_result()


def retrieve_imported_data():
    collection = Collection(name=COLLECTION_NAME)
    print("Create index...")
    for field in collection.schema.fields:
        if (field.dtype == DataType.FLOAT_VECTOR or field.dtype == DataType.FLOAT16_VECTOR
                or field.dtype == DataType.BFLOAT16_VECTOR):
            collection.create_index(field_name=field.name, index_params={
                "index_type": "FLAT",
                "params": {},
                "metric_type": "L2"
            })
        elif field.dtype == DataType.BINARY_VECTOR:
            collection.create_index(field_name=field.name, index_params={
                "index_type": "BIN_FLAT",
                "params": {},
                "metric_type": "HAMMING"
            })
        elif field.dtype == DataType.SPARSE_FLOAT_VECTOR:
            collection.create_index(field_name=field.name, index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2}
            })

    ids = [100, 15000]
    print(f"Load collection and query items {ids}")
    collection.load()
    expr = f"id in {ids}"
    print(expr)
    results = collection.query(expr=expr, output_fields=["*", "vector"])
    print("Query results:")
    for item in results:
        print(item)


def call_volume_import(volume_name: str, path: str):
    print(f"\n===================== import files to cluster ====================")
    resp = bulk_import(
        url=CLOUD_ENDPOINT,
        api_key=API_KEY,
        cluster_id=CLUSTER_ID,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        volume_name=volume_name,
        data_paths=[[path]]
    )
    print(resp.json())
    job_id = resp.json()['data']['jobId']
    print(f"Create a cloudImport job, job id: {job_id}")

    print(f"\n===================== list import jobs ====================")
    resp = list_import_jobs(
        url=CLOUD_ENDPOINT,
        cluster_id=CLUSTER_ID,
        api_key=API_KEY,
        page_size=10,
        current_page=1,
    )
    print(resp.json())

    while True:
        print("Wait 5 second to check cloudImport job state...")
        time.sleep(5)

        print(f"\n===================== get import job progress ====================")
        resp = get_import_progress(
            url=CLOUD_ENDPOINT,
            cluster_id=CLUSTER_ID,
            api_key=API_KEY,
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


if __name__ == '__main__':
    create_connection()
    example_collection_remote_volume(file_type=BulkFileType.PARQUET)
