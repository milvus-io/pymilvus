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

import os
import json
import random
import threading

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example_bulkwriter")

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    LocalBulkWriter,
    RemoteBulkWriter,
    BulkFileType,
    bulk_insert,
    get_job_progress,
    list_jobs,
)

# minio
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

# milvus
HOST = '127.0.0.1'
PORT = '19530'

COLLECTION_NAME = "test_abc"
DIM = 256

def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nConnected")


def build_collection():
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    field1 = FieldSchema(name="id", dtype=DataType.INT64, auto_id=True, is_primary=True)
    field2 = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    field3 = FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=100)
    schema = CollectionSchema(fields=[field1, field2, field3])
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print("Collection created")
    return collection.schema

def test_local_writer_json(schema: CollectionSchema):
    local_writer = LocalBulkWriter(schema=schema,
                                   local_path="/tmp/bulk_data",
                                   segment_size=4*1024*1024,
                                   file_type=BulkFileType.JSON_RB,
                                   )
    for i in range(10):
        local_writer.append({"id": i, "vector": [random.random() for _ in range(DIM)], "desc": f"description_{i}"})

    local_writer.commit()
    print("test local writer done!")
    print(local_writer.data_path)
    return local_writer.data_path

def test_local_writer_npy(schema: CollectionSchema):
    local_writer = LocalBulkWriter(schema=schema, local_path="/tmp/bulk_data", segment_size=4*1024*1024)
    for i in range(10000):
        local_writer.append({"id": i, "vector": [random.random() for _ in range(DIM)], "desc": f"description_{i}"})

    local_writer.commit()
    print("test local writer done!")
    print(local_writer.data_path)
    return local_writer.data_path


def _append_row(writer: LocalBulkWriter, begin: int, end: int):
    for i in range(begin, end):
        writer.append({"id": i, "vector": [random.random() for _ in range(DIM)], "desc": f"description_{i}"})

def test_parallel_append(schema: CollectionSchema):
    local_writer = LocalBulkWriter(schema=schema,
                                   local_path="/tmp/bulk_data",
                                   segment_size=1000 * 1024 * 1024,
                                   file_type=BulkFileType.JSON_RB,
                                   )
    threads = []
    thread_count = 100
    rows_per_thread = 1000
    for k in range(thread_count):
        x = threading.Thread(target=_append_row, args=(local_writer, k*rows_per_thread, (k+1)*rows_per_thread,))
        threads.append(x)
        x.start()
        print(f"Thread '{x.name}' started")

    for th in threads:
        th.join()

    local_writer.commit()
    print(f"Append finished, {thread_count*rows_per_thread} rows")
    file_path = os.path.join(local_writer.data_path, "1.json")
    with open(file_path, 'r') as file:
        data = json.load(file)

    print("Verify the output content...")
    rows = data['rows']
    assert len(rows) == thread_count*rows_per_thread
    for i in range(len(rows)):
        row = rows[i]
        assert row['desc'] == f"description_{row['id']}"


def test_remote_writer(schema: CollectionSchema):
    remote_writer = RemoteBulkWriter(schema=schema,
                                     remote_path="bulk_data",
                                     local_path="/tmp/bulk_data",
                                     connect_param=RemoteBulkWriter.ConnectParam(
                                         endpoint=MINIO_ADDRESS,
                                         access_key=MINIO_ACCESS_KEY,
                                         secret_key=MINIO_SECRET_KEY,
                                         bucket_name="a-bucket",
                                     ),
                                     segment_size=50 * 1024 * 1024,
                                     )

    for i in range(10000):
        if i % 1000 == 0:
            logger.info(f"{i} rows has been append to remote writer")
        remote_writer.append({"id": i, "vector": [random.random() for _ in range(DIM)], "desc": f"description_{i}"})

    remote_writer.commit()
    print("test remote writer done!")
    print(remote_writer.data_path)
    return remote_writer.data_path


def test_cloud_bulkinsert():
    host = "_your_cloud_url_"
    cluster_id = "_your_cloud_instance_"
    request_url = f"https://{host}/v1/vector/collections/import"

    print(f"===================== import files to cloud vectordb ====================")
    object_url = "_your_object_storage_service_url_"
    object_url_access_key = "_your_object_storage_service_access_key_"
    object_url_secret_key = "_your_object_storage_service_secret_key_"
    resp = bulk_insert(
        url=request_url,
        object_url=object_url,
        access_key=object_url_access_key,
        secret_key=object_url_secret_key,
        cluster_id=cluster_id,
        collection_name=COLLECTION_NAME,
    )
    print(resp)

    print(f"===================== get import job progress ====================")
    job_id = resp['data']['jobId']
    request_url = f"https://{host}/v1/vector/collections/import/get"
    resp = get_job_progress(
        url=request_url,
        job_id=job_id,
        cluster_id=cluster_id,
    )
    print(resp)

    print(f"===================== list import jobs ====================")
    request_url = f"https://{host}/v1/vector/collections/import/list"
    resp = list_jobs(
        url=request_url,
        cluster_id=cluster_id,
        page_size=10,
        current_page=1,
    )
    print(resp)


if __name__ == '__main__':
    create_connection()
    schema = build_collection()

    test_local_writer_json(schema)
    test_local_writer_npy(schema)
    test_remote_writer(schema)
    test_parallel_append(schema)

    # test_cloud_bulkinsert()

