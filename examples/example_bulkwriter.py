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
import time
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
    LocalBulkWriter,
    RemoteBulkWriter,
    BulkFileType,
    bulk_import,
    get_import_progress,
    list_import_jobs,
    BulkInsertState,
)

# minio
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

# milvus
HOST = '127.0.0.1'
PORT = '19530'

SIMPLE_COLLECTION_NAME = "for_bulkwriter"
ALL_TYPES_COLLECTION_NAME = "all_types_for_bulkwriter"
DIM = 512

def gen_binary_vector():
    raw_vector = [random.randint(0, 1) for i in range(DIM)]
    binary_vectors = np.packbits(raw_vector, axis=-1).tolist()
    return binary_vectors

def gen_float_vector():
    return [random.random() for _ in range(DIM)]

def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nConnected")


def build_simple_collection():
    print(f"\n===================== create collection ====================")
    if utility.has_collection(SIMPLE_COLLECTION_NAME):
        utility.drop_collection(SIMPLE_COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=SIMPLE_COLLECTION_NAME, schema=schema)
    print(f"Collection '{collection.name}' created")
    return collection.schema

def build_all_type_schema(bin_vec: bool, has_array: bool):
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
        FieldSchema(name="vector", dtype=DataType.BINARY_VECTOR, dim=DIM) if bin_vec else FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]

    if has_array:
        fields.append(FieldSchema(name="array_str", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.VARCHAR, max_length=128))
        fields.append(FieldSchema(name="array_int", dtype=DataType.ARRAY, max_capacity=100, element_type=DataType.INT64))

    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    return schema

def read_sample_data(file_path: str, writer: [LocalBulkWriter, RemoteBulkWriter]):
    csv_data = pd.read_csv(file_path)
    print(f"The csv file has {csv_data.shape[0]} rows")
    for i in range(csv_data.shape[0]):
        row = {}
        for col in csv_data.columns.values:
            if col == "vector":
                vec = json.loads(csv_data[col][i]) # convert the string format vector to List[float]
                row[col] = vec
            else:
                row[col] = csv_data[col][i]

        writer.append_row(row)

def local_writer(schema: CollectionSchema, file_type: BulkFileType):
    print(f"\n===================== local writer ({file_type.name}) ====================")
    with LocalBulkWriter(
            schema=schema,
            local_path="/tmp/bulk_writer",
            segment_size=128*1024*1024,
            file_type=file_type,
    ) as local_writer:
        # read data from csv
        read_sample_data("./data/train_embeddings.csv", local_writer)

        # append rows
        for i in range(100000):
            local_writer.append_row({"path": f"path_{i}", "vector": gen_float_vector(), "label": f"label_{i}"})

        print(f"{local_writer.total_row_count} rows appends")
        print(f"{local_writer.buffer_row_count} rows in buffer not flushed")
        local_writer.commit()
        batch_files = local_writer.batch_files

    print(f"Local writer done! output local files: {batch_files}")


def remote_writer(schema: CollectionSchema, file_type: BulkFileType):
    print(f"\n===================== remote writer ({file_type.name}) ====================")
    with RemoteBulkWriter(
            schema=schema,
            remote_path="bulk_data",
            connect_param=RemoteBulkWriter.S3ConnectParam(
                endpoint=MINIO_ADDRESS,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                bucket_name="a-bucket",
            ),
            segment_size=512 * 1024 * 1024,
            file_type=file_type,
    ) as remote_writer:
        # read data from csv
        read_sample_data("./data/train_embeddings.csv", remote_writer)

        # append rows
        for i in range(10000):
            remote_writer.append_row({"path": f"path_{i}", "vector": gen_float_vector(), "label": f"label_{i}"})

        print(f"{remote_writer.total_row_count} rows appends")
        print(f"{remote_writer.buffer_row_count} rows in buffer not flushed")
        remote_writer.commit()
        batch_files = remote_writer.batch_files

    print(f"Remote writer done! output remote files: {batch_files}")

def parallel_append(schema: CollectionSchema):
    print(f"\n===================== parallel append ====================")
    def _append_row(writer: LocalBulkWriter, begin: int, end: int):
        try:
            for i in range(begin, end):
                writer.append_row({"path": f"path_{i}", "vector": gen_float_vector(), "label": f"label_{i}"})
                if i%100 == 0:
                    print(f"{threading.current_thread().name} inserted {i-begin} items")
        except Exception as e:
            print("failed to append row!")

    local_writer = LocalBulkWriter(
        schema=schema,
        local_path="/tmp/bulk_writer",
        segment_size=128 * 1024 * 1024, # 128MB
        file_type=BulkFileType.JSON,
    )
    threads = []
    thread_count = 10
    rows_per_thread = 1000
    for k in range(thread_count):
        x = threading.Thread(target=_append_row, args=(local_writer, k*rows_per_thread, (k+1)*rows_per_thread,))
        threads.append(x)
        x.start()
        print(f"Thread '{x.name}' started")

    for th in threads:
        th.join()
        print(f"Thread '{th.name}' finished")

    print(f"{local_writer.total_row_count} rows appends")
    print(f"{local_writer.buffer_row_count} rows in buffer not flushed")
    local_writer.commit()
    print(f"Append finished, {thread_count*rows_per_thread} rows")

    row_count = 0
    batch_files = local_writer.batch_files
    for batch in batch_files:
        for file_path in batch:
            with open(file_path, 'r') as file:
                data = json.load(file)

            rows = data['rows']
            row_count = row_count + len(rows)
            print(f"The file {file_path} contains {len(rows)} rows. Verify the content...")

            for row in rows:
                pa = row['path']
                label = row['label']
                assert pa.replace("path_", "") == label.replace("label_", "")

    assert row_count == thread_count * rows_per_thread
    print("Data is correct")


def all_types_writer(bin_vec: bool, schema: CollectionSchema, file_type: BulkFileType)->list:
    print(f"\n===================== all field types ({file_type.name}) binary_vector={bin_vec} ====================")
    with RemoteBulkWriter(
            schema=schema,
            remote_path="bulk_data",
            connect_param=RemoteBulkWriter.S3ConnectParam(
                endpoint=MINIO_ADDRESS,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                bucket_name="a-bucket",
            ),
            file_type=file_type,
    ) as remote_writer:
        print("Append rows")
        batch_count = 10000
        for i in range(batch_count):
            row = {
                "id": i,
                "bool": True if i%5 == 0 else False,
                "int8": i%128,
                "int16": i%1000,
                "int32": i%100000,
                "int64": i,
                "float": i/3,
                "double": i/7,
                "varchar": f"varchar_{i}",
                "json": {"dummy": i, "ok": f"name_{i}"},
                "vector": gen_binary_vector() if bin_vec else gen_float_vector(),
                f"dynamic_{i}": i,
                # bulkinsert doesn't support import npy with array field, the below values will be stored into dynamic field
                "array_str": [f"str_{k}" for k in range(5)],
                "array_int": [k for k in range(10)],
            }
            remote_writer.append_row(row)

        # append rows by numpy type
        for i in range(batch_count):
            remote_writer.append_row({
                "id": np.int64(i+batch_count),
                "bool": True if i % 3 == 0 else False,
                "int8": np.int8(i%128),
                "int16": np.int16(i%1000),
                "int32": np.int32(i%100000),
                "int64": np.int64(i),
                "float": np.float32(i/3),
                "double": np.float64(i/7),
                "varchar": f"varchar_{i}",
                "json": json.dumps({"dummy": i, "ok": f"name_{i}"}),
                "vector": np.array(gen_binary_vector()).astype(np.dtype("int8")) if bin_vec else np.array(gen_float_vector()).astype(np.dtype("float32")),
                f"dynamic_{i}": i,
                # bulkinsert doesn't support import npy with array field, the below values will be stored into dynamic field
                "array_str": np.array([f"str_{k}" for k in range(5)], np.dtype("str")),
                "array_int": np.array([k for k in range(10)], np.dtype("int64")),
            })

        print(f"{remote_writer.total_row_count} rows appends")
        print(f"{remote_writer.buffer_row_count} rows in buffer not flushed")
        print("Generate data files...")
        remote_writer.commit()
        print(f"Data files have been uploaded: {remote_writer.batch_files}")
        return remote_writer.batch_files


def call_bulkinsert(schema: CollectionSchema, batch_files: list):
    print(f"\n===================== call bulkinsert ====================")
    if utility.has_collection(ALL_TYPES_COLLECTION_NAME):
        utility.drop_collection(ALL_TYPES_COLLECTION_NAME)

    collection = Collection(name=ALL_TYPES_COLLECTION_NAME, schema=schema)
    print(f"Collection '{collection.name}' created")

    task_ids = []
    for files in batch_files:
        task_id = utility.do_bulk_insert(collection_name=ALL_TYPES_COLLECTION_NAME, files=files)
        task_ids.append(task_id)
        print(f"Create a bulkinert task, task id: {task_id}")

    while len(task_ids) > 0:
        print("Wait 1 second to check bulkinsert tasks state...")
        time.sleep(1)
        for id in task_ids:
            state = utility.get_bulk_insert_state(task_id=id)
            if state.state == BulkInsertState.ImportFailed or state.state == BulkInsertState.ImportFailedAndCleaned:
                print(f"The task {state.task_id} failed, reason: {state.failed_reason}")
                task_ids.remove(id)
            elif state.state == BulkInsertState.ImportCompleted:
                print(f"The task {state.task_id} completed")
                task_ids.remove(id)

    print(f"Collection row number: {collection.num_entities}")


def retrieve_imported_data(bin_vec: bool):
    collection = Collection(name=ALL_TYPES_COLLECTION_NAME)
    print("Create index...")
    index_param = {
        "index_type": "BIN_FLAT",
        "params": {},
        "metric_type": "HAMMING"
    } if bin_vec else {
        "index_type": "FLAT",
        "params": {},
        "metric_type": "L2"
    }
    collection.create_index(field_name="vector", index_params=index_param)

    ids = [100, 5000]
    print(f"Load collection and query items {ids}")
    collection.load()
    expr = f"id in {ids}"
    print(expr)
    results = collection.query(expr=expr, output_fields=["*", "vector"])
    print("Query results:")
    for item in results:
        print(item)

def cloud_bulkinsert():
    url = "https://_your_cloud_server_url_"
    api_key = "_api_key_for_the_url_"
    cluster_id = "_your_cloud_instance_id_"
    collection_name = "_collection_name_on_the_cloud_"

    print(f"\n===================== import files to cloud vectordb ====================")
    object_url = "_your_object_storage_service_url_"
    object_url_access_key = "_your_object_storage_service_access_key_"
    object_url_secret_key = "_your_object_storage_service_secret_key_"
    resp = bulk_import(
        url=url,
        api_key=api_key,
        object_url=object_url,
        access_key=object_url_access_key,
        secret_key=object_url_secret_key,
        cluster_id=cluster_id,
        collection_name=collection_name,
    )
    print(resp.json())

    print(f"\n===================== get import job progress ====================")
    job_id = resp.json()['data']['jobId']
    resp = get_import_progress(
        url=url,
        api_key=api_key,
        job_id=job_id,
        cluster_id=cluster_id,
    )
    print(resp.json())

    print(f"\n===================== list import jobs ====================")
    resp = list_import_jobs(
        url=url,
        api_key=api_key,
        cluster_id=cluster_id,
        page_size=10,
        current_page=1,
    )
    print(resp.json())


if __name__ == '__main__':
    create_connection()

    file_types = [
        BulkFileType.JSON,
        BulkFileType.NUMPY,
        BulkFileType.PARQUET,
    ]

    schema = build_simple_collection()
    for file_type in file_types:
        local_writer(schema=schema, file_type=file_type)

    for file_type in file_types:
        remote_writer(schema=schema, file_type=file_type)

    parallel_append(schema)

    # float vectors + all scalar types
    for file_type in file_types:
        # Note: bulkinsert doesn't support import npy with array field
        schema = build_all_type_schema(bin_vec=False, has_array=False if file_type==BulkFileType.NUMPY else True)
        batch_files = all_types_writer(bin_vec=False, schema=schema, file_type=file_type)
        call_bulkinsert(schema, batch_files)
        retrieve_imported_data(bin_vec=False)

    # binary vectors + all scalar types
    for file_type in file_types:
        # Note: bulkinsert doesn't support import npy with array field
        schema = build_all_type_schema(bin_vec=True, has_array=False if file_type == BulkFileType.NUMPY else True)
        batch_files = all_types_writer(bin_vec=True, schema=schema, file_type=file_type)
        call_bulkinsert(schema, batch_files)
        retrieve_imported_data(bin_vec=True)

    # # to call cloud bulkinsert api, you need to apply a cloud service from Zilliz Cloud(https://zilliz.com/cloud)
    # cloud_bulkinsert()

