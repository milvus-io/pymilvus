import json
import logging
import math
import time
from typing import List

from examples.orm_deprecated.bulk_import.data_gengerator import *

logging.basicConfig(level=logging.INFO)

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility,
)

from pymilvus.bulk_writer import (
    RemoteBulkWriter,
    BulkFileType,
    bulk_import,
    get_import_progress,
)

# minio
MINIO_ADDRESS = "0.0.0.0:9000"
MINIO_SECRET_KEY = "minioadmin"
MINIO_ACCESS_KEY = "minioadmin"

# milvus
HOST = '127.0.0.1'
PORT = '19530'

ALL_TYPES_COLLECTION_NAME = "test_bulkwriter"
DIM = 16

def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nConnected")

def build_nullalbe_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="bool", dtype=DataType.BOOL, nullable=True),
        FieldSchema(name="int8", dtype=DataType.INT8, nullable=True),
        FieldSchema(name="int16", dtype=DataType.INT16, nullable=True),
        FieldSchema(name="int32", dtype=DataType.INT32, nullable=True),
        FieldSchema(name="int64", dtype=DataType.INT64, nullable=True),
        FieldSchema(name="float", dtype=DataType.FLOAT, nullable=True),
        FieldSchema(name="double", dtype=DataType.DOUBLE, nullable=True),
        FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=512, nullable=True),
        FieldSchema(name="json", dtype=DataType.JSON, nullable=True),

        FieldSchema(name="array_str", dtype=DataType.ARRAY, max_capacity=100,
                    element_type=DataType.VARCHAR, max_length=128, nullable=True),
        FieldSchema(name="array_int", dtype=DataType.ARRAY, max_capacity=100,
                    element_type=DataType.INT16, nullable=True),

        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="binary_vector", dtype=DataType.BINARY_VECTOR, dim=DIM),
        FieldSchema(name="bfloat16_vector", dtype=DataType.BFLOAT16_VECTOR, dim=DIM),
    ]

    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    return schema

def build_nullable_default_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="bool", dtype=DataType.BOOL, nullable=True, default_value=False),
        FieldSchema(name="int8", dtype=DataType.INT8, nullable=True, default_value=np.int8(8)),
        FieldSchema(name="int16", dtype=DataType.INT16, nullable=False, default_value=np.int16(16)),
        FieldSchema(name="int32", dtype=DataType.INT32, nullable=True, default_value=None),
        FieldSchema(name="int64", dtype=DataType.INT64, nullable=False, default_value=np.int64(64)),
        FieldSchema(name="float", dtype=DataType.FLOAT, nullable=True, default_value=np.float32(3.2)),
        FieldSchema(name="double", dtype=DataType.DOUBLE, nullable=False, default_value=np.float64(6.4)),
        FieldSchema(name="varchar", dtype=DataType.VARCHAR, max_length=512, nullable=True, default_value="this is default"),

        FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]

    schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
    return schema

def build_collection(schema: CollectionSchema):
    if utility.has_collection(ALL_TYPES_COLLECTION_NAME):
        utility.drop_collection(ALL_TYPES_COLLECTION_NAME)

    collection = Collection(name=ALL_TYPES_COLLECTION_NAME, schema=schema)
    print(f"Collection '{collection.name}' created")

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

    collection.load()
    return collection


def gen_nullable_data():
    return [
        {
            "id": 1,
            "bool": None,
            "int8": None,
            "int16": None,
            "int32": None,
            "int64": None,
            "float": None,
            "double": None,
            "varchar": None,
            "json": None,
            "array_str": None,
            "array_int": None,

            "float_vector": gen_float_vector(False, DIM),
            "sparse_vector": gen_sparse_vector(False),
            "binary_vector": gen_binary_vector(False, DIM),
            "bfloat16_vector": gen_bf16_vector(False, DIM),

            "dummy1": None, # dynamic field
        },
        {
            "id": 2,
            "bool": True,
            "int8": 22,
            "int16": -222,
            "int32": 22222,
            "int64": -2222222,
            "float": 0.2222,
            "double": 2.2222222,
            "varchar": "22222",
            "json": {"222": None},
            "array_str": [f"str_{k}" for k in range(5)],
            "array_int": [k for k in range(10)],

            "float_vector": gen_float_vector(False, DIM),
            "sparse_vector": gen_sparse_vector(False),
            "binary_vector": gen_binary_vector(False, DIM),
            "bfloat16_vector": gen_bf16_vector(False, DIM),

            "dummy2": 2222,  # dynamic field
        },
        {
            "id": np.int64(3),
            "bool": False,
            "int8": np.int8(33),
            "int16": np.int16(-333),
            "int32": np.int32(33333),
            "int64": np.int64(-3333333),
            "float": np.float32(0.33333),
            "double": np.float64(3.333333333),
            "varchar": "333333",
            "json": {"333": 33},
            "array_str": np.array([f"str_{k}" for k in range(5)], np.dtype("str")),
            "array_int": np.array([k for k in range(10)], np.dtype("int16")),

            "float_vector": gen_float_vector(True, DIM),
            "sparse_vector": gen_sparse_vector(True),
            "binary_vector": gen_binary_vector(True, DIM),
            "bfloat16_vector": gen_bf16_vector(True, DIM),
        },
    ]


def gen_nullable_default_data():
    return [
        {
            "id": 1,

            "float_vector": gen_float_vector(True, DIM),
        },
        {
            "id": 2,
            "bool": None,
            "int8": None,
            "int16": None,
            "int32": None,
            "int64": None,
            "float": None,
            "double": None,
            "varchar": None,

            "float_vector": gen_float_vector(True, DIM),
        },
        {
            "id": 3,
            "bool": True,
            "int8": 33,
            "int16":-333,
            "int32": 33333,
            "int64": -3333333,
            "float": 0.33333,
            "double": 3.333333333,
            "varchar": "333333",

            "float_vector": gen_float_vector(True, DIM),
        },
    ]

def gen_data_files(schema: CollectionSchema, file_type: BulkFileType, rows: list)-> List[List[str]]:
    print(f"\n===================== File type:{file_type.name} ====================")
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
    ) as writer:
    # with LocalBulkWriter(
    #         schema=schema,
    #         local_path="/tmp/bulk_writer",
    #         segment_size=128 * 1024 * 1024,
    #         file_type=file_type,
    # ) as writer:
        print("Append rows")
        for row in rows:
            writer.append_row(row)

        print(f"{writer.total_row_count} rows appends")
        print(f"{writer.buffer_row_count} rows in buffer not flushed")
        print("Generate data files...")
        writer.commit()
        print(f"Data files have been uploaded: {writer.batch_files}")
        return writer.batch_files


def call_bulkinsert(batch_files: List[List[str]]):
    url = f"http://{HOST}:{PORT}"
    print(f"\n===================== Import files to milvus ====================")
    resp = bulk_import(
        url=url,
        collection_name=ALL_TYPES_COLLECTION_NAME,
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

    collection = Collection(name=ALL_TYPES_COLLECTION_NAME)
    collection.load(refresh=True)
    print(f"Collection row number: {collection.num_entities}")


def compare_values(a, b, field_name: str):
    if isinstance(a, list) and isinstance(b, list):
        if len(b) == len(a):
            # array, float vector field
            for i, val in enumerate(a):
                compare_values(val, b[i], field_name)
        elif len(b) == 1:
            # binary vector field
            if len(b[0]) == DIM/8:
                if len(a) == DIM:
                    a = np.packbits(a, axis=-1)
                int8_array = np.frombuffer(b[0], dtype=np.uint8)
                for i, val in enumerate(a):
                    compare_values(val, int8_array[i], field_name)
        else:
            raise Exception(f"Unknown case for field '{field_name}': {a} vs {b}")
    elif isinstance(b, list) and isinstance(b[0], bytes):
        # flaot16/bfloat16 vector field
        compare_values(a, b[0], field_name)
    elif isinstance(a, dict) and isinstance(b, dict):
        # json, sparse vector field
        if "indices" in a.keys():
            a = dict(zip(a["indices"], a["values"]))
        for k, v in a.items():
            compare_values(v, b[k], field_name)
    elif isinstance(a, dict) and isinstance(b, str):
        # json field
        s = json.dumps(a)
        compare_values(s, b, field_name)
    elif isinstance(a, float):
        if not math.isclose(a, b, abs_tol=1e-5):
            raise Exception(f"Unmatch value for field `{field_name}`: {a} vs {b}")
    elif a != b:
        raise Exception(f"Unmatch value for field `{field_name}`: {a} vs {b}")

def verify_data(compare_rows: list):
    collection = Collection(name=ALL_TYPES_COLLECTION_NAME)
    ids = [row["id"] for row in compare_rows]
    print(f"Query items {ids}")
    expr = f"id in {ids}"
    print(expr)
    results = collection.query(expr=expr, output_fields=["*"])
    print(f"\n===================== Query results ====================")
    for i, row in enumerate(compare_rows):
        result = results[i]
        print(result)
        for k, v in row.items():
            compare_values(v, result[k], k)
    print("Retrieved data is correct")

if __name__ == '__main__':
    create_connection()

    file_types = [
        BulkFileType.JSON,
        BulkFileType.PARQUET,
        BulkFileType.CSV,
    ]

    # deal with nullable schema
    for file_type in file_types:
        schema = build_nullalbe_schema()
        rows = gen_nullable_data()
        batch_files = gen_data_files(schema=schema, file_type=file_type, rows=rows)
        if len(batch_files) > 0:
            build_collection(schema)
            call_bulkinsert(batch_files)
            verify_data(rows)

    # deal with nullalbe&default_value schema
    for file_type in file_types:
        schema = build_nullable_default_schema()
        rows = gen_nullable_default_data()
        batch_files = gen_data_files(schema=schema, file_type=file_type, rows=rows)
        if len(batch_files) > 0:
            build_collection(schema)
            call_bulkinsert(batch_files)
            verify_data(rows)
