import datetime
import pytz
import time
import numpy as np
from typing import List

from pymilvus import (
    MilvusClient,
    CollectionSchema, DataType,
)

from pymilvus.bulk_writer import (
    RemoteBulkWriter, LocalBulkWriter,
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

COLLECTION_NAME = "for_bulkwriter"
DIM = 16  # must >= 8
ROW_COUNT = 10

client = MilvusClient(uri="http://localhost:19530", user="root", password="Milvus")
print(client.get_server_version())


def gen_float_vector(i):
    return [i / 4 for _ in range(DIM)]


def gen_binary_vector(i, to_numpy_arr: bool):
    raw_vector = [(i + k) % 2 for k in range(DIM)]
    if to_numpy_arr:
        return np.packbits(raw_vector, axis=-1)
    return raw_vector


def gen_sparse_vector(i, indices_values: bool):
    raw_vector = {}
    dim = 3
    if indices_values:
        raw_vector["indices"] = [i + k for k in range(dim)]
        raw_vector["values"] = [(i + k) / 8 for k in range(dim)]
    else:
        for k in range(dim):
            raw_vector[i + k] = (i + k) / 8
    return raw_vector


def build_schema():
    schema = MilvusClient.create_schema(enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="bool", datatype=DataType.BOOL)
    schema.add_field(field_name="int8", datatype=DataType.INT8)
    schema.add_field(field_name="int16", datatype=DataType.INT16)
    schema.add_field(field_name="int32", datatype=DataType.INT32)
    schema.add_field(field_name="int64", datatype=DataType.INT64)
    schema.add_field(field_name="float", datatype=DataType.FLOAT)
    schema.add_field(field_name="double", datatype=DataType.DOUBLE)
    schema.add_field(field_name="varchar", datatype=DataType.VARCHAR, max_length=100)
    schema.add_field(field_name="json", datatype=DataType.JSON)
    schema.add_field(field_name="timestamp", datatype=DataType.TIMESTAMPTZ)
    schema.add_field(field_name="geometry", datatype=DataType.GEOMETRY)

    schema.add_field(field_name="array_bool", datatype=DataType.ARRAY, element_type=DataType.BOOL, max_capacity=10)
    schema.add_field(field_name="array_int8", datatype=DataType.ARRAY, element_type=DataType.INT8, max_capacity=10)
    schema.add_field(field_name="array_int16", datatype=DataType.ARRAY, element_type=DataType.INT16, max_capacity=10)
    schema.add_field(field_name="array_int32", datatype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=10)
    schema.add_field(field_name="array_int64", datatype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=10)
    schema.add_field(field_name="array_float", datatype=DataType.ARRAY, element_type=DataType.FLOAT, max_capacity=10)
    schema.add_field(field_name="array_double", datatype=DataType.ARRAY, element_type=DataType.DOUBLE, max_capacity=10)
    schema.add_field(field_name="array_varchar", datatype=DataType.ARRAY, element_type=DataType.VARCHAR,
                     max_capacity=10, max_length=100)

    schema.add_field(field_name="float_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="binary_vector", datatype=DataType.BINARY_VECTOR, dim=DIM)

    struct_schema = MilvusClient.create_struct_field_schema()
    struct_schema.add_field("struct_bool", DataType.BOOL)
    struct_schema.add_field("struct_int8", DataType.INT8)
    struct_schema.add_field("struct_int16", DataType.INT16)
    struct_schema.add_field("struct_int32", DataType.INT32)
    struct_schema.add_field("struct_int64", DataType.INT64)
    struct_schema.add_field("struct_float", DataType.FLOAT)
    struct_schema.add_field("struct_double", DataType.DOUBLE)
    struct_schema.add_field("struct_varchar", DataType.VARCHAR, max_length=100)
    struct_schema.add_field("struct_float_vec", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("struct_field", datatype=DataType.ARRAY, element_type=DataType.STRUCT,
                     struct_schema=struct_schema, max_capacity=1000)
    schema.verify()
    return schema


def build_collection(schema: CollectionSchema):
    index_params = client.prepare_index_params()
    for field in schema.fields:
        if (field.dtype in [DataType.FLOAT_VECTOR, DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]):
            index_params.add_index(field_name=field.name,
                                   index_type="AUTOINDEX",
                                   metric_type="L2")
        elif field.dtype == DataType.BINARY_VECTOR:
            index_params.add_index(field_name=field.name,
                                   index_type="AUTOINDEX",
                                   metric_type="HAMMING")
        elif field.dtype == DataType.SPARSE_FLOAT_VECTOR:
            index_params.add_index(field_name=field.name,
                                   index_type="SPARSE_INVERTED_INDEX",
                                   metric_type="IP")

    for struct_field in schema.struct_fields:
        for field in struct_field.fields:
            if (field.dtype == DataType.FLOAT_VECTOR):
                index_params.add_index(field_name=f"{struct_field.name}[{field.name}]",
                                       index_name=f"{struct_field.name}_{field.name}",
                                       index_type="HNSW",
                                       metric_type="MAX_SIM_COSINE")

    print(f"Drop collection: {COLLECTION_NAME}")
    client.drop_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    print(f"Collection created: {COLLECTION_NAME}")
    print(client.describe_collection(collection_name=COLLECTION_NAME))


def gen_row(i):
    shanghai_tz = pytz.timezone("Asia/Shanghai")
    row = {
        "id": i,
        "float_vector": gen_float_vector(i),
        "sparse_vector": gen_sparse_vector(i, False if i % 2 == 0 else True),
        "binary_vector": gen_binary_vector(i, False if i % 2 == 0 else True),
        "bool": True,
        "int8": i % 128,
        "int16": i % 32768,
        "int32": i,
        "int64": i,
        "float": i / 4,
        "double": i / 3,
        "varchar": f"varchar_{i}",
        "json": {"dummy": i},
        "timestamp": shanghai_tz.localize(
            datetime.datetime(2025, 1, 1, 0, 0, 0) + datetime.timedelta(days=i)
        ).isoformat(),
        "geometry": f"POINT ({i} {i})",

        "array_bool": [True if (i + k) % 2 == 0 else False for k in range(4)],
        "array_int8": [(i + k) % 128 for k in range(4)],
        "array_int16": [(i + k) % 32768 for k in range(4)],
        "array_int32": [(i + k) + 1000 for k in range(4)],
        "array_int64": [(i + k) + 100 for k in range(5)],
        "array_float": [(i + k) / 4 for k in range(5)],
        "array_double": [(i + k) / 3 for k in range(5)],
        "array_varchar": [f"element_{i + k}" for k in range(5)],

        "struct_field": [
            {
                "struct_bool": True,
                "struct_int8": i % 128,
                "struct_int16": i % 32768,
                "struct_int32": i,
                "struct_int64": i,
                "struct_float": i / 4,
                "struct_double": i / 3,
                "struct_varchar": f"aaa_{i}",
                "struct_float_vec": gen_float_vector(i)
            },
            {
                "struct_bool": False,
                "struct_int8": -(i % 128),
                "struct_int16": -(i % 32768),
                "struct_int32": -i,
                "struct_int64": -i,
                "struct_float": -i / 4,
                "struct_double": -i / 3,
                "struct_varchar": f"aaa_{i * 1000}",
                "struct_float_vec": gen_float_vector(i)
            },
        ],
    }
    return row


def bulk_writer(writer):
    for i in range(ROW_COUNT):
        row = gen_row(i)
        print(row)
        writer.append_row(row)
        if ((i + 1) % 1000 == 0) or (i == ROW_COUNT - 1):
            print(f"{i + 1} rows appends")

    print(f"{writer.total_row_count} rows appends")
    print(f"{writer.buffer_row_count} rows in buffer not flushed")
    writer.commit()
    batch_files = writer.batch_files
    print(f"Remote writer done! output remote files: {batch_files}")
    return batch_files


def remote_writer(schema: CollectionSchema, file_type: BulkFileType):
    print(f"\n===================== remote writer ({file_type.name}) ====================")
    with RemoteBulkWriter(
            schema=schema,
            remote_path="bulk_data",
            local_path="/tmp/PARQUET",
            connect_param=RemoteBulkWriter.S3ConnectParam(
                endpoint=MINIO_ADDRESS,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                bucket_name="a-bucket",
            ),
            segment_size=512 * 1024 * 1024,
            file_type=file_type,
    ) as writer:
        return bulk_writer(writer)


def call_bulk_import(batch_files: List[List[str]]):
    url = f"http://{HOST}:{PORT}"

    print(f"\n===================== import files to milvus ====================")
    resp = bulk_import(
        url=url,
        collection_name=COLLECTION_NAME,
        files=batch_files,
    )
    print(resp.json())
    job_id = resp.json()['data']['jobId']
    print(f"Create a bulk_import job, job id: {job_id}")

    while True:
        print("Wait 2 second to check bulk_import job state...")
        time.sleep(2)

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


def local_writer(schema: CollectionSchema, file_type: BulkFileType):
    print(f"\n===================== local writer ({file_type.name}) ====================")
    writer = LocalBulkWriter(
        schema=schema,
        local_path="./" + file_type.name,
        chunk_size=16 * 1024 * 1024,
        file_type=file_type
    )
    return bulk_writer(writer)


def verify_imported_data():
    # refresh_load() ensure the import data is loaded
    client.refresh_load(collection_name=COLLECTION_NAME)
    res = client.query(collection_name=COLLECTION_NAME, filter="", output_fields=["count(*)"],
                       consistency_level="Strong")
    print(f'row count: {res[0]["count(*)"]}')
    results = client.query(collection_name=COLLECTION_NAME,
                           filter="id >= 0",
                           output_fields=["*"])
    print(f"\n===================== query results ====================")
    for item in results:
        print(item)
        id = item["id"]
        original_row = gen_row(id)
        for key in original_row.keys():
            if key not in item:
                raise Exception(f"{key} is missed in query result")
            if key == "binary_vector":
                # returned binary vector is wrapped by a list, this is a bug
                original_row[key] = [bytes(gen_binary_vector(id, True).tolist())]
            elif key == "sparse_vector":
                # returned sparse vector is id-pair format
                original_row[key] = gen_sparse_vector(id, False)
            elif key == "timestamp":
                # TODO: compare the timestamp values
                continue
            if item[key] != original_row[key]:
                raise Exception(f"value of {key} is unequal, original value: {original_row[key]}, query value: {item[key]}")
        print(f"Query result of id={id} is correct")


def test_file_type(file_type: BulkFileType):
    print(f"\n########################## {file_type.name} ##################################")
    schema = build_schema()
    batch_files = local_writer(schema=schema, file_type=file_type)
    build_collection(schema)
    batch_files = remote_writer(schema=schema, file_type=file_type)
    call_bulk_import(batch_files=batch_files)
    verify_imported_data()


if __name__ == '__main__':
    file_types = [
        BulkFileType.PARQUET,
        BulkFileType.JSON,
        BulkFileType.CSV,
    ]
    for file_type in file_types:
        test_file_type(file_type)
