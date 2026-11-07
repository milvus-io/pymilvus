"""Run against an explicitly supplied server with nullable struct import support.

Set MILVUS_BULK_IMPORT_URI and MILVUS_BULK_IMPORT_STAGE_COMMAND after confirming
the test environment. The staging command receives a local file and a unique
object key, uploads through the approved staging route, and prints its import
path. This module does not configure or start storage services.
"""

import json
import os
import shlex
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import ml_dtypes
import numpy as np
import pyarrow.parquet as pq
import pytest
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, StructFieldSchema
from pymilvus.bulk_writer import BulkFileType, LocalBulkWriter, bulk_import, get_import_progress

URI = os.environ.get("MILVUS_BULK_IMPORT_URI")
pytestmark = pytest.mark.skipif(not URI, reason="MILVUS_BULK_IMPORT_URI is not configured")
FORMATS = [BulkFileType.JSON, BulkFileType.JSONL, BulkFileType.PARQUET, BulkFileType.CSV]
VECTORS = [
    ("v", DataType.FLOAT_VECTOR, 2, "FLAT", "L2"),
    ("binary", DataType.BINARY_VECTOR, 8, "BIN_FLAT", "HAMMING"),
    ("half", DataType.FLOAT16_VECTOR, 2, "FLAT", "L2"),
    ("bfloat", DataType.BFLOAT16_VECTOR, 2, "FLAT", "L2"),
    ("int8", DataType.INT8_VECTOR, 2, "HNSW", "L2"),
]


def collection_schema():
    struct = StructFieldSchema()
    for name, dtype, dim, _, _ in VECTORS:
        struct.add_field(name, dtype, dim=dim, nullable=True)
    struct.add_field("score", DataType.INT64, nullable=True)
    struct.add_field("label", DataType.VARCHAR, max_length=32, nullable=True)
    struct.add_field(
        "tags",
        DataType.ARRAY,
        element_type=DataType.INT64,
        max_capacity=4,
        nullable=True,
        element_nullable=True,
    )
    schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        nullable=True,
        max_capacity=8,
    )
    return schema


def sample_rows():
    return [
        {"id": 1, "clips": None},
        {"id": 2, "clips": []},
        {
            "id": 3,
            "clips": [
                {"v": None, "tags": [10, None], "label": 'a,"b;null'},
                {
                    "v": [1.0, 0.0],
                    "binary": b"\x01",
                    "half": np.array([1.0, 2.0], dtype=np.float16),
                    "bfloat": np.array([1.0, 2.0], dtype=ml_dtypes.bfloat16),
                    "int8": np.array([1, 2], dtype=np.int8),
                    "score": 7,
                    "tags": None,
                },
                {
                    "v": [4.0, 0.0],
                    "binary": b"\x03",
                    "half": np.array([3.0, 4.0], dtype=np.float16),
                    "bfloat": np.array([3.0, 4.0], dtype=ml_dtypes.bfloat16),
                    "int8": np.array([3, 4], dtype=np.int8),
                    "tags": [],
                    "label": "null",
                },
            ],
        },
        {"id": 4, "clips": [{}, {}]},
        {"id": 5, "clips": [{"v": [8.0, 0.0]}]},
    ]


def expected_rows(mode):
    empty = {**{name: None for name, *_ in VECTORS}, "score": None, "tags": None, "label": None}
    elements = [
        {**empty, "tags": [10, None], "label": 'a,"b;null'},
        {
            **empty,
            "v": [1.0, 0.0],
            "binary": [b"\x01"],
            "half": [1.0, 2.0],
            "bfloat": [16256, 16384],
            "int8": [1, 2],
            "score": 7,
        },
        {
            **empty,
            "v": [4.0, 0.0],
            "binary": [b"\x03"],
            "half": [3.0, 4.0],
            "bfloat": [16448, 16512],
            "int8": [3, 4],
            "tags": [],
            "label": "null",
        },
    ]
    result = {
        1: None,
        2: [],
        3: elements,
        4: [dict(empty), dict(empty)],
        5: [{**empty, "v": [8.0, 0.0]}],
    }
    for clips in result.values():
        for element in clips or []:
            if mode == "add":
                element.update(added_number=None, added_vector=None, added_tags=None)
            elif mode == "drop":
                element.pop("label")
    return result


def stage_file(file_path):
    command = os.environ.get("MILVUS_BULK_IMPORT_STAGE_COMMAND")
    assert command, "Configure an approved MILVUS_BULK_IMPORT_STAGE_COMMAND before running E2E"
    key = f"bulk_writer_test/{uuid4().hex}/{file_path.name}"
    result = subprocess.run(
        [*shlex.split(command), str(file_path), key],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    import_path = result.stdout.strip()
    assert import_path and "\n" not in import_path, "Staging must print exactly one import path"
    return import_path


def wait_for_import(collection_name, file_path, config):
    token = os.environ.get("MILVUS_TOKEN", "")
    db_name = os.environ.get("MILVUS_BULK_IMPORT_DB", "")
    response = bulk_import(
        url=URI,
        collection_name=collection_name,
        api_key=token,
        db_name=db_name,
        files=[[stage_file(file_path)]],
        options=config,
    ).json()
    job_id = response["data"]["jobId"]
    deadline = time.monotonic() + 180
    while True:
        data = get_import_progress(url=URI, job_id=job_id, api_key=token, db_name=db_name).json()[
            "data"
        ]
        if data["state"] == "Failed":
            pytest.fail(f"Import {job_id} failed: {json.dumps(data)}")
        if data["state"] == "Completed":
            assert int(data["progress"]) == 100
            return
        if time.monotonic() >= deadline:
            pytest.fail(f"Import {job_id} timed out: {json.dumps(data)}")
        time.sleep(0.5)


def evolve_schema(client, name, mode):
    if mode == "drop":
        client.drop_collection_field(name, field_name="clips[label]")
    elif mode == "add":
        client.add_collection_field(name, "clips[added_number]", DataType.INT64, nullable=True)
        client.add_collection_field(
            name, "clips[added_vector]", DataType.FLOAT_VECTOR, dim=2, nullable=True
        )
        client.add_collection_field(
            name,
            "clips[added_tags]",
            DataType.ARRAY,
            element_type=DataType.INT64,
            max_capacity=4,
            nullable=True,
            element_nullable=True,
        )


def index_and_load(client, name, mode):
    index = client.prepare_index_params()
    for field, _, _, index_type, metric in VECTORS:
        index.add_index(
            f"clips[{field}]",
            index_type=index_type,
            metric_type=metric,
            params={"M": 8, "efConstruction": 32} if index_type == "HNSW" else {},
        )
    if mode == "add":
        index.add_index("clips[added_vector]", index_type="FLAT", metric_type="L2")
    client.create_index(name, index)
    client.load_collection(name)


def queried(client, name, fields):
    return {
        row["id"]: row["clips"]
        for row in client.query(name, filter="id >= 0", output_fields=["id", *fields])
    }


@pytest.mark.parametrize("file_type", FORMATS)
@pytest.mark.parametrize("mode", ["unchanged", "add", "drop"])
def test_bulk_writer_import_matches_insert(tmp_path, file_type, mode):
    token = os.environ.get("MILVUS_TOKEN", "")
    db_name = os.environ.get("MILVUS_BULK_IMPORT_DB", "default")
    client = MilvusClient(uri=URI, token=token, db_name=db_name, timeout=30)
    owned = []
    config = {"sep": ";", "nullkey": "NULL"} if file_type == BulkFileType.CSV else {}
    try:
        schema = collection_schema()
        rows = sample_rows()
        writer_rows = deepcopy(rows)
        writer_rows[2]["clips"][0]["deleted"] = "ignored"
        with LocalBulkWriter(schema, str(tmp_path), file_type=file_type, config=config) as writer:
            for row in writer_rows:
                writer.append_row(row)
            writer.commit()
            assert len(writer.batch_files) == 1 and len(writer.batch_files[0]) == 1
            file_path = Path(writer.batch_files[0][0])

        names = [f"bulk_struct_{uuid4().hex}" for _ in range(2)]
        for name in names:
            assert not client.has_collection(name)
            client.create_collection(name, schema=schema, consistency_level="Strong")
            owned.append(name)
            evolve_schema(client, name, mode)

        wait_for_import(names[0], file_path, config)
        for row in rows:
            for element in row["clips"] or []:
                if mode == "drop":
                    element.pop("label", None)
        client.insert(names[1], rows)
        client.flush(names[1])
        for name in names:
            index_and_load(client, name, mode)

        expected = expected_rows(mode)
        assert queried(client, names[0], ["clips"]) == expected
        assert queried(client, names[1], ["clips"]) == expected
        partial = [{"v": value["v"], "tags": value["tags"]} for value in expected[3]]
        for name in names:
            assert queried(client, name, ["clips[v]", "clips[tags]"])[3] == partial
        for field, _, _, index_type, metric in VECTORS:
            queries = [rows[2]["clips"][2][field], rows[2]["clips"][1][field]]
            for name in names:
                found = client.search(
                    name,
                    queries,
                    anns_field=f"clips[{field}]",
                    filter="id == 3",
                    search_params={
                        "metric_type": metric,
                        "params": {"ef": 8} if index_type == "HNSW" else {},
                    },
                    limit=5,
                    output_fields=["clips"],
                )
                assert [[(hit["id"], hit["offset"]) for hit in result] for result in found] == [
                    [(3, 2), (3, 1)],
                    [(3, 1), (3, 2)],
                ]
                assert all(
                    hit["entity"]["clips"] == expected[3] for result in found for hit in result
                )
        for name in names:
            found = client.search(
                name,
                [[8.0, 0.0]],
                anns_field="clips[v]",
                filter="id == 5",
                search_params={"metric_type": "L2"},
                limit=2,
            )
            assert [(hit["id"], hit["offset"]) for hit in found[0]] == [(5, 0)]
    finally:
        try:
            for name in owned:
                client.drop_collection(name)
        finally:
            client.close()


def test_parquet_import_after_all_struct_children_replaced(tmp_path):
    token = os.environ.get("MILVUS_TOKEN", "")
    db_name = os.environ.get("MILVUS_BULK_IMPORT_DB", "default")
    client = MilvusClient(uri=URI, token=token, db_name=db_name, timeout=30)
    owned = []
    try:
        schema = collection_schema()
        schema.add_field("anchor", DataType.FLOAT_VECTOR, dim=2)
        rows = [{**row, "anchor": [float(row["id"]), 0.0]} for row in sample_rows()]
        with LocalBulkWriter(schema, str(tmp_path), file_type=BulkFileType.PARQUET) as writer:
            for row in deepcopy(rows):
                writer.append_row(row)
            writer.commit()
            assert len(writer.batch_files) == 1 and len(writer.batch_files[0]) == 1
            file_path = Path(writer.batch_files[0][0])

        file_children = set(pq.read_schema(file_path).field("clips").type.value_type.names)
        assert file_children
        names = [f"bulk_struct_{uuid4().hex}" for _ in range(2)]
        for name in names:
            assert not client.has_collection(name)
            client.create_collection(name, schema=schema, consistency_level="Strong")
            owned.append(name)
            client.add_collection_field(name, "clips[added_number]", DataType.INT64, nullable=True)
            for child in sorted(file_children):
                client.drop_collection_field(name, field_name=f"clips[{child}]")
            current = CollectionSchema.construct_from_dict(client.describe_collection(name))
            clips = next(field for field in current.struct_fields if field.name == "clips")
            assert {field.name for field in clips.fields} == {"added_number"}
            assert file_children.isdisjoint(field.name for field in clips.fields)

        wait_for_import(names[0], file_path, {})
        expected = {
            row["id"]: (
                None if row["clips"] is None else [{"added_number": None} for _ in row["clips"]]
            )
            for row in rows
        }
        client.insert(
            names[1],
            [{**row, "clips": expected[row["id"]]} for row in rows],
        )
        client.flush(names[1])
        for name in names:
            index = client.prepare_index_params()
            index.add_index("anchor", index_type="FLAT", metric_type="L2")
            client.create_index(name, index)
            client.load_collection(name)
            assert queried(client, name, ["clips"]) == expected
            assert queried(client, name, ["clips[added_number]"]) == expected
    finally:
        try:
            for name in owned:
                client.drop_collection(name)
        finally:
            client.close()
