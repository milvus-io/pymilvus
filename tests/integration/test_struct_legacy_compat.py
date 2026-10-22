"""Set MILVUS_LEGACY_STRUCT_URI to a supplied Milvus 3.0 cluster."""

import inspect
import logging
import os
from uuid import uuid4

import numpy as np
import pytest
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
)
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import StructFieldSchema

LOGGER = logging.getLogger(__name__)
URI = os.environ.get("MILVUS_LEGACY_STRUCT_URI")
pytestmark = pytest.mark.skipif(not URI, reason="MILVUS_LEGACY_STRUCT_URI is not configured")


async def call(method, *args, **kwargs):
    LOGGER.info("CALL %s args=%r kwargs=%r", method.__qualname__, args, kwargs)
    try:
        result = method(*args, **kwargs)
        result = await result if inspect.isawaitable(result) else result
    except Exception as exc:
        LOGGER.info("CALL_FAILED %s: %s: %s", method.__qualname__, type(exc).__name__, exc)
        raise
    LOGGER.info("CALL_OK %s", method.__qualname__)
    return result


def legacy_schema():
    struct = StructFieldSchema()
    struct.add_field("v", DataType.FLOAT_VECTOR, dim=2)
    struct.add_field("score", DataType.INT64)
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("number", DataType.INT64, nullable=True),
            FieldSchema("defaulted", DataType.INT64, default_value=7),
            FieldSchema(
                "array", DataType.ARRAY, element_type=DataType.INT64, max_capacity=8, nullable=True
            ),
            FieldSchema("json", DataType.JSON, nullable=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2, nullable=True),
        ]
    )
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=struct,
        max_capacity=16,
        nullable=True,
    )
    return schema


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [MilvusClient, AsyncMilvusClient])
async def test_legacy_struct_and_ordinary_nullable(client_type):
    name = "struct_legacy_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        index = MilvusClient.prepare_index_params()
        index.add_index("vector", index_type="FLAT", metric_type="L2")
        index.add_index("clips[v]", index_type="FLAT", metric_type="L2")
        await call(
            client.create_collection,
            name,
            schema=legacy_schema(),
            index_params=index,
            consistency_level="Strong",
        )
        described = await call(client.describe_collection, name)
        restored = CollectionSchema.construct_from_dict(described)
        assert restored.struct_fields[0].nullable
        assert all(
            not field.nullable and not field.element_nullable
            for field in restored.struct_fields[0].fields
        )
        clips = [{"v": [1.0, 0.0], "score": 1}, {"v": [3.0, 0.0], "score": 3}]
        rows = [
            {"id": 1, "number": None, "array": None, "json": None, "vector": None, "clips": None},
            {"id": 2, "number": 2, "array": [], "json": {}, "vector": [2.0, 0.0], "clips": []},
            {
                "id": 3,
                "number": 3,
                "array": [1, 2],
                "json": {"a": 1},
                "vector": [3.0, 0.0],
                "clips": clips,
            },
            {
                "id": 4,
                "number": None,
                "array": None,
                "json": None,
                "vector": None,
                "clips": [{"v": [4.0, 0.0], "score": 4}],
            },
        ]
        await call(client.insert, name, rows)
        await call(client.upsert, name, [rows[2]])
        result = await call(
            client.query,
            name,
            filter="id >= 0",
            output_fields=["id", "number", "array", "json", "vector", "clips"],
        )
        assert sorted(result, key=lambda row: row["id"]) == rows
        defaults = await call(
            client.query, name, filter="id >= 0", output_fields=["id", "defaulted"]
        )
        assert len(defaults) == len(rows) and all(row["defaulted"] == 7 for row in defaults)
        found = await call(
            client.search,
            name,
            [[3.0, 0.0]],
            anns_field="clips[v]",
            search_params={"metric_type": "L2"},
            filter="id == 3",
            limit=2,
            output_fields=["clips"],
        )
        assert [(hit["id"], hit["offset"]) for hit in found[0]] == [(3, 1), (3, 0)]
        assert all(hit["entity"]["clips"] == clips for hit in found[0])
        matched = await call(
            client.query, name, filter="MATCH_ANY(clips, $[score] > 1)", output_fields=["id"]
        )
        assert {row["id"] for row in matched} == {3, 4}
        matched = await call(
            client.query, name, filter="element_filter(clips, $[score] > 1)", output_fields=["id"]
        )
        assert {(row["id"], row["offset"]) for row in matched} == {(3, 1), (4, 0)}
        request = AnnSearchRequest(
            [[3.0, 0.0]], "clips[v]", {"metric_type": "L2"}, 2, expr="id == 3"
        )
        found = await call(client.hybrid_search, name, [request, request], RRFRanker(), limit=2)
        assert {(hit["id"], hit["offset"]) for hit in found[0]} == {(3, 0), (3, 1)}

        async def check_nullable_outputs():
            outputs = ["number", "array", "json", "vector", "clips"]
            expected = {row["id"]: {key: row[key] for key in outputs} for row in rows}
            found = await call(
                client.search,
                name,
                [[3.0, 0.0]],
                anns_field="clips[v]",
                search_params={"metric_type": "L2"},
                limit=10,
                output_fields=outputs,
            )
            assert {hit["id"] for hit in found[0]} == {3, 4}
            for hit in found[0]:
                assert hit["entity"] == expected[hit["id"]]
            nullable_request = AnnSearchRequest([[3.0, 0.0]], "clips[v]", {"metric_type": "L2"}, 10)
            found = await call(
                client.hybrid_search,
                name,
                [nullable_request, nullable_request],
                RRFRanker(),
                limit=10,
                output_fields=outputs,
            )
            assert {hit["id"] for hit in found[0]} == {3, 4}
            for hit in found[0]:
                assert hit["entity"] == expected[hit["id"]]

        await check_nullable_outputs()
        await call(client.flush, name)
        result = await call(
            client.query,
            name,
            filter="id >= 0",
            output_fields=["id", "number", "array", "json", "vector", "clips"],
        )
        assert sorted(result, key=lambda row: row["id"]) == rows
        defaults = await call(
            client.query, name, filter="id >= 0", output_fields=["id", "defaulted"]
        )
        assert len(defaults) == len(rows) and all(row["defaulted"] == 7 for row in defaults)
        await check_nullable_outputs()
        await call(client.add_collection_field, name, "label", DataType.INT64, nullable=True)
        result = await call(client.query, name, filter="id >= 0", output_fields=["id", "label"])
        assert all(row["label"] is None for row in result)
        with pytest.raises(MilvusException):
            await call(
                client.add_collection_field,
                name,
                "clips[unsupported]",
                DataType.INT64,
                nullable=True,
            )
        fields = (await call(client.describe_collection, name))["fields"]
        assert not any(field["name"] == "unsupported" for field in fields)
        await call(
            client.add_collection_struct_field,
            name,
            "extra",
            legacy_schema().struct_fields[0],
            max_capacity=16,
            nullable=True,
        )
        extra_index = MilvusClient.prepare_index_params()
        extra_index.add_index("extra[v]", index_type="FLAT", metric_type="L2")
        await call(client.create_index, name, extra_index)
        await call(client.load_collection, name)
        added = await call(client.query, name, filter="id >= 0", output_fields=["id", "extra"])
        assert len(added) == len(rows) and all(row["extra"] is None for row in added)
        await call(client.drop_collection_field, name, field_name="label")
        assert not any(
            field["name"] == "label"
            for field in (await call(client.describe_collection, name))["fields"]
        )
    finally:
        try:
            if owned:
                if inspector.has_collection(name):
                    inspector.drop_collection(name)
                remaining = inspector.has_collection(name)
                LOGGER.info("COLLECTION_CLEANUP uri=%s name=%s exists=%s", URI, name, remaining)
                assert not remaining, f"Collection cleanup failed: {name}"
        finally:
            inspector.close()
            await call(client.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [MilvusClient, AsyncMilvusClient])
@pytest.mark.parametrize(
    "dtype,value,dim,expected,index_type,metric",
    [
        (DataType.BINARY_VECTOR, b"\x01", 8, [b"\x01"], "BIN_FLAT", "HAMMING"),
        (
            DataType.FLOAT16_VECTOR,
            np.array([1.0, 2.0], dtype="float16"),
            2,
            [1.0, 2.0],
            "FLAT",
            "L2",
        ),
        (DataType.BFLOAT16_VECTOR, b"\x80\x3f\x00\x40", 2, [16256, 16384], "FLAT", "L2"),
        (DataType.INT8_VECTOR, np.array([1, 2], dtype="int8"), 2, [1, 2], "HNSW", "L2"),
    ],
)
async def test_legacy_other_vector_types(
    client_type, dtype, value, dim, expected, index_type, metric
):
    name = "struct_legacy_vector_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        struct = StructFieldSchema()
        struct.add_field("v", dtype, dim=dim)
        struct.add_field("score", DataType.INT64)
        schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
        schema.add_field(
            "clips",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct,
            max_capacity=8,
            nullable=True,
        )
        index = MilvusClient.prepare_index_params()
        index.add_index(
            "clips[v]",
            index_type=index_type,
            metric_type=metric,
            params={"M": 8, "efConstruction": 32} if index_type == "HNSW" else {},
        )
        await call(
            client.create_collection,
            name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )
        clips = [{"v": value, "score": 1}, {"v": value, "score": 2}]
        await call(
            client.insert,
            name,
            [{"id": 1, "clips": None}, {"id": 2, "clips": []}, {"id": 3, "clips": clips}],
        )
        await call(client.upsert, name, [{"id": 3, "clips": clips}])
        expected_clips = [{"v": expected, "score": 1}, {"v": expected, "score": 2}]
        for sealed in (False, True):
            if sealed:
                await call(client.flush, name)
            result = await call(client.query, name, filter="id >= 0", output_fields=["id", "clips"])
            assert {row["id"]: row["clips"] for row in result} == {
                1: None,
                2: [],
                3: expected_clips,
            }
            found = await call(
                client.search,
                name,
                [value],
                anns_field="clips[v]",
                limit=3,
                search_params={
                    "metric_type": metric,
                    "params": {"ef": 8} if index_type == "HNSW" else {},
                },
                output_fields=["clips"],
            )
            assert {(hit["id"], hit["offset"]) for hit in found[0]} == {(3, 0), (3, 1)}
            assert all(hit["entity"]["clips"] == expected_clips for hit in found[0])
    finally:
        try:
            if owned:
                if inspector.has_collection(name):
                    inspector.drop_collection(name)
                remaining = inspector.has_collection(name)
                LOGGER.info("COLLECTION_CLEANUP uri=%s name=%s exists=%s", URI, name, remaining)
                assert not remaining, f"Collection cleanup failed: {name}"
        finally:
            inspector.close()
            await call(client.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [MilvusClient, AsyncMilvusClient])
@pytest.mark.parametrize(
    "dtype,value,dim,payload,index_type,metric",
    [
        (DataType.BINARY_VECTOR, b"\x01", 8, b"\x01", "BIN_FLAT", "HAMMING"),
        (
            DataType.FLOAT16_VECTOR,
            np.array([1.0, 2.0], dtype="float16"),
            2,
            b"\x00\x3c\x00\x40",
            "FLAT",
            "L2",
        ),
        (DataType.BFLOAT16_VECTOR, b"\x80\x3f\x00\x40", 2, b"\x80\x3f\x00\x40", "FLAT", "L2"),
        (DataType.INT8_VECTOR, np.array([1, 2], dtype="int8"), 2, b"\x01\x02", "HNSW", "L2"),
    ],
)
async def test_legacy_ordinary_nullable_vectors(
    client_type, dtype, value, dim, payload, index_type, metric
):
    name = "struct_legacy_nullable_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        schema = CollectionSchema(
            [
                FieldSchema("id", DataType.INT64, is_primary=True),
                FieldSchema("anchor", DataType.FLOAT_VECTOR, dim=2),
                FieldSchema("ordinary", dtype, dim=dim, nullable=True),
            ]
        )
        index = MilvusClient.prepare_index_params()
        index.add_index("anchor", index_type="FLAT", metric_type="L2")
        index.add_index(
            "ordinary",
            index_type=index_type,
            metric_type=metric,
            params={"M": 8, "efConstruction": 32} if index_type == "HNSW" else {},
        )
        await call(
            client.create_collection,
            name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )
        rows = [
            {"id": 1, "anchor": [1.0, 0.0], "ordinary": None},
            {"id": 2, "anchor": [2.0, 0.0], "ordinary": value},
        ]
        await call(client.insert, name, rows)
        await call(client.upsert, name, [rows[1]])
        for sealed in (False, True):
            if sealed:
                await call(client.flush, name)
            result = await call(
                client.query, name, filter="id >= 0", output_fields=["id", "ordinary"]
            )
            assert {row["id"]: row["ordinary"] for row in result} == {1: None, 2: [payload]}
            found = await call(
                client.search,
                name,
                [[2.0, 0.0]],
                anns_field="anchor",
                search_params={"metric_type": "L2"},
                limit=2,
                output_fields=["ordinary"],
            )
            assert {hit["id"]: hit["entity"]["ordinary"] for hit in found[0]} == {
                1: None,
                2: payload,
            }
            request = AnnSearchRequest([[2.0, 0.0]], "anchor", {"metric_type": "L2"}, 2)
            found = await call(
                client.hybrid_search,
                name,
                [request, request],
                RRFRanker(),
                limit=2,
                output_fields=["ordinary"],
            )
            assert {hit["id"]: hit["entity"]["ordinary"] for hit in found[0]} == {
                1: None,
                2: payload,
            }
            found = await call(
                client.search,
                name,
                [value],
                anns_field="ordinary",
                limit=2,
                search_params={
                    "metric_type": metric,
                    "params": {"ef": 8} if index_type == "HNSW" else {},
                },
            )
            assert [hit["id"] for hit in found[0]] == [2]
    finally:
        try:
            if owned:
                if inspector.has_collection(name):
                    inspector.drop_collection(name)
                remaining = inspector.has_collection(name)
                LOGGER.info("COLLECTION_CLEANUP uri=%s name=%s exists=%s", URI, name, remaining)
                assert not remaining, f"Collection cleanup failed: {name}"
        finally:
            inspector.close()
            await call(client.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [MilvusClient, AsyncMilvusClient])
async def test_legacy_embedding_list_search(client_type):
    name = "struct_legacy_emb_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        struct = StructFieldSchema()
        struct.add_field("v", DataType.FLOAT_VECTOR, dim=2)
        schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
        schema.add_field(
            "clips",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct,
            max_capacity=16,
        )
        index = MilvusClient.prepare_index_params()
        index.add_index(
            "clips[v]",
            index_type="HNSW",
            metric_type="MAX_SIM_IP",
            params={"M": 8, "efConstruction": 32},
        )
        await call(
            client.create_collection,
            name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )
        await call(
            client.insert, name, [{"id": 1, "clips": [{"v": [1.0, 0.0]}, {"v": [0.0, 1.0]}]}]
        )
        found = await call(
            client.search,
            name,
            [EmbeddingList([[1.0, 0.0]])],
            anns_field="clips[v]",
            search_params={"metric_type": "MAX_SIM_IP", "params": {"ef": 8}},
            limit=1,
            output_fields=["clips"],
        )
        assert found[0][0]["id"] == 1
        assert "offset" not in found[0][0]
        assert found[0][0]["entity"]["clips"] == [{"v": [1.0, 0.0]}, {"v": [0.0, 1.0]}]
    finally:
        try:
            if owned:
                if inspector.has_collection(name):
                    inspector.drop_collection(name)
                remaining = inspector.has_collection(name)
                LOGGER.info("COLLECTION_CLEANUP uri=%s name=%s exists=%s", URI, name, remaining)
                assert not remaining, f"Collection cleanup failed: {name}"
        finally:
            inspector.close()
            await call(client.close)
