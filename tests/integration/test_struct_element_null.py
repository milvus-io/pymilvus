"""Set MILVUS_ELEMENT_NULL_URI after obtaining the new-cluster address."""

import asyncio
import inspect
import logging
import os
import time
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
from pymilvus.client.prepare import Prepare
from pymilvus.client.utils import check_status
from pymilvus.exceptions import MilvusException
from pymilvus.orm.schema import StructFieldSchema

LOGGER = logging.getLogger(__name__)
URI = os.environ.get("MILVUS_ELEMENT_NULL_URI")
pytestmark = pytest.mark.skipif(not URI, reason="MILVUS_ELEMENT_NULL_URI is not configured")


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


def collection_schema(nullable=True):
    clips = StructFieldSchema()
    clips.add_field("v", DataType.FLOAT_VECTOR, dim=2, nullable=True)
    clips.add_field("score", DataType.INT64, nullable=True)
    clips.add_field(
        "tags",
        DataType.ARRAY,
        element_type=DataType.INT64,
        max_capacity=8,
        nullable=True,
        element_nullable=True,
    )
    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema("marker", DataType.INT64, nullable=True),
        ]
    )
    schema.add_field(
        "clips",
        DataType.ARRAY,
        element_type=DataType.STRUCT,
        struct_schema=clips,
        max_capacity=16,
        nullable=nullable,
    )
    return schema


def sample_clips():
    return [
        {"v": [1.0, 0.0], "score": 1, "tags": [10, None]},
        {"v": None, "score": None, "tags": None},
        {"v": [3.0, 0.0], "score": 3, "tags": []},
    ]


async def queried(client, name, ids):
    rows = await call(client.query, name, filter=f"id in {ids}", output_fields=["id", "clips"])
    return {row["id"]: row["clips"] for row in rows}


async def compact_and_reload(client, inspector, name):
    await call(client.flush, name)
    job = await call(client.compact, name)
    deadline = time.monotonic() + 120
    while await call(client.get_compaction_state, job) != "Completed":
        if time.monotonic() >= deadline:
            pytest.fail(f"Compaction {job} did not complete within 120 seconds")
        await asyncio.sleep(0.5)
    # The existing async handler does not implement get_compaction_plans.
    plans = inspector.get_compaction_plans(job)
    assert any(
        plan.sources and plan.targets for plan in plans.plans
    ), "Compaction produced no output segment"
    await call(client.release_collection, name)
    await call(client.load_collection, name)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [MilvusClient, AsyncMilvusClient])
@pytest.mark.parametrize("nullable", [False, True])
async def test_struct_nulls_search_evolution(client_type, nullable):
    name = "struct_null_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        schema = collection_schema(nullable)
        index = MilvusClient.prepare_index_params()
        index.add_index("clips[v]", index_type="FLAT", metric_type="L2")
        await call(
            client.create_collection,
            name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )
        described = await call(client.describe_collection, name)
        restored = CollectionSchema.construct_from_dict(described)
        restored_tags = restored.struct_fields[0].fields[2]
        assert restored.struct_fields[0].nullable == nullable
        assert restored_tags.dtype == DataType.ARRAY
        assert restored_tags.element_type == DataType.INT64
        assert restored_tags.nullable and restored_tags.element_nullable
        assert restored_tags.params["max_capacity"] == 8
        assert restored.struct_fields[0].max_capacity == 16
        rows = [
            {"id": 2, "clips": []},
            {"id": 3, "clips": sample_clips()},
            {"id": 4, "clips": [{}, {}]},
        ]
        if nullable:
            rows.append({"id": 1, "clips": None})
        await call(client.insert, name, rows)
        expected = {
            2: [],
            3: sample_clips(),
            4: [{"v": None, "score": None, "tags": None} for _ in range(2)],
        }
        if nullable:
            expected[1] = None
        assert await queried(client, name, list(expected)) == expected
        await call(client.upsert, name, [{"id": 3, "clips": sample_clips()}])
        replacement = [{"v": [4.0, 0.0], "score": None, "tags": [None, 40]}]
        await call(client.upsert, name, [{"id": 3, "clips": replacement}], partial_update=True)
        assert (await queried(client, name, [3]))[3] == replacement
        await call(client.upsert, name, [{"id": 3, "clips": sample_clips()}], partial_update=True)
        with pytest.raises(MilvusException):
            await call(client.insert, name, [{"id": 8, "clips": [None]}])
        with pytest.raises(MilvusException):
            await call(client.insert, name, [{"id": 8, "clips": [{"unknown": 1}]}])
        with pytest.raises(MilvusException):
            await call(client.add_collection_field, name, "clips[required]", DataType.INT64)

        matches = await call(
            client.query, name, filter="MATCH_ANY(clips, $[score] IS NULL)", output_fields=["id"]
        )
        assert {row["id"] for row in matches} == {3, 4}
        matches = await call(
            client.query,
            name,
            filter="id in [3, 4] and MATCH_ALL(clips, $[score] IS NULL)",
            output_fields=["id"],
        )
        assert {row["id"] for row in matches} == {4}
        matches = await call(
            client.query,
            name,
            filter="MATCH_ANY(clips, array_contains($[tags], 10))",
            output_fields=["id"],
        )
        assert {row["id"] for row in matches} == {3}
        matches = await call(
            client.query,
            name,
            filter="MATCH_ANY(clips, array_length($[tags]) == 2)",
            output_fields=["id"],
        )
        assert {row["id"] for row in matches} == {3}
        matches = await call(
            client.query,
            name,
            filter="element_filter(clips, $[score] IS NULL)",
            output_fields=["id", "clips"],
        )
        assert {(row["id"], row["offset"]) for row in matches} == {(3, 1), (4, 0), (4, 1)}
        for row in matches:
            assert row["clips"] == expected[row["id"]]
        scalar_index = MilvusClient.prepare_index_params()
        scalar_index.add_index("clips[score]", index_type="INVERTED")
        with pytest.raises(MilvusException):
            await call(client.create_index, name, scalar_index)

        async def check_search():
            found = await call(
                client.search,
                name,
                [[3.0, 0.0], [1.0, 0.0]],
                anns_field="clips[v]",
                search_params={"metric_type": "L2"},
                limit=2,
                output_fields=["clips"],
            )
            assert [[hit["offset"] for hit in query] for query in found] == [[2, 0], [0, 2]]
            assert all(
                hit["id"] == 3 and hit["entity"]["clips"] == expected[3]
                for query in found
                for hit in query
            )

        await check_search()
        request = AnnSearchRequest([[3.0, 0.0]], "clips[v]", {"metric_type": "L2"}, 2)
        found = await call(
            client.hybrid_search,
            name,
            [request, request],
            RRFRanker(),
            limit=2,
            output_fields=["clips"],
        )
        assert {(hit["id"], hit["offset"]) for hit in found[0]} == {(3, 0), (3, 2)}
        await call(client.flush, name)
        assert await queried(client, name, list(expected)) == expected
        await check_search()

        await call(client.insert, name, [{"id": 9, "clips": sample_clips()}])
        expected[9] = sample_clips()

        old_schema = inspector._get_connection().describe_collection(name)
        await call(
            client.add_collection_field,
            name,
            "clips[lang]",
            DataType.VARCHAR,
            max_length=16,
            nullable=True,
        )
        await call(
            client.add_collection_field,
            name,
            "clips[more]",
            DataType.ARRAY,
            element_type=DataType.INT64,
            max_capacity=4,
            nullable=True,
            element_nullable=True,
        )
        for clips in expected.values():
            for clip in clips or []:
                clip.update(lang=None, more=None)
        assert await queried(client, name, list(expected)) == expected

        stale = Prepare.row_insert_param(
            name,
            [{"id": 5, "clips": sample_clips()}],
            "",
            old_schema["fields"],
            old_schema["struct_array_fields"],
        )
        assert (
            len(next(f for f in stale.fields_data if f.field_name == "clips").struct_arrays.fields)
            == 3
        )
        check_status(inspector._get_connection()._stub.Insert(stale).status)
        assert (await queried(client, name, [5]))[5] == expected[3]
        await call(client.insert, name, [{"id": 6, "clips": sample_clips()}])
        assert (await queried(client, name, [6]))[6] == expected[3]

        await compact_and_reload(client, inspector, name)
        assert await queried(client, name, list(expected)) == expected
        assert await queried(client, name, [5, 6]) == {5: expected[3], 6: expected[3]}
        compacted_hits = await call(
            client.search,
            name,
            [[3.0, 0.0]],
            anns_field="clips[v]",
            search_params={"metric_type": "L2"},
            filter="id == 3",
            limit=2,
            output_fields=["clips"],
        )
        assert [hit["offset"] for hit in compacted_hits[0]] == [2, 0]
        assert all(hit["entity"]["clips"] == expected[3] for hit in compacted_hits[0])

        await call(
            client.add_collection_field,
            name,
            "clips[v2]",
            DataType.FLOAT_VECTOR,
            dim=2,
            nullable=True,
        )
        added_index = MilvusClient.prepare_index_params()
        added_index.add_index("clips[v2]", index_type="FLAT", metric_type="L2")
        await call(client.create_index, name, added_index)
        await call(client.load_collection, name, refresh=True)
        for clips in expected.values():
            for clip in clips or []:
                clip["v2"] = None
        assert await queried(client, name, list(expected)) == expected
        await call(client.drop_collection_field, name, field_name="clips[v]")
        for clips in expected.values():
            for clip in clips or []:
                clip.pop("v")
        assert await queried(client, name, list(expected)) == expected
        with pytest.raises(MilvusException):
            check_status(inspector._get_connection()._stub.Insert(stale).status)
        await call(client.drop_collection_field, name, field_name="clips[lang]")
        await call(client.add_collection_field, name, "clips[lang]", DataType.INT64, nullable=True)
        assert await queried(client, name, list(expected)) == expected
        await call(
            client.insert,
            name,
            [{"id": 7, "clips": [{}, {"v2": [7.0, 0.0], "lang": 7, "more": [None, 7]}]}],
        )
        values = (await queried(client, name, [7]))[7]
        assert values[0] == {"v2": None, "score": None, "tags": None, "lang": None, "more": None}
        value = values[1]
        assert value == {
            "v2": [7.0, 0.0],
            "score": None,
            "tags": None,
            "lang": 7,
            "more": [None, 7],
        }
        found = await call(
            client.search,
            name,
            [[7.0, 0.0]],
            anns_field="clips[v2]",
            search_params={"metric_type": "L2"},
            filter="id == 7",
            limit=3,
            output_fields=["clips"],
        )
        assert [(hit["id"], hit["offset"]) for hit in found[0]] == [(7, 1)]
        assert found[0][0]["entity"]["clips"] == values
        matched = await call(
            client.query,
            name,
            filter="element_filter(clips, $[lang] == 7)",
            output_fields=["id", "clips"],
        )
        assert [(row["id"], row["offset"]) for row in matched] == [(7, 1)]
        assert matched[0]["clips"] == values
        await call(client.upsert, name, [{"id": 7, "clips": []}])
        assert (await queried(client, name, [7]))[7] == []
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
@pytest.mark.parametrize("nullable", [False, True])
async def test_partial_upsert_preserves_omitted_struct(client_type, nullable, monkeypatch):
    name = "struct_partial_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        index = MilvusClient.prepare_index_params()
        index.add_index("clips[v]", index_type="FLAT", metric_type="L2")
        await call(
            client.create_collection,
            name,
            schema=collection_schema(nullable),
            index_params=index,
            consistency_level="Strong",
        )
        await call(client.insert, name, [{"id": 3, "clips": sample_clips()}])
        query = Prepare.query_request(
            name, "id == 3", ["clips"], [], use_default_consistency=False, consistency_level=0
        )
        response = inspector._get_connection()._stub.Query(query, timeout=30)
        check_status(response.status)
        LOGGER.info("WIRE_EXISTING_QUERY collection=%s\n%s", name, response)
        conn = client._get_connection()
        if inspect.isawaitable(conn):
            conn = await conn
        if isinstance(client, AsyncMilvusClient):
            original = conn._async_stub.Upsert

            async def record_upsert(request, *args, **kwargs):
                assert not any(field.field_name == "clips" for field in request.fields_data)
                LOGGER.info("WIRE_REQUEST Upsert collection=%s\n%s", name, request)
                response = await original(request, *args, **kwargs)
                LOGGER.info("WIRE_RESPONSE Upsert collection=%s\n%s", name, response)
                if response.status.code:
                    replay = await original(request, *args, **kwargs)
                    LOGGER.info("WIRE_DIRECT_REPLAY Upsert collection=%s\n%s", name, replay)
                return response

            monkeypatch.setattr(conn._async_stub, "Upsert", record_upsert)
        else:
            original = conn._stub.Upsert

            def record_upsert(request, *args, **kwargs):
                assert not any(field.field_name == "clips" for field in request.fields_data)
                LOGGER.info("WIRE_REQUEST Upsert collection=%s\n%s", name, request)
                response = original(request, *args, **kwargs)
                LOGGER.info("WIRE_RESPONSE Upsert collection=%s\n%s", name, response)
                if response.status.code:
                    replay = original(request, *args, **kwargs)
                    LOGGER.info("WIRE_DIRECT_REPLAY Upsert collection=%s\n%s", name, replay)
                return response

            monkeypatch.setattr(conn._stub, "Upsert", record_upsert)
        await call(client.upsert, name, [{"id": 3, "marker": 99}], partial_update=True)
        assert (await queried(client, name, [3]))[3] == sample_clips()
        assert await call(client.query, name, filter="id == 3", output_fields=["id", "marker"]) == [
            {"id": 3, "marker": 99}
        ]
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
@pytest.mark.parametrize("metric", ["IP", "COSINE"])
async def test_nullable_vectors_with_other_metrics(client_type, metric):
    name = "struct_metric_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        index = MilvusClient.prepare_index_params()
        index.add_index("clips[v]", index_type="FLAT", metric_type=metric)
        await call(
            client.create_collection,
            name,
            schema=collection_schema(False),
            index_params=index,
            consistency_level="Strong",
        )
        clips = [
            {"v": None, "score": None, "tags": None},
            {"v": [1.0, 0.0], "score": None, "tags": None},
            {"v": [0.0, 1.0], "score": None, "tags": None},
            {"v": None, "score": None, "tags": None},
        ]
        await call(client.insert, name, [{"id": 1, "clips": clips}])
        for sealed in (False, True):
            if sealed:
                await call(client.flush, name)
            found = await call(
                client.search,
                name,
                [[1.0, 0.0]],
                anns_field="clips[v]",
                search_params={"metric_type": metric},
                limit=4,
                output_fields=["clips"],
            )
            assert [(hit["id"], hit["offset"]) for hit in found[0]] == [(1, 1), (1, 2)]
            assert all(hit["entity"]["clips"] == clips for hit in found[0])
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
@pytest.mark.parametrize("leaf_nullable", [False, True])
async def test_required_children_and_nested_leaf_nulls(client_type, leaf_nullable):
    name = "struct_required_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        struct = StructFieldSchema()
        struct.add_field("v", DataType.FLOAT_VECTOR, dim=2, nullable=True)
        struct.add_field("score", DataType.INT64)
        struct.add_field(
            "tags",
            DataType.ARRAY,
            element_type=DataType.INT64,
            max_capacity=4,
            nullable=True,
            element_nullable=leaf_nullable,
        )
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
        index.add_index("clips[v]", index_type="FLAT", metric_type="L2")
        await call(
            client.create_collection,
            name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )
        clips = [
            {"v": None, "score": 1, "tags": None},
            {"v": [1.0, 0.0], "score": 2, "tags": []},
            {"v": None, "score": 3, "tags": [None, None] if leaf_nullable else [1, 2]},
        ]
        await call(
            client.insert,
            name,
            [{"id": 1, "clips": None}, {"id": 2, "clips": []}, {"id": 3, "clips": clips}],
        )
        for invalid in ({"score": None}, {}, {"score": 1, "tags": [1, 2, 3, 4, 5]}):
            with pytest.raises(MilvusException):
                await call(client.insert, name, [{"id": 4, "clips": [invalid]}])
        if not leaf_nullable:
            with pytest.raises(MilvusException):
                await call(
                    client.insert, name, [{"id": 4, "clips": [{"score": 1, "tags": [None]}]}]
                )
        assert await queried(client, name, [1, 2, 3]) == {1: None, 2: [], 3: clips}
        await call(client.flush, name)
        assert await queried(client, name, [1, 2, 3]) == {1: None, 2: [], 3: clips}
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
async def test_later_added_struct_preserves_null_rows_when_child_is_added(client_type):
    name = "struct_added_" + uuid4().hex
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
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=2),
            ]
        )
        index = MilvusClient.prepare_index_params()
        index.add_index("vector", index_type="FLAT", metric_type="L2")
        await call(
            client.create_collection,
            name,
            schema=schema,
            index_params=index,
            consistency_level="Strong",
        )
        await call(client.insert, name, [{"id": 1, "vector": [1.0, 0.0]}])
        clips = collection_schema().struct_fields[0]
        await call(
            client.add_collection_struct_field, name, "clips", clips, max_capacity=16, nullable=True
        )
        index = MilvusClient.prepare_index_params()
        index.add_index("clips[v]", index_type="FLAT", metric_type="L2")
        await call(client.create_index, name, index)
        await call(client.insert, name, [{"id": 2, "vector": [2.0, 0.0], "clips": sample_clips()}])
        await call(client.add_collection_field, name, "clips[lang]", DataType.INT64, nullable=True)
        expected = sample_clips()
        for clip in expected:
            clip["lang"] = None
        assert await queried(client, name, [1, 2]) == {1: None, 2: expected}
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
async def test_other_vector_types_keep_null_offsets(
    client_type, dtype, value, dim, expected, index_type, metric
):
    name = "struct_vector_" + uuid4().hex
    client = client_type(uri=URI, timeout=30)
    inspector = MilvusClient(uri=URI, timeout=30)
    owned = False
    LOGGER.info("COLLECTION_START uri=%s name=%s", URI, name)
    try:
        assert not inspector.has_collection(name), f"Refusing existing collection {name}"
        owned = True
        struct = StructFieldSchema()
        struct.add_field("v", dtype, dim=dim, nullable=True)
        schema = CollectionSchema([FieldSchema("id", DataType.INT64, is_primary=True)])
        schema.add_field(
            "clips",
            DataType.ARRAY,
            element_type=DataType.STRUCT,
            struct_schema=struct,
            max_capacity=8,
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
        await call(client.insert, name, [{"id": 1, "clips": [{"v": None}, {"v": value}, {}]}])
        assert (await queried(client, name, [1]))[1] == [{"v": None}, {"v": expected}, {"v": None}]
        found = await call(
            client.search,
            name,
            [value],
            anns_field="clips[v]",
            search_params={
                "metric_type": metric,
                "params": {"ef": 8} if index_type == "HNSW" else {},
            },
            limit=3,
        )
        assert [(hit["id"], hit["offset"]) for hit in found[0]] == [(1, 1)]
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
