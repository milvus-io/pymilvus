"""Run with MILVUS_RLS_URI and an admin MILVUS_TOKEN against an RLS-capable server.

Set MILVUS_RLS_AUTH=true for RBAC checks. For embedded local storage, set
MILVUS_RLS_IMPORT_DIR to an existing directory shared with the server, with
proxy.rls.importEnforcementEnabled enabled after all components support RLS.
"""

import inspect
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import pytest
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    Collection,
    DataType,
    MilvusClient,
    RRFRanker,
    connections,
    utility,
)
from pymilvus.bulk_writer import bulk_import
from pymilvus.client.types import BulkInsertState
from pymilvus.exceptions import MilvusException

URI = os.environ.get("MILVUS_RLS_URI")
TOKEN = os.environ.get("MILVUS_TOKEN", "")
pytestmark = pytest.mark.skipif(not URI, reason="MILVUS_RLS_URI is not configured")

ACTIONS = [
    "query",
    "query_iterator",
    "search",
    "search_iterator",
    "hybrid_search",
    "insert",
    "upsert",
    "delete",
]
OWNER_POLICY = {
    "policy_type": "permissive",
    "actions": ACTIONS,
    "using_expr": "owner == $current_principal",
    "check_expr": "owner == $current_principal",
}


def row(pk, owner):
    return {"id": pk, "owner": owner, "vector": [float(pk), 0.0]}


def ids(rows):
    return sorted(item["id"] for item in rows)


async def call(method, *args, **kwargs):
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


@pytest.fixture
def collection():
    admin = MilvusClient(uri=URI, token=TOKEN, timeout=30)
    db = "rls_sdk_" + uuid4().hex
    admin.create_database(db)
    client = MilvusClient(uri=URI, token=TOKEN, db_name=db, timeout=30)
    name = "documents"
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("owner", DataType.VARCHAR, max_length=100)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        index = client.prepare_index_params("vector", index_type="FLAT", metric_type="L2")
        client.create_collection(
            name,
            schema=schema,
            index_params=index,
            properties={"rls.enabled": "true"},
            consistency_level="Strong",
        )
        yield client, db, name
    finally:
        try:
            client.drop_collection(name)
            admin.drop_database(db)
        finally:
            client.close()
            admin.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("client_type", [MilvusClient, AsyncMilvusClient])
async def test_management_and_enforcement(collection, client_type):
    _, db, name = collection
    client = client_type(uri=URI, token=TOKEN, db_name=db, timeout=30)
    management = client_type(uri=URI, token=TOKEN, timeout=30)
    try:
        # Override a default-database client's management target on the real wire.
        await call(management.create_row_policy, name, "owner", db_name=db, **OWNER_POLICY)
        with pytest.raises(MilvusException, match="already exist"):
            await call(management.create_row_policy, name, "owner", db_name=db, **OWNER_POLICY)
        await call(
            management.update_row_policy,
            name,
            "owner",
            db_name=db,
            description="Owner access",
            **OWNER_POLICY,
        )
        policies = await call(management.list_row_policies, name, db_name=db)
        assert len(policies) == 1
        assert policies[0]["description"] == "Owner access"
        assert set(policies[0]["actions"]) == {
            "Query",
            "QueryIterator",
            "Search",
            "SearchIterator",
            "HybridSearch",
            "Insert",
            "Upsert",
            "Delete",
        }
        tags = {
            "label": "研发",
            "integer": 2**63 - 1,
            "double": 3.0,
            "owners": ["alice"],
            "empty": [],
        }
        await call(management.set_rls_principal_tags, name, "alice", tags, db_name=db)
        actual = await call(management.get_rls_principal_tags, name, "alice", db_name=db)
        assert actual == tags
        assert type(actual["integer"]) is int
        assert type(actual["double"]) is float
        assert await call(management.list_rls_principals, name, db_name=db) == ["alice"]
        with pytest.raises(MilvusException):
            await call(
                management.set_rls_principal_tags, name, "alice", {"bad": [1, "x"]}, db_name=db
            )
        assert await call(management.get_rls_principal_tags, name, "alice", db_name=db) == tags

        for pk, owner in [(1, "alice"), (2, "bob")]:
            await call(client.insert, name, [row(pk, owner)], rls_principal=owner)
        for principal, expected in [("alice", [1]), ("bob", [2])]:
            assert (
                ids(await call(client.query, name, filter="id >= 0", rls_principal=principal))
                == expected
            )
            assert ids(await call(client.get, name, [1, 2], rls_principal=principal)) == expected
            found = await call(client.search, name, [[0.0, 0.0]], limit=10, rls_principal=principal)
            assert ids(found[0]) == expected
            request = AnnSearchRequest([[0.0, 0.0]], "vector", {"metric_type": "L2"}, 10)
            found = await call(
                client.hybrid_search,
                name,
                [request],
                RRFRanker(),
                limit=10,
                rls_principal=principal,
            )
            assert ids(found[0]) == expected
        for method, args in [(client.query, {"filter": "id >= 0"}), (client.delete, {"ids": [1]})]:
            with pytest.raises(MilvusException, match="rls_principal"):
                await call(method, name, **args)

        # A rejected batch must not insert its otherwise valid first row.
        with pytest.raises(MilvusException, match="denied by RLS"):
            await call(client.insert, name, [row(3, "alice"), row(4, "bob")], rls_principal="alice")
        with pytest.raises(MilvusException, match="denied by RLS"):
            await call(client.upsert, name, [row(2, "alice")], rls_principal="alice")
        assert ids(await call(client.query, name, filter="id >= 0", skip_rls=True)) == [1, 2]
        await call(client.upsert, name, [row(1, "alice"), row(3, "alice")], rls_principal="alice")
        await call(client.delete, name, ids=[1, 2], rls_principal="alice")
        assert ids(await call(client.query, name, filter="id >= 0", skip_rls=True)) == [2, 3]
        await call(client.flush, name)
        assert ids(await call(client.query, name, filter="id >= 0", rls_principal="alice")) == [3]

        # Policy/tag changes must affect the next request without client recreation.
        await call(
            management.update_row_policy,
            name,
            "owner",
            db_name=db,
            policy_type="permissive",
            actions=ACTIONS,
            using_expr="owner == $current_principal_tags['owner']",
            check_expr="owner == $current_principal_tags['owner']",
        )
        await call(management.set_rls_principal_tags, name, "alice", {"owner": "bob"}, db_name=db)
        assert ids(await call(client.query, name, filter="id >= 0", rls_principal="alice")) == [2]
        await call(management.delete_rls_principal_tags, name, "alice", ["owner"], db_name=db)
        assert await call(client.query, name, filter="id >= 0", rls_principal="alice") == []
        await call(management.delete_rls_principal_tags, name, "alice", db_name=db)
        with pytest.raises(MilvusException, match="does not exist"):
            await call(management.get_rls_principal_tags, name, "alice", db_name=db)
        assert await call(management.list_rls_principals, name, db_name=db) == []
        await call(management.drop_row_policy, name, "owner", db_name=db)
        assert await call(management.list_row_policies, name, db_name=db) == []
        with pytest.raises(MilvusException, match="denied by RLS"):
            await call(client.query, name, filter="id >= 0", rls_principal="alice")
        with pytest.raises(MilvusException, match="denied by RLS"):
            await call(client.insert, name, [row(5, "alice")], rls_principal="alice")
        await call(client.alter_collection_properties, name, {"rls.force": "true"})
        with pytest.raises(MilvusException, match=r"rls\.force"):
            await call(client.query, name, filter="id >= 0", skip_rls=True)
    finally:
        await call(client.close)
        await call(management.close)


@pytest.mark.parametrize("kind", ["query", "search"])
def test_iterator_pages(collection, kind):
    client, _, name = collection
    client.create_row_policy(name, "owner", **OWNER_POLICY)
    for owner, keys in [("alice", [1, 3, 5]), ("bob", [2, 4, 6])]:
        client.insert(name, [row(pk, owner) for pk in keys], rls_principal=owner)
    client.flush(name)
    options = {"batch_size": 1, "output_fields": ["id"], "rls_principal": "alice"}
    iterator = (
        client.query_iterator(name, filter="id >= 0", **options)
        if kind == "query"
        else client.search_iterator(name, [[0.0, 0.0]], **options)
    )
    found = []
    try:
        while batch := iterator.next():
            found.extend(item["id"] if kind == "query" else item.id for item in batch)
    finally:
        iterator.close()
    assert sorted(found) == [1, 3, 5]


def test_orm_column_writes(collection):
    client, db, name = collection
    client.create_row_policy(name, "owner", **OWNER_POLICY)
    alias = "rls_" + uuid4().hex
    connections.connect(alias=alias, uri=URI, token=TOKEN, db_name=db)
    try:
        orm = Collection(name, using=alias)
        orm.insert([[1, 2], ["alice", "alice"], [[1.0, 0.0], [2.0, 0.0]]], rls_principal="alice")
        orm.upsert([[1], ["alice"], [[3.0, 0.0]]], rls_principal="alice")
        with pytest.raises(MilvusException, match="denied by RLS"):
            orm.insert([[3], ["bob"], [[3.0, 0.0]]], rls_principal="alice")
        with pytest.raises(MilvusException, match="denied by RLS"):
            orm.upsert([[1], ["bob"], [[3.0, 0.0]]], rls_principal="alice")
        assert ids(client.query(name, filter="id >= 0", rls_principal="alice")) == [1, 2]
    finally:
        connections.disconnect(alias)


@pytest.mark.skipif(
    os.environ.get("MILVUS_RLS_AUTH") != "true", reason="Requires auth and admin token"
)
def test_skip_requires_privilege(collection):
    client, db, name = collection
    client.create_row_policy(name, "owner", **OWNER_POLICY)
    client.insert(name, [row(1, "alice")], rls_principal="alice")
    user = "rls_user_" + uuid4().hex[:16]
    role = "rls_role_" + uuid4().hex[:16]
    password = uuid4().hex
    client.create_user(user, password)
    try:
        client.create_role(role)
        try:
            client.grant_privilege_v2(role, "Query", name, db_name=db)
            client.grant_role(user, role)
            reader = MilvusClient(uri=URI, token=f"{user}:{password}", db_name=db, timeout=30)
            try:
                # The application principal is independent of the Milvus login user.
                assert ids(reader.query(name, filter="id >= 0", rls_principal="alice")) == [1]
                with pytest.raises(MilvusException, match="requires SkipRLS privilege"):
                    reader.query(name, filter="id >= 0", skip_rls=True)
                client.grant_privilege_v2(role, "SkipRLS", name, db_name=db)
                assert ids(reader.query(name, filter="id >= 0", skip_rls=True)) == [1]
                client.alter_collection_properties(name, {"rls.force": "true"})
                with pytest.raises(MilvusException, match=r"rls\.force"):
                    reader.query(name, filter="id >= 0", skip_rls=True)
            finally:
                reader.close()
                client.revoke_role(user, role)
        finally:
            client.drop_role(role, force_drop=True)
    finally:
        client.drop_user(user)


@pytest.mark.skipif(
    not os.environ.get("MILVUS_RLS_IMPORT_DIR"), reason="Requires shared local storage"
)
@pytest.mark.parametrize("transport", ["grpc", "rest"])
def test_bulk_import(collection, transport):
    client, db, name = collection
    client.create_row_policy(name, "owner", **OWNER_POLICY)
    alias = "rls_import_" + uuid4().hex
    connections.connect(alias=alias, uri=URI, token=TOKEN, db_name=db)
    path = Path(os.environ["MILVUS_RLS_IMPORT_DIR"]) / (uuid4().hex + ".json")
    try:
        for owner, skip, expected_state in [
            ("alice", False, BulkInsertState.ImportCompleted),
            ("bob", False, BulkInsertState.ImportFailed),
            ("bob", True, BulkInsertState.ImportCompleted),
        ]:
            path.write_text(json.dumps({"rows": [row(1 if owner == "alice" else 2, owner)]}))
            if transport == "grpc":
                task_id = utility.do_bulk_insert(
                    name, [str(path)], using=alias, rls_principal="alice", skip_rls=skip, timeout=30
                )
            else:
                response = bulk_import(
                    URI,
                    name,
                    db_name=db,
                    files=[[str(path)]],
                    api_key=TOKEN,
                    options={"rls_principal": "alice", "skip_rls": str(skip).lower()},
                    timeout=30,
                )
                task_id = int(response.json()["data"]["jobId"])
            deadline = time.monotonic() + 180
            while True:
                state = utility.get_bulk_insert_state(task_id, using=alias, timeout=30)
                if state.state in (BulkInsertState.ImportCompleted, BulkInsertState.ImportFailed):
                    break
                assert time.monotonic() < deadline, state
                time.sleep(0.5)
            assert state.state == expected_state, state
            if expected_state == BulkInsertState.ImportFailed:
                assert "denied by RLS" in state.failed_reason
            client.refresh_load(name)
            assert ids(client.query(name, filter="id >= 0", skip_rls=True)) == (
                [1, 2] if skip else [1]
            )
        assert len(utility.list_bulk_insert_tasks(collection_name=name, using=alias)) == 3
        assert ids(client.query(name, filter="id >= 0", rls_principal="alice")) == [1]
    finally:
        connections.disconnect(alias)
        path.unlink(missing_ok=True)
