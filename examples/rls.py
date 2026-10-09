"""Run against a Milvus server with RLS support:

    MILVUS_URI=http://localhost:19530 MILVUS_TOKEN=user:password python examples/rls.py

The example creates and removes its own collection. The authenticated Milvus
account needs collection and RLS management privileges. In applications, derive
rls_principal from trusted authentication, not from user-supplied request data.
"""

import asyncio
import os
from uuid import uuid4

from pymilvus import AsyncMilvusClient, DataType, MilvusClient
from pymilvus.exceptions import MilvusException


async def read_async(uri, token, collection):
    client = AsyncMilvusClient(uri=uri, token=token)
    try:
        rows = await client.query(collection, filter="id >= 0", rls_principal="alice")
        assert [row["id"] for row in rows] == [1]
    finally:
        await client.close()


def main():
    uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
    token = os.environ.get("MILVUS_TOKEN", "")
    client = MilvusClient(uri=uri, token=token)
    collection = "rls_example_" + uuid4().hex
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("owner", DataType.VARCHAR, max_length=100)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    client.create_collection(
        collection, schema=schema, properties={"rls.enabled": "true"}, consistency_level="Strong"
    )
    try:
        definition = {
            "policy_type": "permissive",
            "actions": [
                "query",
                "query_iterator",
                "search",
                "search_iterator",
                "hybrid_search",
                "insert",
                "upsert",
                "delete",
            ],
            "using_expr": "owner == $current_principal",
            "check_expr": "owner == $current_principal",
        }
        client.create_row_policy(collection, "owner", **definition)
        # Updates replace the complete definition; create rejects an existing name.
        client.update_row_policy(collection, "owner", description="Owner access", **definition)
        assert client.list_row_policies(collection)[0]["policy_name"] == "owner"

        tags = {"department": "engineering", "level": 3, "ratio": 3.0, "teams": ["red", "blue"]}
        client.set_rls_principal_tags(collection, "alice", tags)
        assert client.get_rls_principal_tags(collection, "alice") == tags
        assert "alice" in client.list_rls_principals(collection)

        for pk, principal in [(1, "alice"), (2, "bob")]:
            client.insert(
                collection,
                [{"id": pk, "owner": principal, "vector": [0.1, 0.2]}],
                rls_principal=principal,
            )
        index_params = client.prepare_index_params(
            "vector", index_type="AUTOINDEX", metric_type="L2"
        )
        client.create_index(collection, index_params)
        client.load_collection(collection)

        assert [row["id"] for row in client.get(collection, [1, 2], rls_principal="alice")] == [1]
        asyncio.run(read_async(uri, token, collection))
        try:
            client.insert(
                collection, [{"id": 3, "owner": "bob", "vector": [0.1, 0.2]}], rls_principal="alice"
            )
        except MilvusException as exc:
            assert "denied by RLS" in exc.message
        else:
            raise AssertionError("RLS must reject a row owned by another principal")

        # skip_rls=True is per request and requires SkipRLS privilege when auth is enabled.
        # rls.force=true on the collection disables that bypass entirely.
        client.delete_rls_principal_tags(collection, "alice", ["department"])
        client.delete_rls_principal_tags(collection, "alice")  # No keys deletes all tags.
        assert "alice" not in client.list_rls_principals(collection)
        client.drop_row_policy(collection, "owner")
        try:
            client.query(collection, filter="id >= 0", rls_principal="alice")
        except MilvusException as exc:
            assert "denied by RLS" in exc.message
        else:
            raise AssertionError("RLS must reject reads without an applicable policy")
    finally:
        client.drop_collection(collection)
        client.close()


if __name__ == "__main__":
    main()
