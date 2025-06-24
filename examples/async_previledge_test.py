import asyncio

from pymilvus import AsyncMilvusClient, DataType

# Super user credentials for admin operations
super_user = ""
super_password = ""

fmt = "\n=== {:30} ===\n"
collection_name = "test_privilege_collection"
test_user = "test_privilege_user"
test_role_v1 = "test_role_v1"
test_role_v2 = "test_role_v2"
test_db = "test_privilege_database"
privilege_group_name = "test_privilege_group"

# Test data for privilege operations
db_rw_privileges = [
    {"object_type": "Collection", "privilege": "Insert", "object_name": collection_name},
    {"object_type": "Collection", "privilege": "Delete", "object_name": collection_name},
    {"object_type": "Collection", "privilege": "Search", "object_name": collection_name},
    {"object_type": "Collection", "privilege": "Query", "object_name": collection_name},
]

db_ro_privileges = [
    {"object_type": "Collection", "privilege": "Search", "object_name": collection_name},
    {"object_type": "Collection", "privilege": "Query", "object_name": collection_name},
]


async def create_resources(client: AsyncMilvusClient):
    print(fmt.format("Creating Resources"))

    print("Creating database...")
    await client.create_database(test_db)
    print(f"Database {test_db} created")

    print("Creating user and roles...")
    await client.create_user(test_user, "password123")
    await client.create_role(test_role_v1)
    await client.create_role(test_role_v2)
    print(f"User {test_user} and roles {test_role_v1}, {test_role_v2} created")

    print("Creating privilege group...")
    await client.create_privilege_group(privilege_group_name)
    print(f"Privilege group {privilege_group_name} created")

    print("Adding privileges to group...")
    await client.add_privileges_to_group(privilege_group_name, ["Search", "Query", "Insert"])
    print("Privileges added to group")

    print("Creating collection...")
    schema = client.create_schema(
        auto_id=True, description="Test privilege collection", enable_dynamic_field=True
    )
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=128)
    schema.add_field("text", DataType.VARCHAR, max_length=512)

    await client.create_collection(collection_name, schema=schema, using_database=test_db)
    print(f"Collection {collection_name} created in database {test_db}")

    return True


async def test_grant_revoke_privilege(client: AsyncMilvusClient):
    print(fmt.format("Testing grant_privilege / revoke_privilege"))

    # Test grant_privilege for different privileges
    print("Testing grant_privilege...")
    for privilege_item in db_rw_privileges:
        await client.grant_privilege(
            role_name=test_role_v1,
            object_type=privilege_item["object_type"],
            privilege=privilege_item["privilege"],
            object_name=privilege_item["object_name"],
            db_name=test_db,
        )
        print(
            f"Granted {privilege_item['privilege']} on {privilege_item['object_name']} to {test_role_v1}"
        )

    # Check role privileges
    print(f"\nChecking role privileges for {test_role_v1}...")
    role_info = await client.describe_role(test_role_v1)
    print(f"Role {test_role_v1} privileges: {role_info}")

    # Grant role to user
    print(f"\nGranting role {test_role_v1} to user {test_user}...")
    await client.grant_role(test_user, test_role_v1)
    print("Role granted to user")

    # Check user info
    user_info = await client.describe_user(test_user)
    print(f"User {test_user} info: {user_info}")

    # Test revoke_privilege
    print("\nTesting revoke_privilege...")
    for privilege_item in db_ro_privileges:
        await client.revoke_privilege(
            role_name=test_role_v1,
            object_type=privilege_item["object_type"],
            privilege=privilege_item["privilege"],
            object_name=privilege_item["object_name"],
            db_name=test_db,
        )
        print(
            f"Revoked {privilege_item['privilege']} on {privilege_item['object_name']} from {test_role_v1}"
        )

    # Check role privileges after revoke
    print(f"\nChecking role privileges after revoke for {test_role_v1}...")
    role_info = await client.describe_role(test_role_v1)
    print(f"Role {test_role_v1} privileges after revoke: {role_info}")

    # Revoke role from user
    print(f"\nRevoking role {test_role_v1} from user {test_user}...")
    await client.revoke_role(test_user, test_role_v1)
    print("Role revoked from user")


async def test_grant_revoke_privilege_v2(client: AsyncMilvusClient):
    print(fmt.format("Testing grant_privilege_v2 / revoke_privilege_v2"))

    # Test grant_privilege_v2 with custom privilege group
    print("Testing grant_privilege_v2 with custom privilege group...")
    await client.grant_privilege_v2(
        role_name=test_role_v2, privilege=privilege_group_name, collection_name="*", db_name=test_db
    )
    print(
        f"Granted privilege group {privilege_group_name} on all collections in {test_db} to {test_role_v2}"
    )

    # Test grant_privilege_v2 with built-in privilege groups
    print("\nTesting grant_privilege_v2 with built-in privilege groups...")

    # Grant collection-level privileges
    await client.grant_privilege_v2(
        role_name=test_role_v2,
        privilege="CollectionReadWrite",
        collection_name=collection_name,
        db_name=test_db,
    )
    print(f"Granted CollectionReadWrite on {collection_name} to {test_role_v2}")

    # Grant database-level privileges
    await client.grant_privilege_v2(
        role_name=test_role_v2, privilege="DatabaseReadOnly", collection_name="*", db_name=test_db
    )
    print(f"Granted DatabaseReadOnly on database {test_db} to {test_role_v2}")

    # Check role privileges
    print(f"\nChecking role privileges for {test_role_v2}...")
    role_info = await client.describe_role(test_role_v2)
    print(f"Role {test_role_v2} privileges: {role_info}")

    # Grant role to user
    print(f"\nGranting role {test_role_v2} to user {test_user}...")
    await client.grant_role(test_user, test_role_v2)
    print("Role granted to user")

    # Check user info
    user_info = await client.describe_user(test_user)
    print(f"User {test_user} info: {user_info}")

    # Test revoke_privilege_v2
    print("\nTesting revoke_privilege_v2...")

    # Revoke custom privilege group
    await client.revoke_privilege_v2(
        role_name=test_role_v2, privilege=privilege_group_name, collection_name="*", db_name=test_db
    )
    print(f"Revoked privilege group {privilege_group_name} from {test_role_v2}")

    # Revoke built-in privilege group
    await client.revoke_privilege_v2(
        role_name=test_role_v2, privilege="DatabaseReadOnly", collection_name="*", db_name=test_db
    )
    print(f"Revoked DatabaseReadOnly from {test_role_v2}")

    # Revoke collection-level privilege
    await client.revoke_privilege_v2(
        role_name=test_role_v2,
        privilege="CollectionReadWrite",
        collection_name=collection_name,
        db_name=test_db,
    )
    print(f"Revoked CollectionReadWrite from {test_role_v2}")

    # Check role privileges after revoke
    print(f"\nChecking role privileges after revoke for {test_role_v2}...")
    role_info = await client.describe_role(test_role_v2)
    print(f"Role {test_role_v2} privileges after revoke: {role_info}")

    # Revoke role from user
    print(f"\nRevoking role {test_role_v2} from user {test_user}...")
    await client.revoke_role(test_user, test_role_v2)
    print("Role revoked from user")


async def main():
    client = AsyncMilvusClient("http://localhost:19530", user=super_user, password=super_password)

    await create_resources(client)
    await test_grant_revoke_privilege(client)
    await test_grant_revoke_privilege_v2(client)
    print(fmt.format("All Privilege Tests Completed Successfully"))
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
