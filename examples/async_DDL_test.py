import asyncio
import numpy as np
from pymilvus import AsyncMilvusClient, DataType

fmt = "\n=== {:30} ===\n"
dim = 128
collection_name = "test_collection"
test_user = "test_user"
test_role = "test_role"
test_alias = "test_alias"
test_db = "test_database"
partition_name = "test_partition"
group_name = "test_privilege_group"

async def create_resources(client):
    print(fmt.format("Creating Resources"))

    print("Creating database...")
    await client.create_database(test_db)
    print(f"Database {test_db} created")

    print("Creating user and role...")
    await client.create_user(test_user, "password123")
    await client.create_role(test_role)
    print(f"User {test_user} and role {test_role} created")

    print("Creating privilege group...")
    await client.create_privilege_group(group_name)
    print(f"Privilege group {group_name} created")

    print("Creating collection...")
    schema = client.create_schema(
        auto_id=True, description="Test collection", enable_dynamic_field=True
    )
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=512)

    index_params = client.prepare_index_params()
    index_params.add_index("vector", index_type="HNSW", metric_type="L2")

    await client.create_collection(collection_name, schema=schema)
    print(f"Collection {collection_name} created")
    
    print("Creating partition...")
    await client.create_partition(collection_name, partition_name)
    print(f"Partition {partition_name} created")
    
    await client.create_index(collection_name, index_params=index_params)
    print("Index created")
    
    print("Creating alias...")
    await client.create_alias(collection_name, test_alias)
    print(f"Alias {test_alias} created")
    
    print("Loading collection...")
    await client.load_collection(collection_name)
    print("Collection loaded")
    
    return True

async def test_functionality(client):
    print(fmt.format("Testing Functionality"))

    print("Testing server info...")
    server_version = await client.get_server_version()
    print(f"Server version: {server_version}")

    print("Testing collection operations...")
    print(f"has_collection: {await client.has_collection(collection_name)}")
    print(f"list_collections: {await client.list_collections()}")
    print(f"describe_collection: {await client.describe_collection(collection_name)}")

    await client.rename_collection(collection_name, f"{collection_name}_renamed")
    await client.rename_collection(f"{collection_name}_renamed", collection_name)
    print("Collection rename test completed")

    await client.alter_collection_properties(collection_name, {"collection.ttl.seconds": 3600})
    await client.drop_collection_properties(collection_name, ["collection.ttl.seconds"])
    print("Collection properties test completed")

    collection_info = await client.describe_collection(collection_name)
    enable_dynamic = collection_info.get('enable_dynamic_field', False)
    print(f"Dynamic fields enabled: {enable_dynamic}")

    await client.release_collection(collection_name)
    await client.alter_collection_field(collection_name, "text", {"max_length": 256})
    print("Collection field altered")
    await client.load_collection(collection_name)
    print("Collection reloaded")

    print("Testing index operations...")
    print(f"list_indexes: {await client.list_indexes(collection_name)}")
    index_name = (await client.list_indexes(collection_name))[0]
    print(f"describe_index: {await client.describe_index(collection_name, index_name)}")

    await client.release_collection(collection_name)
    await client.alter_index_properties(collection_name, index_name, {"mmap.enabled": True})
    await client.drop_index_properties(collection_name, index_name, ["mmap.enabled"])
    print("Index properties test completed")
    await client.load_collection(collection_name)
    print("Collection reloaded after index operations")

    print("Testing partition operations...")
    print(f"has_partition: {await client.has_partition(collection_name, partition_name)}")
    print(f"list_partitions: {await client.list_partitions(collection_name)}")

    print("Testing user management...")
    print(f"list_users: {await client.list_users()}")
    print(f"describe_user: {await client.describe_user(test_user)}")

    await client.update_password(test_user, "password123", "newpassword123")
    print("Password updated")

    print("Testing role management...")
    print(f"list_roles: {await client.list_roles()}")
    print(f"describe_role: {await client.describe_role(test_role)}")

    await client.grant_role(test_user, test_role)
    await client.revoke_role(test_user, test_role)
    print("Role grant/revoke test completed")

    print("Testing alias operations...")
    print(f"describe_alias: {await client.describe_alias(test_alias)}")
    print(f"list_aliases: {await client.list_aliases(collection_name)}")

    await client.alter_alias(collection_name, test_alias)
    print("Alias altered")

    print("Testing data operations...")
    data = [{
        "vector": np.random.random(dim).tolist(),
        "text": f"test text {i}",
        "dynamic_field_int": i * 10,
        "dynamic_field_str": f"dynamic_{i}",
        "dynamic_field_float": i * 0.5
    } for i in range(10)]

    result = await client.insert(collection_name, data)
    print(f"insert result: {result}")

    await client.flush(collection_name)
    print("flush completed")

    compaction_id = await client.compact(collection_name)
    print(f"compact job id: {compaction_id}")
    print(f"compaction state: {await client.get_compaction_state(compaction_id)}")

    print("Testing stats and load state...")
    print(f"get_collection_stats: {await client.get_collection_stats(collection_name)}")
    partition_stats = await client.get_partition_stats(collection_name, partition_name)
    print(f"get_partition_stats: {partition_stats}")
    print(f"get_load_state: {await client.get_load_state(collection_name)}")

    await client.refresh_load(collection_name)
    print(f"describe_replica: {await client.describe_replica(collection_name)}")

    print("Testing text analysis...")
    analyzer_result = await client.run_analyzer(
        texts=["hello world", "test text"],
        analyzer_params={"tokenizer": "standard"}
    )
    print(f"analyzer result: {analyzer_result}")

    print("Testing database operations...")
    print(f"list_databases: {await client.list_databases()}")
    print(f"describe_database: {await client.describe_database(test_db)}")

    client.use_database(test_db)
    client.using_database("default")
    print("Database switch test completed")

    await client.alter_database_properties(test_db, {"database.replica.num": 1})
    await client.drop_database_properties(test_db, ["database.replica.num"])
    print("Database properties test completed")

    print("Testing privilege group operations...")
    print(f"list_privilege_groups: {await client.list_privilege_groups()}")

    await client.add_privileges_to_group(group_name, ["Insert", "Delete"])
    await client.remove_privileges_from_group(group_name, ["Delete"])
    print("Privilege group operations completed")

async def cleanup_resources(client):
    print(fmt.format("Cleaning Up Resources"))

    print("Dropping alias...")
    await client.drop_alias(test_alias)
    print(f"Alias {test_alias} dropped")

    print("Dropping collection...")
    await client.drop_collection(collection_name)
    print(f"Collection {collection_name} dropped")

    print("Dropping user and role...")
    await client.drop_user(test_user)
    await client.drop_role(test_role)
    print(f"User {test_user} and role {test_role} dropped")

    print("Dropping privilege group...")
    await client.drop_privilege_group(group_name)
    print(f"Privilege group {group_name} dropped")

async def main():
    client = AsyncMilvusClient("http://localhost:19530")
    await create_resources(client)
    await test_functionality(client)
    await cleanup_resources(client)
    await client.close()
    print(fmt.format("Test Completed"))

if __name__ == "__main__":
    asyncio.run(main())