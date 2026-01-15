import time
from pymilvus import MilvusClient, DataType, IndexType, FieldSchema
import datetime
import pytz
import sys

# --- Configuration ---
MILVUS_HOST = "http://localhost:19530"

# --- Test 1: Database Properties ---
DB_NAME_TEST = "milvus_db_property_test"
VALID_TZ_INITIAL = "Asia/Shanghai"
VALID_TZ_NEW = "Europe/London"
INVALID_TZ = "Invalid/Timezone"

# --- Test 2: Collection Properties ---
COLLECTION_NAME_TEST = "tz_property_collection_test"
TIMESTAMP_FIELD = "tsz"
VECTOR_DIM = 4


# ==============================================================================
# 🚀 TEST SCENARIO 1: Database Property Management
# ==============================================================================
def run_database_property_tests(client: MilvusClient):
    """Executes tests for creating, altering, and dropping database properties, focusing on timezone."""

    print("\n\n" + "=" * 80)
    print("🚀 TEST 1: Database Property Management (Create, Alter, Drop Timezone)")
    print("=" * 80)

    # 1. Create Database and Set Initial Timezone
    if DB_NAME_TEST in client.list_databases():
        client.drop_database(db_name=DB_NAME_TEST)

    print(f"1. Creating database: {DB_NAME_TEST} with timezone: {VALID_TZ_INITIAL}")
    try:
        client.create_database(
            db_name=DB_NAME_TEST,
            properties={"timezone": VALID_TZ_INITIAL, "purpose": "Timezone Test"}
        )
        db_info_initial = client.describe_database(db_name=DB_NAME_TEST)
        print(f"✅ Success. Initial timezone: {db_info_initial.get('properties', {}).get('timezone')}")
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return

    # 2. Alter Database Properties (Change timezone to a valid value)
    print(f"\n2. Attempting to change timezone to a valid value: {VALID_TZ_NEW}")
    try:
        client.alter_database_properties(
            db_name=DB_NAME_TEST,
            properties={"timezone": VALID_TZ_NEW}
        )
        db_info_valid = client.describe_database(db_name=DB_NAME_TEST)
        print(f"✅ Success. New timezone: {db_info_valid.get('properties', {}).get('timezone')}")
    except Exception as e:
        print(f"❌ Failed to alter database properties: {e}")

    # 3. Alter Database Properties (Change timezone to an invalid value - Expected Failure)
    print(f"\n3. Attempting to change timezone to an invalid value: {INVALID_TZ} (Expected Failure)")
    try:
        client.alter_database_properties(
            db_name=DB_NAME_TEST,
            properties={"timezone": INVALID_TZ}
        )
        print("❌ ERROR: Invalid timezone change did not raise error.")
    except Exception as e:
        print(f"✅ Successfully caught expected error (Invalid Timezone): {e}")

    # 4. Drop Database Property
    print("\n4. Dropping 'purpose' property.")
    try:
        client.drop_database_properties(db_name=DB_NAME_TEST, property_keys=["purpose"])
        print("✅ Successfully dropped 'purpose' property.")
    except Exception as e:
        print(f"❌ Failed to drop database property: {e}")


# ==============================================================================
# 🚀 TEST SCENARIO 2: Collection Property Management
# ==============================================================================
def run_collection_property_tests(client: MilvusClient):
    """Executes tests for collection creation idempotency and collection/default DB property alteration."""

    print("\n\n" + "=" * 80)
    print("🚀 TEST 2: Collection Property Management (Idempotency and Timezone Alteration)")
    print("=" * 80)

    # 1. Define Schema & Cleanup
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(TIMESTAMP_FIELD, DataType.TIMESTAMPTZ)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    if client.has_collection(COLLECTION_NAME_TEST):
        client.drop_collection(COLLECTION_NAME_TEST)

    # 2. Idempotency Test: Attempting to create the Collection twice
    print("\n2.1. Idempotency Test: First Creation")
    try:
        client.create_collection(COLLECTION_NAME_TEST, schema=schema, consistency_level="Session")
        print(f"✅ Collection '{COLLECTION_NAME_TEST}' created successfully.")
    except Exception as e:
        print(f"❌ Error during first creation: {e}")

    print("\n2.2. Idempotency Test: Second Creation (Expected OK)")
    try:
        client.create_collection(COLLECTION_NAME_TEST, schema=schema, consistency_level="Session")
        print(f"✅ Collection '{COLLECTION_NAME_TEST}' created successfully again (Idempotent success).")
    except Exception as e:
        print(f"❌ Error during second creation: {e}")
        return

    # 3. Initial database timezone setup (on "default" DB)
    print("\n3. Alter default database timezone (Initial setup to Asia/Shanghai)")
    try:
        client.alter_database_properties("default", {"timezone": "Asia/Shanghai"})
        print("✅ Default database timezone set to Asia/Shanghai.")
    except Exception as e:
        print(f"❌ Failed to alter default database timezone: {e}")

    # 4. Create Index (Necessary setup but also part of ts.py's original flow)
    index_params = client.prepare_index_params(
        collection_name=COLLECTION_NAME_TEST, field_name="vec", index_type=IndexType.HNSW,
        metric_type="COSINE", params={"M": 30, "efConstruction": 200},
    )
    index_params.add_index(field_name=TIMESTAMP_FIELD, index_name="tsz_index", index_type="STL_SORT")
    client.create_index(COLLECTION_NAME_TEST, index_params)
    print("✅ Collection indexes created.")

    # 5. Comprehensive Alter Collection Timezone Tests

    # 5.1 Alter collection timezone to Europe/London
    print("\n5.1. Alter collection timezone to Europe/London")
    try:
        client.alter_collection_properties(COLLECTION_NAME_TEST, {"timezone": VALID_TZ_NEW})
        col_info = client.describe_collection(COLLECTION_NAME_TEST)
        print(f"✅ Success. New Collection timezone: {col_info.get('properties', {}).get('timezone')}")
    except Exception as e:
        print(f"❌ Failed to alter collection timezone: {e}")

    # 5.2 Alter collection timezone back to Asia/Shanghai
    print("\n5.2. Alter collection timezone back to Asia/Shanghai")
    try:
        client.alter_collection_properties(COLLECTION_NAME_TEST, {"timezone": VALID_TZ_INITIAL})
        col_info = client.describe_collection(COLLECTION_NAME_TEST)
        print(f"✅ Success. Collection timezone reset to: {col_info.get('properties', {}).get('timezone')}")
    except Exception as e:
        print(f"❌ Failed to alter collection timezone: {e}")

    # 5.3 Alter collection timezone to an invalid value (Error expected)
    print("\n5.3. Alter collection timezone to 'error' (Expected failure)")
    try:
        client.alter_collection_properties(COLLECTION_NAME_TEST, {"timezone": INVALID_TZ})
        print("❌ ERROR: Invalid collection timezone change did not raise error.")
    except Exception as e:
        print(f"✅ Successfully caught expected error: {e}")

    # 6. Invalid Alter Database Timezone Test (on "default" DB)
    print("\n6. Alter default database timezone to 'error' (Expected failure)")
    try:
        client.alter_database_properties("default", {"timezone": INVALID_TZ})
        print("❌ ERROR: Invalid database timezone change did not raise error.")
    except Exception as e:
        print(f"✅ Successfully caught expected error: {e}")


# ==============================================================================
# 🏃‍♂️ MAIN EXECUTION
# ==============================================================================
def main_property_management_tests():
    try:
        client = MilvusClient(uri=MILVUS_HOST)
    except Exception as e:
        print(f"Could not connect to Milvus service {MILVUS_HOST}. Please ensure the service is running. Error: {e}")
        sys.exit(1)

    # Execute all test scenarios
    run_database_property_tests(client)
    run_collection_property_tests(client)

    # --- Final Cleanup ---
    print("\n\n" + "=" * 80)
    print("🧹 Final Cleanup...")
    print("=" * 80)

    # Cleanup for Database Test
    try:
        if DB_NAME_TEST in client.list_databases():
            client.drop_database(db_name=DB_NAME_TEST)
            print(f"✅ Cleanup: Database '{DB_NAME_TEST}' dropped.")
    except Exception as e:
        print(f"❌ Failed to clean up database '{DB_NAME_TEST}': {e}")

    # Cleanup for Collection Test (in default DB)
    try:
        if client.has_collection(COLLECTION_NAME_TEST):
            client.release_collection(COLLECTION_NAME_TEST)
            client.drop_collection(COLLECTION_NAME_TEST)
            print(f"✅ Cleanup: Collection '{COLLECTION_NAME_TEST}' dropped.")
    except Exception as e:
        print(f"❌ Failed to clean up collection '{COLLECTION_NAME_TEST}': {e}")


if __name__ == "__main__":
    main_property_management_tests()