import time
from pymilvus import MilvusClient, DataType, IndexType
import datetime
import pytz
import random
import sys

# --- Configuration ---
MILVUS_HOST = "http://localhost:19530"
VECTOR_DIM = 4

# ==============================================================================
# --- CONSTANTS ---
# ==============================================================================

# Test 1: Query & Search Output Timezone
COLLECTION_NAME_QUERY = "timestamptz_query_search_test"
TIMESTAMP_FIELD_QUERY = "event_time"
# Standard data point stored in UTC for conversion tests (2025-01-01T12:00:00Z)
UTC_BASE_TIME = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=pytz.utc).isoformat()
# Los Angeles Test Time (UTC 02:00:00Z -> LA 2024-12-31T18:00:00-08:00)
UTC_LA_TEST_TIME = datetime.datetime(2025, 1, 1, 2, 0, 0, tzinfo=pytz.utc).isoformat()

# Test 2: Separate DB Property Management
# NEW DB NAME: Renamed for clarity and to prevent conflicts
DB_NAME_SEPARATE_TEST = "tz_mgmt_separate_db"
COLLECTION_NAME_PROP = "timestamptz_prop_col"
VALID_TZ_INITIAL = "Asia/Shanghai"
VALID_TZ_NEW = "Europe/London"
INVALID_TZ = "invalid_timezone_for_test"


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def setup_query_collection(client: MilvusClient, name: str, tz_property: str = "UTC"):
    """Creates a Collection for retrieval tests and inserts standardized UTC data."""
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(TIMESTAMP_FIELD_QUERY, DataType.TIMESTAMPTZ)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

    if client.has_collection(name):
        client.drop_collection(name)

    properties = {"timezone": tz_property}
    client.create_collection(name, schema=schema, consistency_level="Session", properties=properties)

    index_params = client.prepare_index_params(collection_name=name, field_name="vec", index_type=IndexType.HNSW,
                                               metric_type="COSINE")
    client.create_index(name, index_params)
    client.load_collection(name)

    # Insert two records for different test scenarios
    data = [
        # Record 1: Used for general TZ verification (UTC 12:00:00)
        {"id": 1, "vec": [0.5] * VECTOR_DIM, TIMESTAMP_FIELD_QUERY: UTC_BASE_TIME},
        # Record 2: Used for Los Angeles TZ verification (UTC 02:00:00)
        {"id": 2, "vec": [0.2] * VECTOR_DIM, TIMESTAMP_FIELD_QUERY: UTC_LA_TEST_TIME},
    ]
    client.insert(name, data)
    client.flush(name)
    client.load_collection(name)
    print(f"✅ Collection '{name}' created and loaded with UTC data.")


# ------------------------------------------------------------------------------
# 🚀 TEST 1: Query/Search Output Timezone Conversion Verification
# ------------------------------------------------------------------------------
def run_retrieval_timezone_test(client: MilvusClient):
    """Verifies that the timezone parameter in query() and search() correctly converts the output time to different time zones."""

    print("\n\n" + "=" * 80)
    print(f"🚀 TEST 1: {COLLECTION_NAME_QUERY} - Query/Search Output Timezone Conversion Verification")
    print("=" * 80)

    setup_query_collection(client, COLLECTION_NAME_QUERY, tz_property="UTC")

    # Define Timezone Verification Parameters: (TZ Name, Expected TZ Offset, Expected Time Part)
    TZ_VERIFICATIONS = [
        # Base Time: UTC 12:00:00 (id=1)
        ("Asia/Shanghai", "+08:00", "20:00:00"),  # UTC 12:00 -> Shanghai 20:00
        ("America/New_York", "-05:00", "07:00:00"),  # UTC 12:00 -> New York 07:00
        # Base Time: UTC 02:00:00 (id=2), testing the Los Angeles TZ
        ("America/Los_Angeles", "-08:00", "18:00:00"),  # UTC 02:00 -> LA 2024-12-31 18:00
    ]

    # --- 1.1 Query Output Conversion Verification ---
    print("\n--- 1.1 Query Output Conversion Verification ---")
    for output_tz, expected_offset, expected_time_part in TZ_VERIFICATIONS:

        # Use different IDs to match the two base UTC times
        filter_id = '1' if expected_time_part not in ["18:00:00"] else '2'

        results = client.query(
            COLLECTION_NAME_QUERY,
            filter=f"id == {filter_id}",
            output_fields=["id", TIMESTAMP_FIELD_QUERY],
            timezone=output_tz,
        )
        if not results: continue

        ts_output = results[0].get(TIMESTAMP_FIELD_QUERY)
        # Check if the output string contains the correct offset and time part
        is_correct = expected_offset in ts_output and expected_time_part in ts_output
        status = "✅ Success" if is_correct else "❌ Failure"
        print(f"    - Query {output_tz} ({expected_offset}): {status} (Actual: {ts_output})")

    # --- 1.2 Search Output Conversion Verification (Los Angeles) ---
    print("\n--- 1.2 Search Output Conversion Verification (Los Angeles) ---")

    output_tz = "America/Los_Angeles"
    expected_offset = "-08:00"
    expected_time_part = "18:00:00"

    search_vectors = [[0.2] * VECTOR_DIM]
    results = client.search(
        COLLECTION_NAME_QUERY,
        data=search_vectors,
        limit=1,
        output_fields=["id", TIMESTAMP_FIELD_QUERY],
        search_field="vec",
        timezone=output_tz,
    )

    result_ts = results[0][0].get(TIMESTAMP_FIELD_QUERY) if results and results[0] else "N/A"

    is_correct = expected_offset in result_ts and expected_time_part in result_ts
    status = "✅ Success" if is_correct else "❌ Failure"
    print(f"    - Search {output_tz} ({expected_offset}): {status} (Actual: {result_ts})")


# ------------------------------------------------------------------------------
# 🚀 TEST 2: Separate Database and Collection Property Management
# ------------------------------------------------------------------------------
def run_separate_db_property_tests(base_client: MilvusClient) -> MilvusClient | None:
    """Verifies modification and error handling of database and collection properties in a non-'default' database context.

    Returns:
        MilvusClient: The client connected to the separate database, used for cleanup.
    """

    print("\n\n" + "=" * 80)
    print(f"🚀 TEST 2: Separate Database '{DB_NAME_SEPARATE_TEST}' Timezone Property Management")
    print("=" * 80)

    db_client = None

    # --- Pre-Cleanup Logic (Handles existing DB state before creation) ---
    if DB_NAME_SEPARATE_TEST in base_client.list_databases():
        print(f"⚠️ Existing database '{DB_NAME_SEPARATE_TEST}' found. Attempting pre-cleanup.")
        try:
            # Must drop collection inside the database first
            temp_client = MilvusClient(uri=MILVUS_HOST, db_name=DB_NAME_SEPARATE_TEST)
            if temp_client.has_collection(COLLECTION_NAME_PROP):
                temp_client.release_collection(COLLECTION_NAME_PROP)
                temp_client.drop_collection(COLLECTION_NAME_PROP)
                print(f"  ✅ Existing collection '{COLLECTION_NAME_PROP}' dropped from database.")
        except Exception as e:
            print(f"  ❌ Could not clean up collection before database drop attempt: {e}")

        try:
            base_client.drop_database(db_name=DB_NAME_SEPARATE_TEST)
            print(f"  ✅ Existing database '{DB_NAME_SEPARATE_TEST}' dropped.")
        except Exception as e:
            print(f"  ❌ Failed to drop existing database: {e}. Skipping database creation.")
            return None
    # --- End Pre-Cleanup ---

    # 1. Create a separate database and set initial timezone
    try:
        base_client.create_database(
            db_name=DB_NAME_SEPARATE_TEST,
            properties={"timezone": VALID_TZ_INITIAL}
        )
        print(f"✅ Database '{DB_NAME_SEPARATE_TEST}' created successfully with initial timezone: {VALID_TZ_INITIAL}")
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return None

    # 2. Get a client pointing to the new database and create a collection
    db_client = MilvusClient(uri=MILVUS_HOST, db_name=DB_NAME_SEPARATE_TEST)
    schema = db_client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("tsz", DataType.TIMESTAMPTZ)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    db_client.create_collection(COLLECTION_NAME_PROP, schema=schema, consistency_level="Session")

    # 3. Alter Database Properties (VALID_TZ_INITIAL -> VALID_TZ_NEW)
    try:
        base_client.alter_database_properties(DB_NAME_SEPARATE_TEST, {"timezone": VALID_TZ_NEW})
        db_info = base_client.describe_database(db_name=DB_NAME_SEPARATE_TEST)
        print(f"✅ Database timezone successfully altered: {db_info.get('properties', {}).get('timezone')}")
    except Exception as e:
        print(f"❌ Database timezone alteration failed: {e}")

    # 4. Alter Collection Properties (Override database timezone)
    try:
        db_client.alter_collection_properties(COLLECTION_NAME_PROP, {"timezone": "America/New_York"})
        col_info = db_client.describe_collection(COLLECTION_NAME_PROP)
        print(f"✅ Collection timezone successfully altered: {col_info.get('properties', {}).get('timezone')}")
    except Exception as e:
        print(f"❌ Collection timezone alteration failed: {e}")

    # 5. Alter Database Timezone to Invalid Value (Error Handling Verification)
    print("\n--- 5. Verification of Invalid Timezone Error Handling (Expected Failure) ---")
    try:
        base_client.alter_database_properties(DB_NAME_SEPARATE_TEST, {"timezone": INVALID_TZ})
        print("❌ ERROR: Invalid timezone alteration did not raise an error.")
    except Exception as e:
        print(f"✅ Successfully caught expected error (Invalid Timezone): {e}")

    return db_client


# ==============================================================================
# 🏃‍♂️ MAIN EXECUTION
# ==============================================================================
def main_timestamptz_retrieval_suite():
    try:
        client = MilvusClient(uri=MILVUS_HOST)
    except Exception as e:
        print(f"Could not connect to Milvus service {MILVUS_HOST}. Please ensure the service is running. Error: {e}")
        sys.exit(1)

    # Run all test scenarios
    run_retrieval_timezone_test(client)
    db_client_separate = run_separate_db_property_tests(client)  # Store the client for cleanup

    # --- Cleanup all resources ---
    print("\n\n" + "=" * 80)
    print("🧹 Cleaning up all test resources...")
    print("=" * 80)

    # 1. Cleanup Collection in the default database context
    collections_to_clean = [COLLECTION_NAME_QUERY]

    for name in collections_to_clean:
        try:
            if client.has_collection(name):
                client.release_collection(name)
                client.drop_collection(name)
                print(f"✅ Cleanup: Collection '{name}' dropped.")
        except Exception as e:
            print(f"❌ Failed to clean up collection '{name}': {e}")

    # 2. Cleanup Collection AND Database for the separate test context
    if db_client_separate:
        # 2a. Drop collection inside the separate database context
        try:
            if db_client_separate.has_collection(COLLECTION_NAME_PROP):
                db_client_separate.release_collection(COLLECTION_NAME_PROP)
                db_client_separate.drop_collection(COLLECTION_NAME_PROP)
                print(f"✅ Cleanup: Collection '{COLLECTION_NAME_PROP}' dropped from '{DB_NAME_SEPARATE_TEST}'.")
        except Exception as e:
            print(f"❌ Failed to clean up collection '{COLLECTION_NAME_PROP}' in separate DB: {e}")

        # 2b. Drop the separate database itself using the base client
        try:
            if DB_NAME_SEPARATE_TEST in client.list_databases():
                client.drop_database(db_name=DB_NAME_SEPARATE_TEST)
                print(f"✅ Cleanup: Database '{DB_NAME_SEPARATE_TEST}' dropped.")
        except Exception as e:
            print(f"❌ Failed to clean up database '{DB_NAME_SEPARATE_TEST}': {e}")


if __name__ == "__main__":
    main_timestamptz_retrieval_suite()