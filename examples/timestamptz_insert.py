import time
from pymilvus import MilvusClient, DataType, IndexType
import datetime
import pytz
import random
import sys

# --- Configuration ---
MILVUS_HOST = "http://localhost:19530"

# --- Test Scenario 1: Add Field Default Value ---
COLLECTION_NAME_DEFAULT = "default_value_test_col"
TIMESTAMP_FIELD_DEFAULT = "event_time_default"
# Default value for TIMESTAMPTZ field (must be ISO 8601, UTC)
DEFAULT_TS_VALUE_STR = "2025-01-01T00:00:00Z"
EXPLICIT_TS_VALUE_STR = "2026-06-06T12:34:56Z"
EXPECTED_DEFAULT_MATCH = "2025-01-01T00:00:00"

# --- Test Scenario 2: Naive Time Insertion with Collection Timezone ---
COLLECTION_NAME_TZ_INSERT = "timestamptz_tz_test_col"
TIMESTAMP_FIELD_TZ_INSERT = "event_time_tz"
COLLECTION_TZ_STR = "America/New_York"
RAW_TIME_STR = "2024-01-04 14:26:27"  # Naive time

# Pre-calculate expected UTC based on COLLECTION_TZ_STR
COLLECTION_TZ = pytz.timezone(COLLECTION_TZ_STR)
correct_dt_utc = COLLECTION_TZ.localize(
    datetime.datetime.strptime(RAW_TIME_STR, "%Y-%m-%d %H:%M:%S")
).astimezone(pytz.utc)
EXPECTED_CORRECT_UTC = correct_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
OBSERVED_ERROR_UTC_PART = "2024-01-04T14:26:27"  # Naive time mistakenly stored as UTC


# --- Core Setup Function ---
def setup_base_collection(client: MilvusClient, name: str, tz_property: str = None):
    """Creates a generic base collection."""
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)

    if client.has_collection(name):
        client.drop_collection(name)

    properties = {"timezone": tz_property} if tz_property else {}

    client.create_collection(name, schema=schema, consistency_level="Session", properties=properties)

    index_params = client.prepare_index_params(
        collection_name=name,
        field_name="vec",
        index_type=IndexType.HNSW,
        metric_type="COSINE",
        params={"M": 30, "efConstruction": 200},
    )
    client.create_index(name, index_params)

    return schema


# ==============================================================================
# 🚀 TEST SCENARIO 1: add_collection_field Default Value Retroactive Fill
# ==============================================================================
def run_default_value_retrofill_test(client: MilvusClient):
    """Validates the default_value behavior of add_collection_field for TIMESTAMPTZ."""

    print("\n\n" + "=" * 80)
    print(f"🚀 TEST 1: {COLLECTION_NAME_DEFAULT} - Add Field Default Value (Retroactive Fill)")
    print("=" * 80)

    # 1. Setup Base Collection (No timestamp field yet)
    setup_base_collection(client, COLLECTION_NAME_DEFAULT)
    client.load_collection(COLLECTION_NAME_DEFAULT)
    print(f"✅ Base collection '{COLLECTION_NAME_DEFAULT}' created and loaded.")

    # 2. Inserting historical data (missing the event_time field)
    history_data = [
        {"id": 101, "vec": [random.random() for _ in range(4)]},
        {"id": 102, "vec": [random.random() for _ in range(4)]},
    ]

    client.insert(COLLECTION_NAME_DEFAULT, history_data)
    client.flush(COLLECTION_NAME_DEFAULT)
    print(f"✅ Stage 1: Inserted historical data (ID 101, 102) before field addition.")

    # 3. Adding TIMESTAMPTZ field with a default value
    try:
        client.add_collection_field(
            collection_name=COLLECTION_NAME_DEFAULT,
            field_name=TIMESTAMP_FIELD_DEFAULT,
            data_type=DataType.TIMESTAMPTZ,
            nullable=True,
            default_value=DEFAULT_TS_VALUE_STR
        )
        print(
            f"✅ Stage 2: Field '{TIMESTAMP_FIELD_DEFAULT}' added successfully with default value: {DEFAULT_TS_VALUE_STR}")
    except Exception as e:
        print(f"❌ Stage 2: Failed to add field: {e}")
        return

    # 4. Inserting new data (Testing default vs explicit)
    new_data = [
        {"id": 201, "vec": [random.random() for _ in range(4)]},  # Missing field -> Expected Default
        {"id": 202, "vec": [random.random() for _ in range(4)], "event_time_default": EXPLICIT_TS_VALUE_STR}
        # Explicit -> Expected Explicit
    ]

    client.insert(COLLECTION_NAME_DEFAULT, new_data)
    client.flush(COLLECTION_NAME_DEFAULT)
    print("✅ Stage 3: Inserted new data (ID 201, 202) after field addition.")

    # 5. Verification
    client.load_collection(COLLECTION_NAME_DEFAULT)
    query_results = client.query(
        collection_name=COLLECTION_NAME_DEFAULT,
        filter="id in [101, 102, 201, 202]",
        output_fields=["id", TIMESTAMP_FIELD_DEFAULT],
        timezone="UTC"
    )

    results_map = {r.get('id'): r.get(TIMESTAMP_FIELD_DEFAULT) for r in query_results}
    print("\n--- Verification Results ---")

    # ID 101 (Historical Missing) - **Key Test**
    actual_ts_101 = results_map.get(101)
    if actual_ts_101 and EXPECTED_DEFAULT_MATCH in str(actual_ts_101):
        print(f"✅ Result 1 (ID 101, Historical): **Successfully retroactively filled** with default: {actual_ts_101}")
    else:
        print(f"❌ Result 1 (ID 101, Historical): **FAILED**, not retroactively filled: {actual_ts_101}")

    # ID 201 (New Missing)
    actual_ts_201 = results_map.get(201)
    if actual_ts_201 and EXPECTED_DEFAULT_MATCH in str(actual_ts_201):
        print(f"✅ Result 2 (ID 201, New Missing): Successfully filled with default: {actual_ts_201}")
    else:
        print(f"❌ Result 2 (ID 201, New Missing): Default value did not take effect: {actual_ts_201}")

    # ID 202 (New Explicit)
    actual_ts_202 = results_map.get(202)
    if actual_ts_202 and "2026-06-06T12:34:56" in str(actual_ts_202):
        print(f"✅ Result 3 (ID 202, New Explicit): Successfully overrode default value: {actual_ts_202}")
    else:
        print(f"❌ Result 3 (ID 202, New Explicit): Explicit value failed: {actual_ts_202}")


# ==============================================================================
# 🚀 TEST SCENARIO 2: Naive Time Insertion with Collection Timezone
# ==============================================================================
def run_naive_insertion_tz_test(client: MilvusClient):
    """Verifies that naive time strings are correctly interpreted using the Collection's timezone setting."""

    print("\n\n" + "=" * 80)
    print(f"🚀 TEST 2: {COLLECTION_NAME_TZ_INSERT} - Naive Time Insertion (TZ Conversion)")
    print("=" * 80)
    print(f"Collection Timezone: {COLLECTION_TZ_STR}")
    print(f"Raw Input (Naive): {RAW_TIME_STR}")
    print(f"Expected UTC Storage: {EXPECTED_CORRECT_UTC}")

    # 1. Setup Collection with TIMESTAMPTZ field and Collection Timezone
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vec", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field(TIMESTAMP_FIELD_TZ_INSERT, DataType.TIMESTAMPTZ)

    # Drop and recreate the collection to ensure TIMESTAMPTZ field is present from start
    if client.has_collection(COLLECTION_NAME_TZ_INSERT):
        client.drop_collection(COLLECTION_NAME_TZ_INSERT)

    client.create_collection(
        COLLECTION_NAME_TZ_INSERT,
        schema=schema,
        consistency_level="Session",
        properties={"timezone": COLLECTION_TZ_STR}
    )

    # Create index for vector field
    index_params = client.prepare_index_params(
        collection_name=COLLECTION_NAME_TZ_INSERT, field_name="vec", index_type=IndexType.HNSW,
        metric_type="COSINE", params={"M": 30, "efConstruction": 200},
    )
    client.create_index(COLLECTION_NAME_TZ_INSERT, index_params)
    client.load_collection(COLLECTION_NAME_TZ_INSERT)
    print("✅ Setup complete for Naive Time Insertion Test.")

    # 2. Insert raw data (naive time string)
    insert_data = [
        {"id": 1, "vec": [random.random() for _ in range(4)], TIMESTAMP_FIELD_TZ_INSERT: RAW_TIME_STR},
    ]

    client.insert(COLLECTION_NAME_TZ_INSERT, insert_data)
    client.flush(COLLECTION_NAME_TZ_INSERT)
    print("✅ Stage 1: Insertion of naive time string successful.")

    # 3. Query data and verify UTC storage
    client.load_collection(COLLECTION_NAME_TZ_INSERT)
    query_results = client.query(
        collection_name=COLLECTION_NAME_TZ_INSERT,
        filter="id == 1",
        output_fields=["id", TIMESTAMP_FIELD_TZ_INSERT],
        timezone="UTC"  # Retrieve internal UTC storage time
    )

    if not query_results:
        print("❌ Query result is empty.")
        return

    actual_ts_str = query_results[0].get(TIMESTAMP_FIELD_TZ_INSERT)

    print("\n--- Verification Results ---")
    print(f"Actual Query Result (UTC): {actual_ts_str}")

    # Verification Logic: Check if the actual result matches the expected 19:26:27Z
    if actual_ts_str and EXPECTED_CORRECT_UTC in actual_ts_str:
        print(
            f"✅ Verification SUCCESS: Milvus correctly converted '{RAW_TIME_STR}' to UTC based on '{COLLECTION_TZ_STR}'.")
        print(f" (Stored UTC: {EXPECTED_CORRECT_UTC})")
    elif actual_ts_str and OBSERVED_ERROR_UTC_PART in actual_ts_str:
        print(f"❌ Verification FAILED (Issue Reproduced): Milvus mistakenly treated '{RAW_TIME_STR}' as UTC time.")
        print(f" (Mistaken UTC: {actual_ts_str})")
    else:
        print(f"⚠️ Verification FAILED: Actual result '{actual_ts_str}' does not match any expectation.")


# ==============================================================================
# 🏃‍♂️ MAIN EXECUTION (Updated Function Name)
# ==============================================================================
def main_timestamptz_tests():
    try:
        client = MilvusClient(uri=MILVUS_HOST)
    except Exception as e:
        print(f"Could not connect to Milvus service {MILVUS_HOST}. Please ensure the service is running. Error: {e}")
        sys.exit(1)

    # Run all test scenarios
    run_default_value_retrofill_test(client)
    run_naive_insertion_tz_test(client)

    # --- Cleanup for all collections ---
    print("\n\n" + "=" * 80)
    print("🧹 Cleaning up collections...")
    print("=" * 80)

    collections_to_clean = [COLLECTION_NAME_DEFAULT, COLLECTION_NAME_TZ_INSERT]

    for name in collections_to_clean:
        try:
            if client.has_collection(name):
                client.release_collection(name)
                client.drop_collection(name)
                print(f"✅ Cleanup: Collection '{name}' dropped.")
        except Exception as e:
            print(f"❌ Failed to clean up collection '{name}': {e}")


if __name__ == "__main__":
    main_timestamptz_tests()