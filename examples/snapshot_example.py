"""
Milvus Snapshot API Usage Examples

This script demonstrates how to use the snapshot functionality in Milvus for:
- Creating backups of collections
- Listing and describing snapshots
- Restoring snapshots to new collections
- Monitoring restore progress
- Managing snapshot lifecycle

Requirements:
- Milvus server running (standalone or cluster)
- pymilvus with snapshot API support
"""

import time
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams

# Configuration
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "demo_collection"
SNAPSHOT_NAME = "demo_backup_20240101"
RESTORED_COLLECTION = "demo_restored"


def create_demo_collection(client: MilvusClient):
    """Create a demo collection with sample data."""
    print("\n=== Creating Demo Collection ===")

    # Define schema
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)

    # Create collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        consistency_level="Strong"
    )
    print(f"✓ Collection '{COLLECTION_NAME}' created")

    # Insert sample data
    import random
    data = [
        {
            "embedding": [random.random() for _ in range(128)],
            "text": f"Sample text {i}"
        }
        for i in range(100)
    ]

    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"✓ Inserted {len(data)} records")

    # Create index
    index_params = IndexParams()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 16}
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params
    )
    print("✓ Index created")


def demo_create_snapshot(client: MilvusClient):
    """Demonstrate snapshot creation."""
    print("\n=== Creating Snapshot ===")

    # Best Practice: Flush data before creating snapshot
    print("Flushing data to ensure all changes are persisted...")
    client.flush(collection_name=COLLECTION_NAME)
    print("✓ Data flushed")

    # Create snapshot
    client.create_snapshot(
        collection_name=COLLECTION_NAME,
        snapshot_name=SNAPSHOT_NAME,
        description="Demo backup created for testing snapshot functionality"
    )
    print(f"✓ Snapshot '{SNAPSHOT_NAME}' created successfully")


def demo_list_snapshots(client: MilvusClient):
    """Demonstrate listing snapshots."""
    print("\n=== Listing Snapshots ===")

    # List all snapshots for the collection
    snapshots = client.list_snapshots(collection_name=COLLECTION_NAME)
    print(f"Found {len(snapshots)} snapshot(s) for collection '{COLLECTION_NAME}':")
    for snapshot in snapshots:
        print(f"  - {snapshot}")



def demo_describe_snapshot(client: MilvusClient):
    """Demonstrate describing a snapshot."""
    print("\n=== Describing Snapshot ===")

    info = client.describe_snapshot(snapshot_name=SNAPSHOT_NAME)

    print(f"Snapshot Information:")
    print(f"  Name:             {info['name']}")
    print(f"  Description:      {info['description']}")
    print(f"  Collection:       {info['collection_name']}")
    print(f"  Partitions:       {info['partition_names']}")
    print(f"  Create Timestamp: {info['create_ts']}")
    print(f"  S3 Location:      {info['s3_location']}")


def demo_restore_snapshot(client: MilvusClient):
    """Demonstrate restoring a snapshot with progress monitoring."""
    print("\n=== Restoring Snapshot ===")

    # Drop restored collection if it exists
    if client.has_collection(RESTORED_COLLECTION):
        print(f"Dropping existing collection '{RESTORED_COLLECTION}'...")
        client.drop_collection(RESTORED_COLLECTION)

    # Start restore operation
    print(f"Starting restore of snapshot '{SNAPSHOT_NAME}' to '{RESTORED_COLLECTION}'...")
    job_id = client.restore_snapshot(
        snapshot_name=SNAPSHOT_NAME,
        collection_name=RESTORED_COLLECTION
    )
    print(f"✓ Restore job started with ID: {job_id}")

    # Monitor restore progress
    print("\nMonitoring restore progress...")
    while True:
        state = client.get_restore_snapshot_state(job_id=job_id)

        job_id_val = state['job_id']
        snapshot_name = state['snapshot_name']
        collection_name = state['collection_name']
        progress = state['progress']
        state_value = state['state']
        time_cost = state['time_cost']

        print(f"  Job {job_id_val}: {snapshot_name} -> Collection {collection_name}")
        print(f"  State: {state_value}, Progress: {progress}%, Time: {time_cost}ms")

        # Check if completed
        if state_value == 2:  # RestoreSnapshotCompleted
            print(f"\n✓ Restore completed successfully in {time_cost}ms!")
            break
        elif state_value == 3:  # RestoreSnapshotFailed
            reason = state.get('reason', 'Unknown error')
            print(f"\n✗ Restore failed: {reason}")
            return False

        time.sleep(1)

    # Verify restored collection
    if client.has_collection(RESTORED_COLLECTION):
        # Load the collection before querying
        print(f"Loading collection '{RESTORED_COLLECTION}'...")
        client.load_collection(collection_name=RESTORED_COLLECTION)
        print("✓ Collection loaded")

        # Query to count entities
        num_entities = client.query(
            collection_name=RESTORED_COLLECTION,
            filter="",
            output_fields=["count(*)"]
        )
        print(f"✓ Restored collection has {num_entities[0]['count(*)']} entities")

    return True


def demo_list_restore_jobs(client: MilvusClient):
    """Demonstrate listing restore jobs."""
    print("\n=== Listing Restore Jobs ===")

    # List all restore jobs
    jobs = client.list_restore_snapshot_jobs()
    print(f"Found {len(jobs)} restore job(s):")

    for job in jobs:
        print(f"  Job {job['job_id']}:")
        print(f"    Snapshot: {job['snapshot_name']}")
        print(f"    Collection: {job['collection_name']}")
        print(f"    State: {job['state']}")
        print(f"    Progress: {job['progress']}%")
        print(f"    Time Cost: {job['time_cost']}ms")
        if job.get('reason'):
            print(f"    Reason: {job['reason']}")

    # List jobs for specific collection
    collection_jobs = client.list_restore_snapshot_jobs(
        collection_name=RESTORED_COLLECTION
    )
    print(f"\nJobs for collection '{RESTORED_COLLECTION}': {len(collection_jobs)}")


def demo_cleanup(client: MilvusClient):
    """Clean up demo resources."""
    print("\n=== Cleanup ===")

    # Drop snapshot
    print(f"Dropping snapshot '{SNAPSHOT_NAME}'...")
    client.drop_snapshot(snapshot_name=SNAPSHOT_NAME)
    print("✓ Snapshot dropped")

    # Drop collections
    for collection in [COLLECTION_NAME, RESTORED_COLLECTION]:
        if client.has_collection(collection):
            print(f"Dropping collection '{collection}'...")
            client.drop_collection(collection)
            print(f"✓ Collection '{collection}' dropped")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Milvus Snapshot API Demo")
    print("=" * 60)

    # Connect to Milvus
    print(f"\nConnecting to Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI)
    print("✓ Connected successfully")

    try:
        # Run demos
        create_demo_collection(client)
        demo_create_snapshot(client)
        demo_list_snapshots(client)
        demo_describe_snapshot(client)

        success = demo_restore_snapshot(client)
        if success:
            demo_list_restore_jobs(client)

        # Cleanup
        demo_cleanup(client)

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

        # Attempt cleanup on error
        try:
            demo_cleanup(client)
        except Exception as cleanup_error:
            print(f"Cleanup also failed: {cleanup_error}")

    finally:
        # Close connection
        print("\nClosing connection...")
        # Note: MilvusClient doesn't have explicit close method


async def async_demo():
    """
    Async version of snapshot demo using AsyncMilvusClient.
    Demonstrates all snapshot-related async APIs.
    """
    from pymilvus import AsyncMilvusClient
    import asyncio

    print("\n" + "=" * 60)
    print("Async Snapshot Demo - AsyncMilvusClient")
    print("=" * 60)

    # Connect to Milvus
    client = AsyncMilvusClient(uri="http://localhost:19530")

    try:
        collection_name = "async_snapshot_demo_collection"

        # 1. Create collection and insert data
        print("\n[1] Creating collection and inserting data...")

        # Create collection
        await client.create_collection(
            collection_name=collection_name,
            dimension=128,
        )
        print(f"✓ Created collection: {collection_name}")

        # Insert some data
        import random
        vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        data = [{"id": i, "vector": vectors[i]} for i in range(100)]
        await client.insert(collection_name=collection_name, data=data)
        print(f"✓ Inserted {len(data)} vectors")

        # Flush to ensure data is persisted
        await client.flush(collection_name)
        print("✓ Data flushed to storage")

        # 2. Create snapshot
        print("\n[2] Creating snapshot...")
        snapshot_name = "async_demo_snapshot_v1"
        await client.create_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name,
            description="Async demo snapshot with 100 vectors"
        )
        print(f"✓ Created snapshot: {snapshot_name}")

        # 3. List snapshots
        print("\n[3] Listing all snapshots...")
        snapshots = await client.list_snapshots(collection_name=collection_name)
        print(f"✓ Found {len(snapshots)} snapshot(s):")
        for snap in snapshots:
            print(f"  - {snap}")

        # 4. Describe snapshot
        print("\n[4] Describing snapshot...")
        snapshot_info = await client.describe_snapshot(snapshot_name=snapshot_name)
        print(f"✓ Snapshot details:")
        print(f"  Name: {snapshot_info['name']}")
        print(f"  Collection: {snapshot_info['collection_name']}")
        print(f"  Description: {snapshot_info['description']}")
        print(f"  Partitions: {snapshot_info['partition_names']}")

        # 5. Restore snapshot
        print("\n[5] Restoring snapshot to new collection...")
        restored_collection = "async_restored_collection"
        job_id = await client.restore_snapshot(
            snapshot_name=snapshot_name,
            collection_name=restored_collection
        )
        print(f"✓ Restore job started, job_id: {job_id}")

        # 6. Monitor restore progress
        print("\n[6] Monitoring restore progress...")
        while True:
            state = await client.get_restore_snapshot_state(job_id=job_id)
            progress = state["progress"]
            state_value = state["state"]

            print(f"  Job {job_id}: State {state_value} - {progress}% complete", end="\r")

            # Check if completed (state: 2=Completed, 3=Failed)
            if state_value == 2:  # RestoreSnapshotCompleted
                print()  # New line
                print(f"✓ Restore completed successfully!")
                break
            elif state_value == 3:  # RestoreSnapshotFailed
                print()  # New line
                print(f"✗ Restore failed: {state.get('reason', 'Unknown error')}")
                break

            await asyncio.sleep(1)

        # 7. List restore jobs
        print("\n[7] Listing all restore jobs...")
        jobs = await client.list_restore_snapshot_jobs()
        print(f"✓ Found {len(jobs)} restore job(s):")
        for job in jobs:
            print(f"  - Job {job['job_id']}: {job['snapshot_name']} -> "
                  f"{job['collection_name']} ({job['state']})")

        # Cleanup
        print("\n[Cleanup] Removing test resources...")
        try:
            await client.drop_collection(collection_name)
            print(f"✓ Dropped collection: {collection_name}")
        except Exception as e:
            print(f"  Warning: Failed to drop collection {collection_name}: {e}")

        try:
            await client.drop_collection(restored_collection)
            print(f"✓ Dropped collection: {restored_collection}")
        except Exception as e:
            print(f"  Warning: Failed to drop collection {restored_collection}: {e}")

        try:
            await client.drop_snapshot(snapshot_name=snapshot_name)
            print(f"✓ Dropped snapshot: {snapshot_name}")
        except Exception as e:
            print(f"  Warning: Failed to drop snapshot {snapshot_name}: {e}")

        print("\n" + "=" * 60)
        print("Async demo completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close connection
        print("\nClosing async connection...")
        # Note: AsyncMilvusClient will auto-close


if __name__ == "__main__":
    import sys

    # Run sync demo by default
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        # Run async demo
        import asyncio
        asyncio.run(async_demo())
    else:
        # Run sync demo
        main()
