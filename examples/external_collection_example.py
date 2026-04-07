"""
Milvus External Collection API Usage Examples

This script demonstrates how to use the external collection functionality in Milvus for:
- Creating external collections with external data source mapping
- Refreshing external collection data from external sources
- Monitoring refresh job progress
- Listing refresh jobs

External collections allow Milvus to query data stored in external systems (e.g., Iceberg,
Hudi, Delta Lake on S3/GCS) by mapping external fields to Milvus schema fields.

Requirements:
- Milvus server running with external table support
- pymilvus with external collection API support
- An accessible external data source (e.g., S3 bucket with Iceberg table)
"""

import asyncio
import sys
import time

from pymilvus import AsyncMilvusClient, MilvusClient, DataType

# Configuration
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "my_external_table"


def demo_create_external_collection(client: MilvusClient):
    """Create an external collection with field mappings to an external data source."""
    print("\n=== Creating External Collection ===")

    # Define schema with external field mappings
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        # External data source configuration
        external_source="s3://my-data-lake/iceberg/user_embeddings",
        external_spec='{"format": "iceberg", "catalog": "glue", "database": "analytics"}',
    )

    # Fields with external_field specify the mapping to columns in the external data source
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        external_field="row_id",  # Maps to "row_id" column in external table
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1024,
        external_field="content",  # Maps to "content" column in external table
    )
    # Vector field without external_field - generated/managed by Milvus
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=768,
    )

    # Drop existing collection if any
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    print(f"Created external collection '{COLLECTION_NAME}'")
    print(f"  External source: s3://my-data-lake/iceberg/user_embeddings")
    print(f"  Field mappings: id->row_id, text->content")


def demo_refresh_external_collection(client: MilvusClient):
    """Trigger a data refresh to sync data from the external source."""
    print("\n=== Refreshing External Collection ===")

    # Trigger refresh - data will be synced from the external source
    job_id = client.refresh_external_collection(
        collection_name=COLLECTION_NAME,
    )
    print(f"Refresh job started: job_id={job_id}")

    # Monitor progress until completion
    while True:
        progress = client.get_refresh_external_collection_progress(job_id=job_id)
        print(
            f"  State: {progress.state}, Progress: {progress.progress}%, "
            f"Source: {progress.external_source}"
        )

        if progress.state == "RefreshCompleted":
            elapsed = progress.end_time - progress.start_time
            print(f"Refresh completed in {elapsed}ms")
            break
        elif progress.state == "RefreshFailed":
            print(f"Refresh failed: {progress.reason}")
            break

        time.sleep(2)


def demo_refresh_with_new_source(client: MilvusClient):
    """Refresh an external collection with a new/updated data source."""
    print("\n=== Refreshing With New Source ===")

    # Optionally override the external source for this refresh
    job_id = client.refresh_external_collection(
        collection_name=COLLECTION_NAME,
        external_source="s3://my-data-lake/iceberg/user_embeddings_v2",
        external_spec='{"format": "iceberg", "catalog": "glue", "snapshot": "latest"}',
    )
    print(f"Refresh with new source started: job_id={job_id}")

    # Poll for completion
    while True:
        progress = client.get_refresh_external_collection_progress(job_id=job_id)
        if progress.state in ("RefreshCompleted", "RefreshFailed"):
            print(f"Final state: {progress.state}")
            break
        time.sleep(2)


def demo_list_refresh_jobs(client: MilvusClient):
    """List all refresh jobs for an external collection."""
    print("\n=== Listing Refresh Jobs ===")

    # List jobs for a specific collection
    jobs = client.list_refresh_external_collection_jobs(
        collection_name=COLLECTION_NAME,
    )

    if not jobs:
        print("No refresh jobs found.")
        return

    for job in jobs:
        print(
            f"  Job {job.job_id}: state={job.state}, "
            f"progress={job.progress}%, "
            f"collection={job.collection_name}"
        )

    # List all jobs across all collections
    print("\nAll refresh jobs:")
    all_jobs = client.list_refresh_external_collection_jobs()
    print(f"  Total jobs: {len(all_jobs)}")


def demo_cleanup(client: MilvusClient):
    """Clean up the demo collection."""
    print("\n=== Cleanup ===")
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped collection '{COLLECTION_NAME}'")


# ============================================================
# Async version
# ============================================================

async def async_demo():
    """Async version of the external collection demo."""
    print("\n" + "=" * 50)
    print("ASYNC EXTERNAL COLLECTION DEMO")
    print("=" * 50)

    client = AsyncMilvusClient(uri=MILVUS_URI)

    # Create external collection (reuse sync schema creation)
    from pymilvus.orm.schema import CollectionSchema, FieldSchema

    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.INT64, external_field="row_id"),
            FieldSchema("text", DataType.VARCHAR, max_length=1024, external_field="content"),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
        ],
        external_source="s3://my-data-lake/iceberg/user_embeddings",
        external_spec='{"format": "iceberg", "catalog": "glue"}',
    )

    if await client.has_collection(COLLECTION_NAME):
        await client.drop_collection(COLLECTION_NAME)
    await client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"Created external collection '{COLLECTION_NAME}' (async)")

    # Refresh
    job_id = await client.refresh_external_collection(collection_name=COLLECTION_NAME)
    print(f"Refresh started: job_id={job_id}")

    # Monitor
    while True:
        progress = await client.get_refresh_external_collection_progress(job_id=job_id)
        print(f"  {progress.state}: {progress.progress}%")
        if progress.state in ("RefreshCompleted", "RefreshFailed"):
            break
        await asyncio.sleep(2)

    # List jobs
    jobs = await client.list_refresh_external_collection_jobs(collection_name=COLLECTION_NAME)
    print(f"Total refresh jobs: {len(jobs)}")

    # Cleanup
    await client.drop_collection(COLLECTION_NAME)
    print("Cleanup done (async)")
    await client.close()


# ============================================================
# Main
# ============================================================

def main():
    """Run the sync demo."""
    print("=" * 50)
    print("EXTERNAL COLLECTION API DEMO")
    print("=" * 50)

    client = MilvusClient(uri=MILVUS_URI)

    try:
        demo_create_external_collection(client)
        demo_refresh_external_collection(client)
        demo_refresh_with_new_source(client)
        demo_list_refresh_jobs(client)
    finally:
        demo_cleanup(client)
        client.close()


if __name__ == "__main__":
    if "--async" in sys.argv:
        asyncio.run(async_demo())
    else:
        main()
