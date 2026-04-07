"""
Milvus External Collection API Usage Examples

This script demonstrates the full lifecycle of an external collection:
1. Generate Parquet test data and upload to MinIO
2. Create an external collection with field mappings
3. Refresh to sync data from MinIO
4. Monitor refresh progress
5. Create index, load, and search
6. List refresh jobs
7. Cleanup

Requirements:
- Milvus server running with external table support (standalone, local MinIO)
- pymilvus, pyarrow, minio
- pip install pyarrow minio

Usage:
    python external_collection_example.py          # sync version
    python external_collection_example.py --async  # async version
"""

import asyncio
import io
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

from pymilvus import AsyncMilvusClient, DataType, MilvusClient

# Configuration — adjust these to match your environment
MILVUS_URI = "http://localhost:19530"
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "a-bucket"  # same bucket Milvus uses
EXTERNAL_DATA_PREFIX = "external-demo"  # path prefix in MinIO

COLLECTION_NAME = "product_embeddings"
DIM = 128
NUM_ROWS = 5000
NUM_FILES = 5


# ============================================================
# Step 1: Prepare test data in MinIO
# ============================================================


def prepare_test_data():
    """Generate Parquet files with test data and upload to MinIO."""
    print("\n=== Preparing Test Data in MinIO ===")

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    # Clean up old data
    try:
        objects = minio_client.list_objects(MINIO_BUCKET, prefix=f"{EXTERNAL_DATA_PREFIX}/")
        for obj in objects:
            minio_client.remove_object(MINIO_BUCKET, obj.object_name)
    except Exception:
        pass

    rows_per_file = NUM_ROWS // NUM_FILES
    total_uploaded = 0

    for i in range(NUM_FILES):
        start_id = i * rows_per_file
        ids = list(range(start_id, start_id + rows_per_file))
        names = [f"product_{idx}" for idx in ids]

        # Vectors stored as FixedSizeList<Float32> — Milvus auto-normalizes to FixedSizeBinary
        vectors = np.random.rand(rows_per_file, DIM).astype(np.float32)

        table = pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "name": pa.array(names, type=pa.string()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    vectors.flatten(), list_size=DIM
                ),
            }
        )

        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)

        object_name = f"{EXTERNAL_DATA_PREFIX}/data_{i}.parquet"
        minio_client.put_object(
            MINIO_BUCKET, object_name, buf, length=buf.getbuffer().nbytes
        )
        total_uploaded += rows_per_file

    print(f"  Uploaded {NUM_FILES} Parquet files ({total_uploaded} rows) to "
          f"{MINIO_BUCKET}/{EXTERNAL_DATA_PREFIX}/")
    # external_source is a path relative to the Milvus-configured MinIO bucket root,
    # NOT a full s3:// URI. Milvus already knows the bucket from its config.
    return f"{EXTERNAL_DATA_PREFIX}/"


# ============================================================
# Step 2-6: External collection operations
# ============================================================


def demo_create_external_collection(client: MilvusClient, external_source: str):
    """Create an external collection with field mappings."""
    print("\n=== Creating External Collection ===")

    # External tables do not support: primary key, auto_id, dynamic field,
    # partition key, clustering key, or functions.
    schema = client.create_schema(
        external_source=external_source,
        external_spec='{"format": "parquet"}',
    )

    # Every field must declare external_field to map to a column in the Parquet files
    schema.add_field(
        field_name="product_id", datatype=DataType.INT64, external_field="id"
    )
    schema.add_field(
        field_name="product_name",
        datatype=DataType.VARCHAR,
        max_length=256,
        external_field="name",
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=DIM,
        external_field="vector",
    )

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    # Verify with describe
    info = client.describe_collection(COLLECTION_NAME)
    print(f"  Created '{COLLECTION_NAME}'")
    print(f"  external_source: {info.get('external_source')}")
    print(f"  external_spec:   {info.get('external_spec')}")
    for f in info["fields"]:
        ef = f.get("external_field", "")
        if ef:
            print(f"  field '{f['name']}' -> '{ef}'")


def demo_refresh_and_wait(client: MilvusClient):
    """Trigger refresh and wait for completion."""
    print("\n=== Refreshing External Collection ===")

    job_id = client.refresh_external_collection(collection_name=COLLECTION_NAME)
    print(f"  Refresh job started: job_id={job_id}")

    while True:
        progress = client.get_refresh_external_collection_progress(job_id=job_id)
        print(f"  {progress.state}: {progress.progress}%")

        if progress.state == "RefreshCompleted":
            elapsed = progress.end_time - progress.start_time
            print(f"  Completed in {elapsed}ms")
            return job_id
        elif progress.state == "RefreshFailed":
            print(f"  Failed: {progress.reason}")
            return job_id

        time.sleep(2)


def demo_index_load_search(client: MilvusClient):
    """Create index, load, and search the external collection."""
    print("\n=== Index + Load + Search ===")

    # Create vector index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", index_type="AUTOINDEX", metric_type="L2"
    )
    client.create_index(COLLECTION_NAME, index_params)
    print("  Index created")

    # Load collection
    client.load_collection(COLLECTION_NAME)
    print("  Collection loaded")

    # Search
    query_vector = np.random.rand(1, DIM).astype(np.float32).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=5,
        output_fields=["product_id", "product_name"],
        anns_field="embedding",
    )
    print(f"  Search returned {len(results[0])} results:")
    for hit in results[0]:
        print(
            f"    id={hit['entity']['product_id']}, "
            f"name={hit['entity']['product_name']}, "
            f"distance={hit['distance']:.4f}"
        )


def demo_list_refresh_jobs(client: MilvusClient):
    """List all refresh jobs."""
    print("\n=== Listing Refresh Jobs ===")

    jobs = client.list_refresh_external_collection_jobs(
        collection_name=COLLECTION_NAME
    )
    print(f"  Found {len(jobs)} job(s):")
    for job in jobs:
        print(
            f"    Job {job.job_id}: state={job.state}, "
            f"progress={job.progress}%"
        )


def demo_cleanup(client: MilvusClient):
    """Drop collection and clean MinIO data."""
    print("\n=== Cleanup ===")
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"  Dropped '{COLLECTION_NAME}'")

    # Clean MinIO test data
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    objects = list(
        minio_client.list_objects(MINIO_BUCKET, prefix=f"{EXTERNAL_DATA_PREFIX}/")
    )
    for obj in objects:
        minio_client.remove_object(MINIO_BUCKET, obj.object_name)
    print(f"  Cleaned {len(objects)} files from MinIO")


# ============================================================
# Async version
# ============================================================


async def async_demo():
    """Async version of the full demo."""
    print("\n" + "=" * 50)
    print("ASYNC EXTERNAL COLLECTION DEMO")
    print("=" * 50)

    external_source = prepare_test_data()

    client = AsyncMilvusClient(uri=MILVUS_URI)

    schema = client.create_schema(
        external_source=external_source,
        external_spec='{"format": "parquet"}',
    )
    schema.add_field(
        field_name="product_id", datatype=DataType.INT64, external_field="id"
    )
    schema.add_field(
        field_name="product_name",
        datatype=DataType.VARCHAR,
        max_length=256,
        external_field="name",
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=DIM,
        external_field="vector",
    )

    if await client.has_collection(COLLECTION_NAME):
        await client.drop_collection(COLLECTION_NAME)
    await client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
    print(f"  Created '{COLLECTION_NAME}' (async)")

    # Refresh
    job_id = await client.refresh_external_collection(
        collection_name=COLLECTION_NAME
    )
    print(f"  Refresh started: job_id={job_id}")

    while True:
        progress = await client.get_refresh_external_collection_progress(
            job_id=job_id
        )
        print(f"  {progress.state}: {progress.progress}%")
        if progress.state in ("RefreshCompleted", "RefreshFailed"):
            break
        await asyncio.sleep(2)

    # Index + Load + Search
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding", index_type="AUTOINDEX", metric_type="L2"
    )
    await client.create_index(COLLECTION_NAME, index_params)
    await client.load_collection(COLLECTION_NAME)
    print("  Index created + collection loaded")

    query_vector = np.random.rand(1, DIM).astype(np.float32).tolist()
    results = await client.search(
        collection_name=COLLECTION_NAME,
        data=query_vector,
        limit=5,
        output_fields=["product_id", "product_name"],
        anns_field="embedding",
    )
    print(f"  Search returned {len(results[0])} results")

    # List jobs
    jobs = await client.list_refresh_external_collection_jobs(
        collection_name=COLLECTION_NAME
    )
    print(f"  Total refresh jobs: {len(jobs)}")

    # Cleanup
    await client.drop_collection(COLLECTION_NAME)
    print("  Cleanup done (async)")
    await client.close()

    # Clean MinIO
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    for obj in minio_client.list_objects(
        MINIO_BUCKET, prefix=f"{EXTERNAL_DATA_PREFIX}/"
    ):
        minio_client.remove_object(MINIO_BUCKET, obj.object_name)
    print("  MinIO cleaned")


# ============================================================
# Main
# ============================================================


def main():
    """Run the sync demo."""
    print("=" * 50)
    print("EXTERNAL COLLECTION API DEMO")
    print("=" * 50)

    external_source = prepare_test_data()
    client = MilvusClient(uri=MILVUS_URI)

    try:
        demo_create_external_collection(client, external_source)
        demo_refresh_and_wait(client)
        demo_index_load_search(client)
        demo_list_refresh_jobs(client)
    finally:
        demo_cleanup(client)
        client.close()


if __name__ == "__main__":
    if "--async" in sys.argv:
        asyncio.run(async_demo())
    else:
        main()
