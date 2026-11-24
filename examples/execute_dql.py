"""Example demonstrating synchronous and asynchronous ``execute_dql`` (``DQL.search``)."""

import asyncio

import numpy as np

from pymilvus import DQL, AsyncMilvusClient, DataType, MilvusClient

URI = "http://localhost:19530"
ARCHIVE_DB = "execute_dql_archive"
PAPERS_COL = "execute_dql_papers"
GUIDES_COL = "execute_dql_guidelines"
DIM = 8

FMT = "\n=== {:48} ===\n"


def _setup_collection(client: MilvusClient, collection_name: str, rows: list) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    schema.add_field("year", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )
    client.insert(collection_name, rows)
    client.load_collection(collection_name)


def run_sync_example(query_vec: list) -> None:
    client = MilvusClient(URI)
    archive_client = None

    try:
        # Prepare a collection in the default database
        rng = np.random.default_rng(seed=19530)
        paper_rows = [
            {
                "id": i,
                "embedding": rng.random(DIM).astype(np.float32).tolist(),
                "title": f"paper-{i}",
                "year": 2020 + (i % 5),
            }
            for i in range(10)
        ]
        _setup_collection(client, PAPERS_COL, paper_rows)

        # Prepare a second collection in a separate database
        if ARCHIVE_DB not in client.list_databases():
            client.create_database(ARCHIVE_DB)
        archive_client = MilvusClient(URI, db_name=ARCHIVE_DB)
        guide_rows = [
            {
                "id": 100 + i,
                "embedding": rng.random(DIM).astype(np.float32).tolist(),
                "title": f"guideline-{i}",
                "year": 2022 + (i % 3),
            }
            for i in range(6)
        ]
        _setup_collection(archive_client, GUIDES_COL, guide_rows)

        # 1. Execute multiple independent searches across collections & databases in one RPC
        print(FMT.format("Sync execute_dql (cross-collection & cross-DB)"))
        reqs = [
            DQL.search(
                PAPERS_COL,
                data=[query_vec],
                anns_field="embedding",
                filter="year >= 2023",
                limit=3,
                output_fields=["title", "year"],
            ),
            DQL.search(
                PAPERS_COL,
                data=[query_vec],
                anns_field="embedding",
                filter="year < 2022",
                limit=2,
                output_fields=["title", "year"],
            ),
            DQL.search(
                GUIDES_COL,
                data=[query_vec],
                anns_field="embedding",
                limit=3,
                output_fields=["title", "year"],
                db_name=ARCHIVE_DB,
            ),
        ]

        # `timeout` is a server-side shared execution budget (in seconds), NOT a gRPC deadline.
        results = client.execute_dql(reqs, max_concurrency=4, timeout=5.0)

        # Strict unpacking when all items are expected to succeed:
        recent_papers, older_papers, archive_guides = [item.unwrap() for item in results]
        print("recent papers:", [hit["entity"]["title"] for hit in recent_papers[0]])
        print("older papers:", [hit["entity"]["title"] for hit in older_papers[0]])
        print("archive guides:", [hit["entity"]["title"] for hit in archive_guides[0]])

        # 2. Tolerant reading when one sub-request fails (e.g., missing collection)
        print(FMT.format("Sync execute_dql (partial failure handling)"))
        mixed_results = client.execute_dql(
            [
                DQL.search(PAPERS_COL, data=[query_vec], anns_field="embedding", limit=2),
                DQL.search("non_existent_collection", data=[query_vec], limit=2),
            ],
            max_concurrency=2,
            timeout=5.0,
        )
        for idx, item in enumerate(mixed_results):
            if not item.ok:
                print(f"reqs[{idx}] failed: code={item.error.code}, message={item.error.message}")
            else:
                print(f"reqs[{idx}] succeeded: ids={[hit['id'] for hit in item.result[0]]}")

    finally:
        if client.has_collection(PAPERS_COL):
            client.drop_collection(PAPERS_COL)
        if archive_client is not None:
            if archive_client.has_collection(GUIDES_COL):
                archive_client.drop_collection(GUIDES_COL)
            archive_client.close()
        if ARCHIVE_DB in client.list_databases():
            client.drop_database(ARCHIVE_DB)
        client.close()


async def run_async_example(query_vec: list) -> None:
    print(FMT.format("Async execute_dql"))
    async_client = AsyncMilvusClient(URI)
    try:
        # Prepare a temporary collection for the async call
        sync_client = MilvusClient(URI)
        try:
            rng = np.random.default_rng(seed=42)
            rows = [
                {
                    "id": i,
                    "embedding": rng.random(DIM).astype(np.float32).tolist(),
                    "title": f"async-paper-{i}",
                    "year": 2024,
                }
                for i in range(5)
            ]
            _setup_collection(sync_client, PAPERS_COL, rows)
        finally:
            sync_client.close()

        results = await async_client.execute_dql(
            [
                DQL.search(
                    PAPERS_COL,
                    data=[query_vec],
                    anns_field="embedding",
                    limit=2,
                    output_fields=["title"],
                ),
                DQL.search(
                    PAPERS_COL,
                    data=[query_vec],
                    anns_field="embedding",
                    filter="id >= 3",
                    limit=2,
                    output_fields=["title"],
                ),
            ],
            max_concurrency=2,
            timeout=5.0,
        )
        for idx, item in enumerate(results):
            hits = item.unwrap()[0]
            print(f"async reqs[{idx}]: {[hit['entity']['title'] for hit in hits]}")
    finally:
        if await async_client.has_collection(PAPERS_COL):
            await async_client.drop_collection(PAPERS_COL)
        await async_client.close()


if __name__ == "__main__":
    vec = np.random.default_rng(seed=7).random(DIM).astype(np.float32).tolist()
    run_sync_example(vec)
    asyncio.run(run_async_example(vec))
