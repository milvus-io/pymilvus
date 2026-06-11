import asyncio

import numpy as np

from pymilvus import AsyncMilvusClient

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "async_example_collection"
DIMENSION = 8
INITIAL_ROWS = 1000
PARALLEL_INSERT_TASKS = 5
ROWS_PER_TASK = 50
PARALLEL_SEARCH_TASKS = 10
SEARCH_LIMIT = 5

rng = np.random.default_rng(seed=19530)


def build_rows(count: int):
    vectors = rng.random((count, DIMENSION)).astype("float32")
    return [{"vector": vector.tolist()} for vector in vectors]


async def insert_parallel(
    client: AsyncMilvusClient,
    collection_name: str,
    num_tasks: int = PARALLEL_INSERT_TASKS,
    rows_per_task: int = ROWS_PER_TASK,
) -> None:
    async def insert_batch(batch_id: int) -> None:
        result = await client.insert(collection_name, build_rows(rows_per_task))
        print(f"Batch {batch_id}: inserted {result['insert_count']} entities")

    await asyncio.gather(*(insert_batch(i) for i in range(num_tasks)))
    print(f"Total parallel inserts completed: {num_tasks * rows_per_task} entities")


async def search_parallel(
    client: AsyncMilvusClient,
    collection_name: str,
    num_tasks: int = PARALLEL_SEARCH_TASKS,
) -> None:
    async def search_once(task_id: int) -> None:
        query_vector = rng.random((1, DIMENSION)).astype("float32").tolist()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        result = await client.search(
            collection_name,
            data=query_vector,
            anns_field="vector",
            search_params=search_params,
            limit=SEARCH_LIMIT,
        )
        print(f"Search {task_id}: found {len(result[0])} results")

    await asyncio.gather(*(search_once(i) for i in range(num_tasks)))
    print(f"Total parallel searches completed: {num_tasks}")


async def main() -> None:
    async with AsyncMilvusClient(uri=MILVUS_URI) as client:
        if await client.has_collection(COLLECTION_NAME):
            await client.drop_collection(COLLECTION_NAME)

        try:
            await client.create_collection(
                COLLECTION_NAME,
                dimension=DIMENSION,
                auto_id=True,
                metric_type="COSINE",
            )

            insert_result = await client.insert(COLLECTION_NAME, build_rows(INITIAL_ROWS))
            print(f"Inserted {insert_result['insert_count']} entities")

            query_vector = rng.random((1, DIMENSION)).astype("float32").tolist()
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            result = await client.search(
                COLLECTION_NAME,
                data=query_vector,
                anns_field="vector",
                search_params=search_params,
                limit=SEARCH_LIMIT,
            )
            print("Search result IDs:", [hit["id"] for hit in result[0]])

            print("\n--- Testing parallel inserts ---")
            await insert_parallel(client, COLLECTION_NAME)

            print("\n--- Testing parallel searches ---")
            await search_parallel(client, COLLECTION_NAME)
        finally:
            if await client.has_collection(COLLECTION_NAME):
                await client.release_collection(COLLECTION_NAME)
                await client.drop_collection(COLLECTION_NAME)


if __name__ == "__main__":
    asyncio.run(main())
