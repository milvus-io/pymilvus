import asyncio

from pymilvus import AsyncMilvusClient


async def example_sync_optimize():
    """Example: Wait for optimization to complete"""
    client = AsyncMilvusClient(uri="http://localhost:19530")

    result = await client.optimize("my_collection", target_size="512MB", wait=True)
    print(f"Optimization completed: {result.status}")
    print(f"Compaction ID: {result.compaction_id}")
    print(f"Progress stages: {result.progress}")


async def example_async_optimize_with_progress():
    """Example: Track optimization progress asynchronously"""
    client = AsyncMilvusClient(uri="http://localhost:19530")

    task = await client.optimize("my_collection", target_size="512MB", wait=False)

    while not task.done():
        print(f"Current progress: {task.progress()}")
        await asyncio.sleep(1)

    result = await task.result()
    print(f"Optimization completed: {result.status}")


async def example_cancel_optimize():
    """Example: Cancel an ongoing optimization"""
    client = AsyncMilvusClient(uri="http://localhost:19530")

    task = await client.optimize("my_collection", target_size="512MB", wait=False)

    await asyncio.sleep(2)

    if task.cancel():
        print("Optimization cancelled successfully")
    else:
        print("Could not cancel - task already completed")


async def example_multiple_optimizations():
    """Example: Run multiple optimizations concurrently"""
    client = AsyncMilvusClient(uri="http://localhost:19530")

    task1 = await client.optimize("collection1", target_size="256MB", wait=False)
    task2 = await client.optimize("collection2", target_size="512MB", wait=False)
    task3 = await client.optimize("collection3", target_size="1GB", wait=False)

    tasks = [task1, task2, task3]

    while any(not task.done() for task in tasks):
        for i, task in enumerate(tasks, 1):
            if not task.done():
                print(f"Collection {i}: {task.progress()}")
        await asyncio.sleep(1)

    results = await asyncio.gather(
        task1.result(),
        task2.result(),
        task3.result(),
        return_exceptions=True
    )

    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"Collection {i} failed: {result}")
        else:
            print(f"Collection {i} completed: {result.status}")


if __name__ == "__main__":
    asyncio.run(example_sync_optimize())
