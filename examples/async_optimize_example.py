import asyncio

from pymilvus import AsyncMilvusClient

URI = "http://localhost:19530"

async def example_sync_optimize():
    """Example: Wait for optimization to complete"""
    client = AsyncMilvusClient(uri=URI)
    await client.create_collection("my_collection", dimension=128)

    result = await client.optimize("my_collection", target_size="1GB", wait=True)
    print(f"Optimization completed: {result.status}")
    print(f"Compaction ID: {result.compaction_id}")
    print(f"Progress stages: {result.progress}")


async def example_async_optimize_with_progress():
    """Example: Track optimization progress asynchronously"""
    client = AsyncMilvusClient(uri=URI)

    task = await client.optimize("my_collection", target_size="1GB", wait=False)

    while not task.done():
        print(f"Current progress: {task.progress()}")
        await asyncio.sleep(1)

    result = await task.result()
    print(f"Optimization completed: {result.status}")


async def example_cancel_optimize():
    """Example: Cancel an ongoing optimization"""
    client = AsyncMilvusClient(uri=URI)

    task = await client.optimize("my_collection", target_size="512MB", wait=False)

    await asyncio.sleep(2)

    if task.cancel():
        print("Optimization cancelled successfully")
    else:
        print("Could not cancel - task already completed")


if __name__ == "__main__":
    asyncio.run(example_sync_optimize())
