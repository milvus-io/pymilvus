# Warning! needs a running Milvus

import asyncio
from pymilvus.aio import AsyncMilvusClient

HOST = "127.0.0.1"
PORT = "19530"


async def run(aio_client, num):
    version = await aio_client.get_server_version()
    print(f"run: {num}: {version}")
    return 


async def main():
    aio_client = AsyncMilvusClient(HOST, PORT)

    tasks = []
    for i in range(10):
        task = asyncio.create_task(run(aio_client, i))
        tasks.append(task)

    for t in tasks:
        await t

if __name__ == "__main__":
    asyncio.run(main())
