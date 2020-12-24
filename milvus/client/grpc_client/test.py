import asyncio
from grpc_client import GRPCClient


async def main():
    t = GRPCClient("127.0.0.1:19530", False, 0, "")
    print(await t.server_version())



if __name__ == "__main__":
    asyncio.run(main())

