from pymilvus import (
    MilvusClient,
)

milvus_client = MilvusClient("http://localhost:19530")

version = milvus_client.get_server_version()
print(f"server version: {version}")
