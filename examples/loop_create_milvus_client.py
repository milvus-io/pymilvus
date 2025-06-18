import time
import numpy as np
from pymilvus import (
    MilvusClient,
)

fmt = "\n=== {:30} ===\n"
dim = 8
collection_name = "hello_milvus"

def play_func():
    for i in range(100):
        milvus_client = MilvusClient("http://localhost:19530")
        has_collection = milvus_client.has_collection(collection_name, timeout=5)

if __name__ == "__main__":
    play_func()
