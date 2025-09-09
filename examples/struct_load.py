
import time
from pymilvus import (
    MilvusClient,
    EmbeddingList,
)
import numpy as np

DIM = 16  # Updated to match prepare-data-struct.py

COLLECTION_NAME = "Documents"
milvus_client = MilvusClient("http://localhost:19530")
milvus_client.load_collection(COLLECTION_NAME)

result = milvus_client.query(
    collection_name=COLLECTION_NAME,
    filter="",
    output_fields=["struct_float_vec2"],
    limit=2,
)

print(result)

rng = np.random.default_rng(seed=19530)



# Create search queries using EmbeddingList - much cleaner!
# For testing purposes, using random test data
queries = [
    EmbeddingList.from_random_test(7, DIM, seed=19530),  # Query with 7 vectors
    EmbeddingList.from_random_test(4, DIM, seed=19531),  # Query with 4 vectors  
]

embeddingList = EmbeddingList()
# Query with 2 vectors
embeddingList.add(np.random.randn(DIM))
embeddingList.add(np.random.randn(DIM))

queries.append(embeddingList)

# In production, you would use real embeddings:
field = "struct_float_vec2"
res = milvus_client.search(COLLECTION_NAME, data=queries, limit=2, anns_field=field,
                     output_fields=["struct_field"])

for i, (hits, query) in enumerate(zip(res, queries)):
    print(f"============== Query {i+1} ({query}) - Total hits: {len(hits)}")
    for hit in hits:
        print(f"    Hit id: {hit.id}, distance: {hit.distance}")
        print(f"        entity: {hit}")
