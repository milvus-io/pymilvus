"""
Columnar Search Result Example

This example demonstrates the columnar search result feature, which is enabled by default 
for all search operations. Columnar storage provides significantly better performance 
and memory efficiency.

Key Benefits:
1. O(1) initialization instead of O(nq x topk)
2. Minimal memory usage (references protobuf data instead of copying)
3. Lazy access - data extracted on-demand

Usage:
    # Columnar search is enabled by default
    result = client.search(collection_name, search_vectors, limit=5)

    # The result can be used exactly like regular search results
    for hits in result:
        for hit in hits:
            print(hit.id, hit.distance, hit['field_name'])
"""

import numpy as np
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)

fmt = "\n=== {:30} ===\n"
dim = 8
num_entities = 1000
collection_name = "columnar_demo"

# Connect to Milvus
milvus_client = MilvusClient("http://localhost:19530")

# Clean up existing collection
if milvus_client.has_collection(collection_name, timeout=5):
    milvus_client.drop_collection(collection_name)

# Create collection schema with multiple fields for demonstration
schema = milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("vector2", DataType.FLOAT_VECTOR, dim=dim)
schema.add_field("category", DataType.VARCHAR, max_length=100)
schema.add_field("score", DataType.FLOAT)

# Create index
index_params = milvus_client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="L2", nlist=128)
index_params.add_index(field_name="vector2", index_type="IVF_FLAT", metric_type="L2", nlist=128)

print(fmt.format("Create collection"))
milvus_client.create_collection(
    collection_name,
    schema=schema,
    index_params=index_params,
    consistency_level="Strong",
)

# Insert data
print(fmt.format("Insert data"))
rng = np.random.default_rng(seed=42)
categories = ["electronics", "clothing", "food", "books", "toys"]
rows = [
    {
        "id": i,
        "vector": rng.random(dim).tolist(),
        "vector2": rng.random(dim).tolist(),
        "category": categories[i % len(categories)],
        "score": float(rng.random()),
        "extra_info": f"item_{i}",  # Dynamic field
    }
    for i in range(num_entities)
]
milvus_client.insert(collection_name, rows)

# Load collection
milvus_client.load_collection(collection_name)

# Search vector
search_vectors = rng.random((3, dim)).tolist()

# ==============================================================================
# Basic Search (Now Columnar by Default)
# ==============================================================================
print(fmt.format("Basic Search (Default Columnar)"))
result = milvus_client.search(
    collection_name,
    data=search_vectors,
    anns_field="vector",
    limit=5,
    output_fields=["category", "score", "extra_info"],
)

print("Search results:")
for i, hits in enumerate(result):
    print(f"\nQuery {i}:")
    for hit in hits:
        # Access fields just like regular search result
        print(f"  ID: {hit.id}, Distance: {hit.distance:.4f}")
        print(f"    Category: {hit['category']}, Score: {hit['score']:.4f}")
        print(f"    Extra Info: {hit['extra_info']}")

# ==============================================================================
# Hybrid Search (Also Columnar by Default)
# ==============================================================================
print(fmt.format("Hybrid Search (Default Columnar)"))

# Create search requests for multiple vector fields
req1 = AnnSearchRequest(
    data=[search_vectors[0]],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=5,
)
req2 = AnnSearchRequest(
    data=[search_vectors[0]],
    anns_field="vector2",
    param={"metric_type": "L2"},
    limit=5,
)

hybrid_result = milvus_client.hybrid_search(
    collection_name,
    [req1, req2],
    ranker=RRFRanker(),
    limit=5,
    output_fields=["category", "score"],
)

print("Hybrid search results:")
for hits in hybrid_result:
    for hit in hits:
        print(f"  ID: {hit.id}, Distance: {hit.distance:.4f}, Category: {hit['category']}")

# ==============================================================================
# API Compatibility Demo
# ==============================================================================
print(fmt.format("API Compatibility"))

# Columnar result supports all the same access patterns
result = milvus_client.search(
    collection_name,
    data=[search_vectors[0]],
    anns_field="vector",
    limit=3,
    output_fields=["category", "score"],
)

hits = result[0]  # Get first query's results
print("Various access patterns:")

# 1. Indexing
print(f"\n1. Indexing: hits[0] = {hits[0]}")

# 2. Slicing
print(f"2. Slicing: hits[0:2] = {hits[0:2]}")

# 3. Length
print(f"3. Length: len(hits) = {len(hits)}")

# 4. Iteration
print("4. Iteration:")
for hit in hits:
    print(f"   - {hit.id}: {hit.distance:.4f}")

# 5. Dict-like access
hit = hits[0]
print(f"\n5. Dict-like access:")
print(f"   hit['category'] = {hit['category']}")
print(f"   hit.get('score', 0) = {hit.get('score', 0)}")
print(f"   'category' in hit = {'category' in hit}")

# 6. Property access
print(f"\n6. Property access:")
print(f"   hit.id = {hit.id}")
print(f"   hit.distance = {hit.distance}")
print(f"   hit.pk = {hit.pk}")  # Alias for id
print(f"   hit.score = {hit.score}")  # Alias for distance

# 7. Convert to dict
print(f"\n7. Convert to dict: hit.to_dict() = {hit.to_dict()}")

# Clean up
print(fmt.format("Clean up"))
milvus_client.drop_collection(collection_name)
print("Done!")
