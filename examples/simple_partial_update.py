#!/usr/bin/env python3

"""
Simple partial update example for Milvus.

This example demonstrates the difference between partial and full upsert operations:

1. partial_update=True: Update only specified fields, preserve others
   - Only provide fields you want to change
   - Vectors and other fields remain unchanged
   - Ideal for metadata updates

2. partial_update=False (default): Replace entire entity
   - Must provide ALL required fields including vectors
   - Missing fields may cause errors or data loss
   - Use for complete entity replacement

Key takeaway: Always use partial_update=True when updating subset of fields!
"""

import numpy as np
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

logger.info("Starting simple partial update demo")
logger.warning("Setting up collection with explicit schema and data...")
logger.debug("Setting up collection with explicit schema and data...")
# Setup
dim = 8
collection_name = "simple_partial_update"
client = MilvusClient("http://localhost:19530")

print("=== Simple Partial Update Demo ===\n")

# 1. Create collection with explicit schema and insert data
print("1. Setting up collection with explicit schema and data...")
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# Define schema explicitly
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False, description="Primary key"),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim, description="Vector field for similarity search"),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100, description="Product name"),
    FieldSchema(name="price", dtype=DataType.FLOAT, description="Product price"),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50, description="Product category")
]

schema = CollectionSchema(fields=fields, description="Collection for partial update demo")

client.create_collection(collection_name, schema=schema, consistency_level="Strong")

index_params = client.prepare_index_params()
index_params.add_index(field_name = "vector", metric_type="L2")

client.create_index(collection_name, index_params)

client.load_collection(collection_name)

# Insert initial data
rng = np.random.default_rng(seed=19530)
initial_data = [
    {
        "id": 1, 
        "vector": rng.random(dim).tolist(),
        "name": "Product A",
        "price": 100.0,
        "category": "Electronics"
    },
    {
        "id": 2,
        "vector": rng.random(dim).tolist(), 
        "name": "Product B",
        "price": 200.0,
        "category": "Home"
    }
]

client.insert(collection_name, initial_data)
print("Initial data inserted")

# Query initial state
results = client.query(collection_name, filter="id > 0", output_fields=["*"])
print("\nInitial state:")
for r in results:
    print(f"  ID: {r['id']}, Name: {r['name']}, Price: ${r['price']}, Category: {r['category']}")

# 2. Partial Update - Update only price
print("\n2. Partial Update - Price only...")
price_update = [{"id": 1, "price": 80.0}]  # 20% discount

# Using partial_update=True preserves other fields
client.upsert(collection_name, price_update, partial_update=True)

results = client.query(collection_name, filter="id == 1", output_fields=["*"])
print(f"After partial price update:")
for r in results:
    print(f"  ID: {r['id']}, Name: {r['name']}, Price: ${r['price']}, Category: {r['category']}")
    print("  ✅ Only price changed, other fields preserved")

# 3. Comparison: Full update (partial_update=False - requires all fields)
print("\n3. Full update (partial_update=False) - Must provide ALL fields")

# Save current state
current_state = client.query(collection_name, filter="id == 2", output_fields=["*"])[0]
print(f"Current state of ID 2:")
print(f"  Name: {current_state['name']}, Price: ${current_state['price']}, Category: {current_state['category']}")

# Full update requires ALL fields including vector
print("\n3a. Full update with all required fields:")
full_update_complete = [{
    "id": 2,
    "vector": current_state['vector'],  # Must include vector
    "name": "Product B - Full Update",   # Can change this
    "price": 150.0,                     # Can change this
    "category": "Home & Garden"          # Can change this
}]

client.upsert(collection_name, full_update_complete)  # partial_update=False (default)

results = client.query(collection_name, filter="id == 2", output_fields=["*"])
print(f"After full update with all fields:")
for r in results:
    print(f"  ID: {r['id']}, Name: {r['name']}, Price: ${r['price']}, Category: {r['category']}")
    print("  ✅ All fields provided, update successful")

# Demonstrate what happens when required fields are missing
print("\n3b. What happens when fields are missing in full update:")
print("❌ This would cause an error or unexpected behavior:")
print("   full_update_incomplete = [{'id': 2, 'price': 175.0}]")
print("   client.upsert(collection_name, full_update_incomplete)  # Missing vector!")
print("   → May fail or cause data loss depending on collection schema")

# 4. Fix using partial update
print("\n4. Using partial update to modify specific fields...")
partial_fix = [{"id": 2, "name": "Product B - Corrected", "category": "Smart Home"}]
client.upsert(collection_name, partial_fix, partial_update=True)

results = client.query(collection_name, filter="id == 2", output_fields=["*"])
print("After partial update correction:")
for r in results:
    print(f"  ID: {r['id']}, Name: {r['name']}, Price: ${r['price']}, Category: {r['category']}")
    print("  ✅ Only specified fields updated, price and vector preserved")

# 5. Multiple field partial update
print("\n5. Multiple field partial update...")
multi_update = [
    {"id": 1, "name": "Product A - Updated", "category": "Premium Electronics"},
    {"id": 2, "name": "Product B - Updated", "category": "Smart Home"}
]

client.upsert(collection_name, multi_update, partial_update=True)

results = client.query(collection_name, filter="id > 0", output_fields=["*"])
print("After multi-field partial update:")
for r in results:
    print(f"  ID: {r['id']}, Name: {r['name']}, Price: ${r['price']}, Category: {r['category']}")

print("\n=== Summary ===")
print("✅ partial_update=True: Updates only specified fields, preserves others")
print("❌ partial_update=False (default): Requires ALL fields, replaces entire entity")
print("💡 Key differences:")
print("   • partial_update=True: Provide only fields you want to change")
print("   • partial_update=False: Must provide ALL required fields (including vectors)")
print("   • Use partial_update=True for metadata updates")
print("   • Use partial_update=False for complete entity replacement")

# Cleanup
client.drop_collection(collection_name)
print("\nCollection dropped. Demo complete!")
