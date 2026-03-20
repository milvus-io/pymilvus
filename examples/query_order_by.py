import numpy as np
from pymilvus import (
    FieldSchema, CollectionSchema, DataType,
)
from pymilvus.milvus_client import MilvusClient
import random

names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]
categories = ["Electronics", "Books", "Clothing", "Food", "Sports"]
collection_name = 'test_query_order_by'
clean_exist = True
prepare_data = True
to_flush = True
batch_num = 3
num_entities, dim = 100, 8
fmt = "\n=== {:30} ===\n"

print(fmt.format("start connecting to Milvus"))
client = MilvusClient(uri="http://localhost:19530")

if clean_exist and client.has_collection(collection_name):
    client.drop_collection(collection_name)

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="price", dtype=DataType.DOUBLE),
    FieldSchema(name="rating", dtype=DataType.INT32),
    FieldSchema(name="stock", dtype=DataType.INT64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
]

schema = CollectionSchema(fields)

print(fmt.format(f"Create collection `{collection_name}`"))
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    consistency_level="Strong"
)

if prepare_data:
    rng = np.random.default_rng(seed=19530)
    random.seed(42)
    print(fmt.format("Start inserting entities"))

    for batch_idx in range(batch_num):
        data = [
            {
                "name": random.choice(names),
                "category": random.choice(categories),
                "price": round(random.uniform(10.0, 500.0), 2),
                "rating": random.randint(1, 5),
                "stock": random.randint(0, 1000),
                "embedding": rng.random(dim).tolist(),
            }
            for _ in range(num_entities)
        ]
        client.insert(collection_name, data)
        if to_flush:
            client.flush(collection_name)
        print(f"inserted batch:{batch_idx}")

    from pymilvus.milvus_client.index import IndexParams
    index_params = IndexParams()
    index_params.add_index("embedding", index_type="IVF_FLAT", metric_type="L2", nlist=128)
    client.create_index(collection_name, index_params)

stats = client.get_collection_stats(collection_name)
print(f"Number of entities in Milvus: {stats.get('row_count', 0)}")
client.load_collection(collection_name)

passed = 0
failed = 0

def verify_sorted(rows, field, ascending=True, label=""):
    """Verify rows are sorted by field."""
    global passed, failed
    values = [r[field] for r in rows]
    is_sorted = all(
        (values[i] <= values[i+1] if ascending else values[i] >= values[i+1])
        for i in range(len(values)-1)
    )
    if is_sorted:
        print(f"  [PASS] {label}: {len(rows)} rows correctly sorted by {field} {'ASC' if ascending else 'DESC'}")
        passed += 1
    else:
        print(f"  [FAIL] {label}: NOT sorted! values={values[:5]}...")
        failed += 1
    return is_sorted


# ==============================================================================
# Test Cases
# ==============================================================================

# 1. ORDER BY single field ASC
print(fmt.format("Test 1: ORDER BY price ASC"))
res = client.query(
    collection_name=collection_name,
    filter="price > 50",
    output_fields=["name", "price"],
    limit=10,
    order_by=["price:asc"],
)
for row in res:
    print(f"  name={row['name']}, price={row['price']}")
verify_sorted(res, "price", ascending=True, label="Test 1")

# 2. ORDER BY single field DESC
print(fmt.format("Test 2: ORDER BY price DESC"))
res = client.query(
    collection_name=collection_name,
    filter="price > 50",
    output_fields=["name", "price"],
    limit=10,
    order_by=["price:desc"],
)
for row in res:
    print(f"  name={row['name']}, price={row['price']}")
verify_sorted(res, "price", ascending=False, label="Test 2")

# 3. ORDER BY integer field ASC
print(fmt.format("Test 3: ORDER BY rating ASC"))
res = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "rating"],
    limit=15,
    order_by=["rating:asc"],
)
for row in res:
    print(f"  name={row['name']}, rating={row['rating']}")
verify_sorted(res, "rating", ascending=True, label="Test 3")

# 4. ORDER BY string field ASC
print(fmt.format("Test 4: ORDER BY category ASC"))
res = client.query(
    collection_name=collection_name,
    filter="rating >= 3",
    output_fields=["name", "category", "rating"],
    limit=10,
    order_by=["category:asc"],
)
for row in res:
    print(f"  name={row['name']}, category={row['category']}")
verify_sorted(res, "category", ascending=True, label="Test 4")

# 5. ORDER BY stock DESC (INT64)
print(fmt.format("Test 5: ORDER BY stock DESC"))
res = client.query(
    collection_name=collection_name,
    filter="category == 'Electronics'",
    output_fields=["name", "stock", "price"],
    limit=10,
    order_by=["stock:desc"],
)
for row in res:
    print(f"  name={row['name']}, stock={row['stock']}, price={row['price']}")
verify_sorted(res, "stock", ascending=False, label="Test 5")

# 6. ORDER BY with complex filter
print(fmt.format("Test 6: Complex filter + ORDER BY"))
res = client.query(
    collection_name=collection_name,
    filter="price > 100 and rating >= 3 and stock > 50",
    output_fields=["name", "price", "rating", "stock"],
    limit=10,
    order_by=["price:desc"],
)
for row in res:
    print(f"  name={row['name']}, price={row['price']}, rating={row['rating']}, stock={row['stock']}")
verify_sorted(res, "price", ascending=False, label="Test 6")

# 7. ORDER BY with offset (pagination)
print(fmt.format("Test 7: ORDER BY with offset"))
page1 = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "price"],
    limit=5,
    offset=0,
    order_by=["price:asc"],
)
page2 = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "price"],
    limit=5,
    offset=5,
    order_by=["price:asc"],
)
print("Page 1:")
for row in page1:
    print(f"  name={row['name']}, price={row['price']}")
print("Page 2:")
for row in page2:
    print(f"  name={row['name']}, price={row['price']}")
verify_sorted(page1, "price", ascending=True, label="Test 7 page1")
verify_sorted(page2, "price", ascending=True, label="Test 7 page2")
# Verify page2 starts after page1 ends
if len(page1) > 0 and len(page2) > 0:
    if page1[-1]["price"] <= page2[0]["price"]:
        print(f"  [PASS] Test 7 pagination: page1 last ({page1[-1]['price']}) <= page2 first ({page2[0]['price']})")
        passed += 1
    else:
        print(f"  [FAIL] Test 7 pagination: page1 last ({page1[-1]['price']}) > page2 first ({page2[0]['price']})")
        failed += 1

# 8. Multi-field ORDER BY: rating DESC, then price ASC
print(fmt.format("Test 8: Multi-field ORDER BY"))
res = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "rating", "price"],
    limit=20,
    order_by=["rating:desc", "price:asc"],
)
for row in res:
    print(f"  name={row['name']}, rating={row['rating']}, price={row['price']}")
# Verify primary sort key
verify_sorted(res, "rating", ascending=False, label="Test 8 primary key")
# Verify secondary sort within same rating groups
rating_groups = {}
for row in res:
    rating_groups.setdefault(row['rating'], []).append(row['price'])
all_secondary_ok = True
for rating, prices in rating_groups.items():
    if not all(prices[i] <= prices[i+1] for i in range(len(prices)-1)):
        print(f"  [FAIL] Test 8 secondary: rating={rating} prices not ASC: {prices}")
        all_secondary_ok = False
if all_secondary_ok:
    print(f"  [PASS] Test 8 secondary key: prices ASC within each rating group")
    passed += 1
else:
    failed += 1

# 9. ORDER BY all records (empty filter)
print(fmt.format("Test 9: ORDER BY with empty filter"))
res = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "price"],
    limit=10,
    order_by=["price:asc"],
)
for row in res:
    print(f"  name={row['name']}, price={row['price']}")
verify_sorted(res, "price", ascending=True, label="Test 9")

# 10. ORDER BY with output_fields not including sort field
print(fmt.format("Test 10: ORDER BY field not in output"))
res = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["name", "category"],
    limit=10,
    order_by=["price:asc"],
)
for row in res:
    print(f"  name={row['name']}, category={row['category']}, has_price={'price' in row}")
# Can't verify sort order since price is not in output, just verify it doesn't crash
print(f"  [PASS] Test 10: query succeeded with {len(res)} rows (sort field not in output)")
passed += 1

# ==============================================================================
# Summary
# ==============================================================================
print(fmt.format("Summary"))
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
if failed > 0:
    print("  *** SOME TESTS FAILED ***")
    exit(1)
else:
    print("  All tests passed!")
