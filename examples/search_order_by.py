from pymilvus.milvus_client.milvus_client import MilvusClient
from pymilvus import DataType
import numpy as np
from typing import List

collection_name = "test_search_order_by"
prepare_new_data = True
clean_exist = False

USER_ID = "id"
PRICE = "price"
RATING = "rating"
CATEGORY = "category"
METADATA = "metadata"  # JSON field
EMBEDDINGS = "embeddings"
DIM = 8
NUM_ENTITIES = 30000
rng = np.random.default_rng(seed=19530)
milvus_client = MilvusClient("http://localhost:19530")

if milvus_client.has_collection(collection_name) and clean_exist:
    milvus_client.drop_collection(collection_name)
    print(f"dropped existed collection {collection_name}")

if not milvus_client.has_collection(collection_name):
    # Create schema with JSON field and enable dynamic fields
    schema = MilvusClient.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name=USER_ID, datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name=PRICE, datatype=DataType.DOUBLE)
    schema.add_field(field_name=RATING, datatype=DataType.DOUBLE)
    schema.add_field(field_name=CATEGORY, datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name=METADATA, datatype=DataType.JSON)  # JSON field for nested data
    schema.add_field(field_name=EMBEDDINGS, datatype=DataType.FLOAT_VECTOR, dim=DIM)
    milvus_client.create_collection(collection_name, dimension=DIM, schema=schema)
    print(f"created collection {collection_name} with JSON field and dynamic fields enabled")

if prepare_new_data:
    categories = [f"cate{i}" for i in range(1, 21)]  # cate1, cate2, ..., cate20
    # Generate price with repetition to support multi-field sorting
    # Use integer prices (10.0, 11.0, etc.) to increase repetition probability
    # This allows testing order_by_fields with price ASC, rating DESC
    price_range = 50  # Price range: 10.0 to 59.0 (50 different prices)
    num_batches = 3  # Insert in 3 batches to create multiple segments
    batch_size = NUM_ENTITIES // num_batches  # 10000 entities per batch
    
    for batch_idx in range(num_batches):
        entities = []
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, NUM_ENTITIES)
        
        for i in range(start_idx, end_idx):
            # Generate integer price (10.0, 11.0, 12.0, ...) to increase repetition
            # Each price value will have ~600 entities (30000 / 50), ensuring repetition
            price = float(10 + (i % price_range))  # Price: 10.0, 11.0, ..., 59.0
            
            # Generate metadata JSON with nested fields for testing order_by_fields
            # Age: 18-80, Score: 0-100, Popularity: 0.0-10.0
            age = 18 + (i % 63)  # Age range: 18-80
            score = i % 101  # Score range: 0-100
            popularity = round(rng.random() * 10, 1)  # Popularity: 0.0-10.0
            
            entity = {
                USER_ID: i,
                PRICE: price,
                # Rating varies independently, so entities with same price can have different ratings
                # Rating precision: 1 decimal place (0.0, 0.1, 0.2, ..., 4.9)
                RATING: round(rng.random() * 5, 1),
                CATEGORY: categories[i % len(categories)],
                # JSON field with nested data for testing order_by_fields on JSON fields
                METADATA: {
                    "age": age,
                    "score": score,
                    "popularity": popularity,
                    "tags": [categories[i % len(categories)], f"tag_{i % 10}"]
                },
                EMBEDDINGS: rng.random((1, DIM))[0].tolist(),
                # Dynamic field example - will be stored in the dynamic field
                "dynamic_views": i * 10 + rng.integers(0, 100)  # Views: varies per entity
            }
            entities.append(entity)
        
        # Insert batch and flush to create separate segments
        milvus_client.insert(collection_name, entities)
        milvus_client.flush(collection_name)
        print(f"Inserted and flushed batch {batch_idx + 1}/{num_batches} ({len(entities)} entities)")
    
    print(f"Finish flush collection: {collection_name} with {num_batches} segments")

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name=EMBEDDINGS,
        index_type='IVF_FLAT',
        metric_type='L2',
        params={"nlist": 1024}
    )
    milvus_client.create_index(collection_name, index_params)
milvus_client.load_collection(collection_name)

nq = 1
limit = 20
def print_res(result: List[List[dict]], title: str):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for i in range(nq):
        r = result[i]
        print(f"search result {i}:")
        for e in r:
            print(f"  {e}")
    print()


vector_to_search = rng.random((nq, DIM), np.float32)

# Basic search with order_by_fields - sort by price ascending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY],
    order_by_fields=[
        {"field": PRICE, "order": "asc"}
    ]
)
print_res(res, "Search with order_by_fields price ASC")

# Search with order_by_fields - sort by rating descending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY],
    order_by_fields=[
        {"field": RATING, "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields rating DESC")

# Search with multiple order_by_fields - sort by price asc, then rating desc
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY],
    order_by_fields=[
        {"field": PRICE, "order": "asc"},
        {"field": RATING, "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields price ASC, rating DESC")

# Search with order_by_fields on JSON field - sort by metadata["age"] ascending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, METADATA],
    order_by_fields=[
        {"field": 'metadata["age"]', "order": "asc"}
    ]
)
print_res(res, "Search with order_by_fields on JSON field: metadata[\"age\"] ASC")

# Search with order_by_fields on JSON field - sort by metadata["score"] descending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, METADATA],
    order_by_fields=[
        {"field": 'metadata["score"]', "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields on JSON field: metadata[\"score\"] DESC")

# Search with order_by_fields on JSON field - sort by metadata["popularity"] descending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, METADATA],
    order_by_fields=[
        {"field": 'metadata["popularity"]', "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields on JSON field: metadata[\"popularity\"] DESC")

# Search with multiple order_by_fields including JSON field - sort by metadata["age"] asc, then metadata["score"] desc
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, METADATA],
    order_by_fields=[
        {"field": 'metadata["age"]', "order": "asc"},
        {"field": 'metadata["score"]', "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields: metadata[\"age\"] ASC, metadata[\"score\"] DESC")

# Search with order_by_fields on dynamic field - sort by dynamic_views descending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, "dynamic_views"],
    order_by_fields=[
        {"field": "dynamic_views", "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields on dynamic field: dynamic_views DESC")

# Mixed order_by_fields - regular field and JSON field
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, METADATA],
    order_by_fields=[
        {"field": PRICE, "order": "asc"},
        {"field": 'metadata["age"]', "order": "desc"}
    ]
)
print_res(res, "Search with order_by_fields: price ASC, metadata[\"age\"] DESC")

# Group By + Order By - group by category and sort by price ascending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY],
    group_by_field=CATEGORY,
    group_size=3,
    strict_group_size=True,
    order_by_fields=[
        {"field": PRICE, "order": "asc"}
    ]
)
print_res(res, "Group By category + Order By price ASC")

# Group By + Order By with JSON field - group by category and sort by metadata["age"] ascending
res = milvus_client.search(
    collection_name=collection_name,
    data=vector_to_search,
    limit=limit,
    anns_field=EMBEDDINGS,
    output_fields=[USER_ID, PRICE, RATING, CATEGORY, METADATA],
    group_by_field=CATEGORY,
    group_size=3,
    strict_group_size=True,
    order_by_fields=[
        {"field": 'metadata["age"]', "order": "asc"}
    ]
)
print_res(res, "Group By category + Order By metadata[\"age\"] ASC")

print("All search_order_by tests completed!")
