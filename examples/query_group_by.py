import numpy as np
from pymilvus import (
    FieldSchema, CollectionSchema, DataType,
)
from pymilvus.milvus_client import MilvusClient
from collections import Counter
from datetime import datetime, timezone
import random

names = ["Green", "Rachel", "Joe", "Chandler", "Phebe", "Ross", "Monica"]
collection_name = 'test_query_group_by'
clean_exist = False 
prepare_data = False
to_flush = True
batch_num = 3
num_entities, dim = 122, 8
fmt = "\n=== {:30} ===\n"
SHOW_STATS_DETAILS = False

print(fmt.format("start connecting to Milvus"))
client = MilvusClient(uri="http://localhost:19530")

if clean_exist and client.has_collection(collection_name):
    client.drop_collection(collection_name)

TS = "ts"
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="c1", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="c2", dtype=DataType.INT16),
    FieldSchema(name="c3", dtype=DataType.INT32),
    FieldSchema(name="c4", dtype=DataType.DOUBLE),
    FieldSchema(name=TS, dtype=DataType.TIMESTAMPTZ, description="timestamp with timezone"),
    FieldSchema(name="c5", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="c6", dtype=DataType.VARCHAR, max_length=512),
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
    print(fmt.format("Start inserting entities"))

    # Keep a small cardinality for TS so GROUP BY results are readable.
    ts_choices = [
        datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).isoformat(),
        datetime(2025, 1, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat(),
        datetime(2025, 1, 1, 2, 0, 0, tzinfo=timezone.utc).isoformat(),
    ]

    # Minimal stats (avoid printing too much)
    pair_counter = Counter()
    c2_sum = 0
    c3_sum = 0
    c4_sum = 0.0
    c2_lt_2 = 0
    c2_dist = Counter()

    for batch_idx in range(batch_num):
        # generate per-batch data
        c1_data = [random.choice(names) for _ in range(num_entities)]
        c2_data = [random.randint(0, 4) for _ in range(num_entities)]
        c3_data = [random.randint(0, 6) for _ in range(num_entities)]
        c4_data = [random.uniform(0.0, 100.0) for _ in range(num_entities)]
        c6_data = [random.choice(names) for _ in range(num_entities)]
        ts_data = [ts_choices[(batch_idx + j) % len(ts_choices)] for j in range(num_entities)]
        vector_data = rng.random((num_entities, dim))

        # collect minimal stats
        pair_counter.update(zip(c1_data, c6_data))
        c2_sum += sum(c2_data)
        c3_sum += sum(c3_data)
        c4_sum += sum(c4_data)
        c2_lt_2 += sum(1 for v in c2_data if v < 2)
        c2_dist.update(c2_data)

        # Convert to dict format for milvus_client.insert()
        data = [
            {
                "pk": str(i),
                "c1": c1_data[i],
                "c2": c2_data[i],
                "c3": c3_data[i],
                "c4": c4_data[i],
                TS: ts_data[i],
                "c5": vector_data[i].tolist(),
                "c6": c6_data[i],
            }
            for i in range(num_entities)
        ]
        
        client.insert(collection_name, data)
        if to_flush:
            print(f"flush batch:{batch_idx}")
            client.flush(collection_name)
        print(f"inserted batch:{batch_idx}")

    total_rows = batch_num * num_entities
    print(fmt.format("Quick stats (compact)"))
    print(f"total_rows: {total_rows}")
    print(f"unique (c1,c6) pairs: {len(pair_counter)}")
    print("top 5 (c1,c6) pairs:")
    for (c1v, c6v), cnt in pair_counter.most_common(5):
        print(f"  ({c1v}, {c6v}): {cnt}")
    print(f"sum(c2): {c2_sum}, sum(c3): {c3_sum}, sum(c4): {c4_sum:.2f}")
    print(f"c2 < 2: {c2_lt_2} ({c2_lt_2/total_rows*100:.2f}%)")
    if SHOW_STATS_DETAILS:
        print(f"c2 distribution: {dict(sorted(c2_dist.items()))}")

        
    print(fmt.format("Start Creating index IVF_FLAT"))
    from pymilvus.milvus_client.index import IndexParams
    index_params = IndexParams()
    index_params.add_index("c5", index_type="IVF_FLAT", metric_type="L2", nlist=128)
    client.create_index(collection_name, index_params)

stats = client.get_collection_stats(collection_name)
print(f"Number of entities in Milvus: {stats.get('row_count', 0)}")  # check the num_entities
client.load_collection(collection_name)


#1. group by TIMESTAMPTZ + max
print(fmt.format("Query: group by TIMESTAMPTZ + max"))
res_ts = client.query(
    collection_name=collection_name,
    filter="c2 < 10",
    output_fields=[TS, "count(c2)", f"max({TS})"],
    timeout=120.0,
    group_by_fields=[TS],
)
for row in res_ts:
    print(f"res={row}")

#2. group by (c1,c6) + min/max
print(fmt.format("Query: group by (c1,c6) + min/max"))
res_minmax = client.query(
    collection_name=collection_name,
    filter="c2 < 10",
    output_fields=["c1", "c6", "min(c2)", "max(c2)"],
    timeout=120.0,
    group_by_fields=["c1", "c6"],
)
for row in res_minmax:
    print(f"res={row}")


#3. group by c1 + avg(c2, c3, c4)
print(fmt.format("Query: group by c1 + avg(c2, c3, c4)"))
res_avg = client.query(
    collection_name=collection_name,
    filter="c2 < 10",
    output_fields=["c1", "avg(c2)", "avg(c3)", "avg(c4)"],
    timeout=120.0,
    group_by_fields=["c1"],
)
for row in res_avg:
    print(f"res={row}")

#4. group by c1 + avg(c2, c3, c4) without expr
print(fmt.format("Query: group by c1 + avg(c2, c3, c4)"))
res_avg = client.query(
    collection_name=collection_name,
    filter="",
    output_fields=["c1", "avg(c2)", "avg(c3)", "avg(c4)"],
    timeout=120.0,
    limit=10,
    group_by_fields=["c1"],
)
for row in res_avg:
    print(f"res={row}")
