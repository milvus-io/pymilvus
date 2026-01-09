import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection)
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
connections.connect("default", host="localhost", port="19530")

if clean_exist and utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

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
collection = Collection(collection_name, schema, consistency_level="Strong")

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

        # collect minimal stats
        pair_counter.update(zip(c1_data, c6_data))
        c2_sum += sum(c2_data)
        c3_sum += sum(c3_data)
        c4_sum += sum(c4_data)
        c2_lt_2 += sum(1 for v in c2_data if v < 2)
        c2_dist.update(c2_data)

        entities = [
            [str(i) for i in range(num_entities * batch_idx, num_entities * (batch_idx + 1))],
            c1_data,
            c2_data,
            c3_data,
            c4_data,
            ts_data,
            rng.random((num_entities, dim)),
            c6_data,
        ]
        collection.insert(entities)
        if to_flush:
            print(f"flush batch:{batch_idx}")
            collection.flush()
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
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("c5", index)
print(f"Number of entities in Milvus: {collection.num_entities}")  # check the num_entities
collection.load()


#1. group by TIMESTAMPTZ + max
print(fmt.format("Query: group by TIMESTAMPTZ + max"))
res_ts = collection.query(
    expr="c2 < 10",
    group_by_fields=[TS],
    output_fields=[TS, "count(c2)", f"max({TS})"],
    timeout=120.0,
)
for row in res_ts:
    print(f"res={row}")

#2. group by (c1,c6) + min/max
print(fmt.format("Query: group by (c1,c6) + min/max"))
res_minmax = collection.query(
    expr="c2 < 10",
    group_by_fields=["c1", "c6"],
    output_fields=["c1", "c6", "min(c2)", "max(c2)"],
    timeout=120.0,
)
for row in res_minmax:
    print(f"res={row}")


#3. group by c1 + avg(c2, c3, c4)
print(fmt.format("Query: group by c1 + avg(c2, c3, c4)"))
res_avg = collection.query(
    expr="c2 < 10",
    group_by_fields=["c1"],
    output_fields=["c1", "avg(c2)", "avg(c3)", "avg(c4)"],
    timeout=120.0,
)
for row in res_avg:
    print(f"res={row}")

#4. group by c1 + avg(c2, c3, c4) without expr
print(fmt.format("Query: group by c1 + avg(c2, c3, c4)"))
res_avg = collection.query(
    expr="",
    group_by_fields=["c1"],
    output_fields=["c1", "avg(c2)", "avg(c3)", "avg(c4)"],
    timeout=120.0,
    limit=10,
)
for row in res_avg:
    print(f"res={row}")    