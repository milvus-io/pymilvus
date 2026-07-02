import time

import numpy as np

from pymilvus import DataType, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.function_chain import col, fn

fmt = "\n=== {:40} ===\n"

COLLECTION_NAME = "function_chain_example"
DIM = 8
EMBEDDING = "embedding"
PUBLISHED_AT = "published_at"
POPULARITY = "popularity"
TITLE = "title"

client = MilvusClient("http://localhost:19530")

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

schema = client.create_schema(enable_dynamic_field=False, auto_id=True)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field(EMBEDDING, DataType.FLOAT_VECTOR, dim=DIM)
schema.add_field(PUBLISHED_AT, DataType.INT64)
schema.add_field(POPULARITY, DataType.DOUBLE)
schema.add_field(TITLE, DataType.VARCHAR, max_length=128)

index_params = client.prepare_index_params()
index_params.add_index(field_name=EMBEDDING, index_type="FLAT", metric_type="IP")

client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
    consistency_level="Strong",
)

rng = np.random.default_rng(seed=19530)
now = int(time.time())
rows = [
    {
        EMBEDDING: rng.random(DIM).astype(np.float32).tolist(),
        PUBLISHED_AT: now - 60,
        POPULARITY: 0.10,
        TITLE: "fresh but not popular",
    },
    {
        EMBEDDING: rng.random(DIM).astype(np.float32).tolist(),
        PUBLISHED_AT: now - 3600,
        POPULARITY: 0.95,
        TITLE: "popular within the last hour",
    },
    {
        EMBEDDING: rng.random(DIM).astype(np.float32).tolist(),
        PUBLISHED_AT: now - 86400,
        POPULARITY: 0.70,
        TITLE: "popular yesterday",
    },
    {
        EMBEDDING: rng.random(DIM).astype(np.float32).tolist(),
        PUBLISHED_AT: now - 86400 * 7,
        POPULARITY: 0.30,
        TITLE: "older and less popular",
    },
]

print(fmt.format("Insert data"))
print(client.insert(COLLECTION_NAME, rows))

client.load_collection(COLLECTION_NAME)

query_vector = rng.random(DIM).astype(np.float32).tolist()

print(fmt.format("Baseline vector search"))
baseline = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field=EMBEDDING,
    search_params={"metric_type": "IP"},
    limit=4,
    output_fields=[TITLE, PUBLISHED_AT, POPULARITY],
)
for hit in baseline[0]:
    print(hit)

# Function Chain is an ordered rerank plan for search results.
# This chain does four things:
# 1. Compute a freshness score from the published_at timestamp.
# 2. Combine the original vector score ($score), freshness, and popularity.
# 3. Round the final score to 4 decimal places.
# 4. Sort by the rewritten $score and keep only the top 3 results.
rerank_chain = (
    FunctionChain(FunctionChainStage.L2_RERANK, name="fresh_popular_rerank")
    .map(
        "freshness",
        fn.decay(
            col(PUBLISHED_AT),
            function="exp",
            origin=now,
            scale=86400,
            offset=0,
            decay=0.5,
        ),
    )
    .map(
        "$score",
        fn.num_combine(
            col("$score"),
            col("freshness"),
            col(POPULARITY),
            mode="weighted",
            weights=[0.7, 0.2, 0.1],
        ),
    )
    .map("$score", fn.round_decimal(col("$score"), decimal=4))
    .sort(col("$score"), desc=True, tie_break_col=col("$id"))
    .limit(3)
)

print(fmt.format("Search with FunctionChain rerank"))
reranked = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field=EMBEDDING,
    search_params={"metric_type": "IP"},
    limit=4,
    output_fields=[TITLE, PUBLISHED_AT, POPULARITY],
    function_chains=rerank_chain,
)
for hit in reranked[0]:
    print(hit)

client.drop_collection(COLLECTION_NAME)
