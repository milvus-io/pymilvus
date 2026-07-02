"""Examples for the new search_aggregation API (Phase 1).

Requires Milvus server with search aggregation support. Run against a local
Milvus by default (http://localhost:19530).

Covers the five cases from the PyMilvus search aggregation design doc §5:
    1. Flat — single level, single field
    2. Flat — composite key + metrics + ordering
    3. JSON field grouping (disabled — server does not yet support JSON paths)
    4. Two-level nested grouping
    5. Three-level nested grouping

Usage:
    python examples/search_aggregation.py                # run all enabled cases
    python examples/search_aggregation.py --cases 1      # just case 1
    python examples/search_aggregation.py --cases 1 2    # cases 1 + 2
    python examples/search_aggregation.py --no-rebuild   # reuse existing collection
"""

import argparse
from typing import List

import numpy as np

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    SearchAggregation,
    TopHits,
)

COLLECTION = "demo_search_aggregation"
DIM = 8
NUM_ENTITIES = 2000
rng = np.random.default_rng(seed=19530)

CATEGORIES = ["electronics", "books", "clothing", "home", "toys"]
BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "BrandF", "BrandG", "BrandH"]
COLORS = ["red", "blue", "green", "black"]
SKUS = [f"sku_{i:03d}" for i in range(20)]


def build_collection(client: MilvusClient, rebuild: bool = True) -> None:
    if client.has_collection(COLLECTION) and not rebuild:
        client.load_collection(COLLECTION)
        return
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    schema = CollectionSchema(
        [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema("category", DataType.VARCHAR, max_length=32),
            FieldSchema("brand", DataType.VARCHAR, max_length=32),
            FieldSchema("color", DataType.VARCHAR, max_length=16),
            FieldSchema("sku", DataType.VARCHAR, max_length=16),
            FieldSchema("price", DataType.DOUBLE),
            FieldSchema("rating", DataType.DOUBLE),
            FieldSchema("in_stock", DataType.BOOL),
            FieldSchema("meta", DataType.JSON),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
        ]
    )
    client.create_collection(COLLECTION, dimension=DIM, schema=schema)

    rows = []
    for i in range(NUM_ENTITIES):
        rows.append(
            {
                "id": i,
                # Decouple group-by dims: color shifts by brand period so
                # (brand, color) enumerates 4*4=16 combos; sku shifts by
                # (brand*color) period so (brand, color, sku) spans the full
                # cartesian space within NUM_ENTITIES.
                "category": CATEGORIES[i % len(CATEGORIES)],
                "brand": BRANDS[i % len(BRANDS)],
                "color": COLORS[(i // len(BRANDS)) % len(COLORS)],
                "sku": SKUS[(i // (len(BRANDS) * len(COLORS))) % len(SKUS)],
                "price": float(10 + (i % 500)),
                "rating": float((i % 50) / 10.0),
                "in_stock": (i % 3) != 0,
                "meta": {
                    "subcategory": f"sub_{i % 8}",
                    "region": ["us", "eu", "apac"][i % 3],
                },
                "embedding": rng.random(DIM).astype(np.float32).tolist(),
            }
        )
    client.insert(COLLECTION, rows)
    client.flush(COLLECTION)

    idx = client.prepare_index_params()
    idx.add_index(field_name="embedding", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 128})
    client.create_index(COLLECTION, idx)
    client.load_collection(COLLECTION)


INDENT_STEP = 4          # spaces per nesting level
BUCKET_MARKER = "●"      # top-level bucket at depth 0
SUB_MARKER = "└─"        # nested bucket marker

NQ = 3                   # number of query vectors per case


def _print_buckets_recursive(buckets, depth: int) -> None:
    bucket_pad = " " * (depth * INDENT_STEP)
    label_pad = " " * (depth * INDENT_STEP + INDENT_STEP // 2)
    hit_pad = " " * (depth * INDENT_STEP + INDENT_STEP)
    marker = BUCKET_MARKER if depth == 0 else SUB_MARKER
    for b in buckets:
        key_str = ", ".join(
            f"{k['field_name'] or k['field_id']}={k['value']!r}" for k in b.key
        )
        metrics_str = ", ".join(f"{k}={v}" for k, v in b.metrics.items())
        head = f"{bucket_pad}{marker} [L{depth}] key[{key_str}] count={b.count}"
        if metrics_str:
            head += f"  metrics={{{metrics_str}}}"
        print(head)
        if b.hits:
            print(f"{label_pad}top_hits:")
            for h in b.hits:
                fields_str = ", ".join(f"{name}={v!r}" for name, v in h.fields.items())
                print(f"{hit_pad}· pk={h.pk} score={h.score:.4f}  fields={{{fields_str}}}")
        if b.sub_groups:
            print(f"{label_pad}sub_groups:")
            _print_buckets_recursive(b.sub_groups, depth + 1)


def print_buckets(label: str, per_nq_buckets) -> None:
    """Print search_aggregation results. per_nq_buckets is List[List[AggregationBucket]]."""
    print(f"\n=== {label} ===")
    for i, buckets in enumerate(per_nq_buckets):
        print(f"--- nq[{i}] ({len(buckets)} buckets) ---")
        _print_buckets_recursive(buckets, depth=0)


def query_vectors(nq: int = NQ) -> List[List[float]]:
    return [rng.random(DIM).astype(np.float32).tolist() for _ in range(nq)]


def case1_single_field(client: MilvusClient) -> None:
    """Single level, single field — simplest form."""
    res = client.search(
        collection_name=COLLECTION,
        data=query_vectors(),
        anns_field="embedding",
        output_fields=["id", "brand", "price"],
        search_aggregation=SearchAggregation(
            fields=["brand"],
            size=4,
            top_hits=TopHits(size=3),
        ),
    )
    print_buckets("Case 1: single field grouping", res.agg_buckets)


def case2_composite_key_with_metrics(client: MilvusClient) -> None:
    """Composite key (brand, color) + metrics + custom ordering + sorted top_hits."""
    res = client.search(
        collection_name=COLLECTION,
        data=query_vectors(),
        anns_field="embedding",
        output_fields=["id", "brand", "color", "price", "rating"],
        search_aggregation=SearchAggregation(
            fields=["brand", "color"],
            size=5,
            metrics={
                "avg_price": {"avg": "price"},
                "doc_count": {"count": "*"},
            },
            order=[{"avg_price": "desc"}],
            top_hits=TopHits(size=2, sort=[{"rating": "desc"}]),
        ),
    )
    print_buckets("Case 2: composite key + metrics + ordering", res.agg_buckets)


def case3_json_field(client: MilvusClient) -> None:
    """Group by JSON path expression.

    Currently disabled — SDK rejects JSON path fields until server support lands.
    Kept for future enablement; main() skips this case.
    """
    res = client.search(
        collection_name=COLLECTION,
        data=query_vectors(),
        anns_field="embedding",
        output_fields=["id", "category"],
        search_aggregation=SearchAggregation(
            fields=["meta['region']"],
            size=4,
            metrics={"avg_score": {"avg": "_score"}},
            order=[{"avg_score": "desc"}],
            top_hits=TopHits(size=2),
        ),
    )
    print_buckets("Case 3: JSON path grouping", res.agg_buckets)


def case4_two_level_nested(client: MilvusClient) -> None:
    """Two-level: category → brand, with metrics + top_hits at both levels."""
    res = client.search(
        collection_name=COLLECTION,
        data=query_vectors(),
        anns_field="embedding",
        output_fields=["id", "category", "brand", "price", "rating"],
        filter="in_stock == true",
        search_aggregation=SearchAggregation(
            fields=["category"],
            size=3,
            metrics={
                "total_revenue": {"sum": "price"},
                "item_count": {"count": "*"},
            },
            order=[{"total_revenue": "desc"}],
            top_hits=TopHits(size=2, sort=[{"_score": "desc"}]),
            sub_aggregation=SearchAggregation(
                fields=["brand"],
                size=2,
                metrics={"avg_rating": {"avg": "rating"}},
                order=[{"avg_rating": "desc"}],
                top_hits=TopHits(size=2, sort=[{"price": "asc"}]),
            ),
        ),
    )
    print_buckets("Case 4: two-level nested", res.agg_buckets)


def case5_three_level_nested(client: MilvusClient) -> None:
    """Three levels: category → brand → (sku, color), with top_hits at every level."""
    res = client.search(
        collection_name=COLLECTION,
        data=query_vectors(),
        anns_field="embedding",
        output_fields=["id", "category", "brand", "sku", "color", "price", "rating"],
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 16},
        },
        search_aggregation=SearchAggregation(
            fields=["category"],
            size=3,
            metrics={"total_revenue": {"sum": "price"}, "item_count": {"count": "*"}},
            order=[{"total_revenue": "desc"}],
            top_hits=TopHits(size=2, sort=[{"_score": "asc"}]),  # best matches in category
            sub_aggregation=SearchAggregation(
                fields=["brand"],
                size=3,
                metrics={"brand_revenue": {"sum": "price"}, "avg_rating": {"avg": "rating"}},
                order=[{"brand_revenue": "desc"}],
                top_hits=TopHits(size=2, sort=[{"rating": "desc"}]),  # best-rated in brand
                sub_aggregation=SearchAggregation(
                    fields=["sku", "color"],
                    size=3,
                    metrics={
                        "min_price": {"min": "price"},
                        "item_count": {"count": "*"},
                    },
                    order=[{"min_price": "asc"}],
                    top_hits=TopHits(size=2, sort=[{"price": "asc"}]),  # cheapest sku/color pair
                ),
            ),
        ),
    )
    print_buckets("Case 5: three-level nested", res.agg_buckets)


SEPARATOR = "=" * 80

CASES = {
    1: case1_single_field,
    2: case2_composite_key_with_metrics,
    3: case3_json_field,  # disabled-by-default (JSON path server support pending)
    4: case4_two_level_nested,
    5: case5_three_level_nested,
}
DEFAULT_CASES = [1, 2, 4, 5]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run search_aggregation example cases.")
    p.add_argument(
        "--cases",
        nargs="+",
        type=int,
        choices=sorted(CASES),
        metavar="N",
        help=f"case numbers to run; default {DEFAULT_CASES}. Case 3 is disabled by default.",
    )
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="reuse existing collection instead of dropping and re-inserting.",
    )
    p.add_argument("--uri", default="http://localhost:19530", help="Milvus URI")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.cases if args.cases else DEFAULT_CASES

    client = MilvusClient(args.uri)
    build_collection(client, rebuild=not args.no_rebuild)

    failed = []
    for n in selected:
        try:
            CASES[n](client)
        except MilvusException as e:
            # Keep going so later cases still run. Case 3 is expected to fail
            # today (SDK bans JSON path fields); other failures bubble up in the
            # summary so they stay visible.
            print(f"\n!!! case {n} failed: {type(e).__name__}: {e}")
            failed.append((n, e))
        print(SEPARATOR)

    if failed:
        print(f"\n{len(failed)} case(s) failed:")
        for n, e in failed:
            print(f"  - case {n}: {type(e).__name__}: {e}")

    client.release_collection(COLLECTION)


if __name__ == "__main__":
    main()
