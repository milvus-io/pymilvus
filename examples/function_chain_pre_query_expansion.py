"""Run PRE query expansion and original-query aggregation on a local server.

Requires the matching experimental server/SDK, pymilvus[arrow] and the current
transform_demo wheel registered under --resource-name. This example creates
three deterministic rows and uses no external model API.
"""

# ruff: noqa: INP001, S101, T201

import argparse

from pymilvus import DataType, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.function_chain import agg, col, fn


def expansion_chains(resource_name: str):
    pre = (
        FunctionChain(FunctionChainStage.PRE_PROCESS)
        .flat_map(
            "query",
            fn.py_udf(
                col("query"),
                resource_name=resource_name,
                udf_params={"mode": "expand_queries", "separator": "|"},
            ),
        )
        .map(
            "query",
            fn.py_udf(
                col("query"),
                resource_name=resource_name,
                udf_params={"mode": "embedding"},
            ),
        )
        .select(["query"])
    )
    post = (
        FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table")
        .union_all()
        .group_by("$query_index")
        .aggregate(
            {
                "matches": agg.count(),
                "ids": agg.collect_list(col("$id")),
                "subqueries": agg.collect_list(col("$subquery_index")),
            }
        )
    )
    return pre, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--collection", default="function_chain_pre_query_expansion_demo")
    parser.add_argument(
        "--resource-name", required=True, help="Registered current transform_demo wheel"
    )
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse this example's existing collection"
    )
    args = parser.parse_args()
    client = MilvusClient(args.uri, timeout=60)
    try:
        if not args.reuse:
            schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("text", DataType.VARCHAR, max_length=256)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
            client.create_collection(args.collection, schema=schema)
            client.insert(
                args.collection,
                [
                    {"text": text, "vector": [float(len(text)), 1.0]}
                    for text in ("alpha", "beta", "gamma")
                ],
            )
            indexes = client.prepare_index_params()
            indexes.add_index("vector", index_type="FLAT", metric_type="COSINE")
            client.create_index(args.collection, indexes)
        client.load_collection(args.collection)
        pre, post = expansion_chains(args.resource_name)
        result = client.search(
            args.collection,
            data=["alpha|beta", "gamma"],
            anns_field="vector",
            limit=2,
            search_params={"metric_type": "COSINE"},
            function_chains=[pre, post],
        )
        rows = sorted(result.rows, key=lambda row: row["$query_index"])
        assert [row["matches"] for row in rows] == [4, 2]
        assert set(rows[0]["subqueries"]) == {0, 1}
        assert set(rows[1]["subqueries"]) == {2}
        print("ORIGINAL_QUERY_GROUPS", result.schema, rows)
    finally:
        client.close()


if __name__ == "__main__":
    main()
