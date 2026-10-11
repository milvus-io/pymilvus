"""Run POST table aggregation and optional Python reduction on a local server.

Requires the matching experimental server/SDK and pymilvus[arrow]. The optional
UDF cases use Milvus's transform_demo 0.3 wheel, registered as a file resource.
This example creates three deterministic rows; it makes no external model calls.
"""

# This example prints results and verifies the table contract.
# ruff: noqa: INP001, S101, T201

import argparse

from pymilvus import AnnSearchRequest, DataType, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.function_chain import agg, col, fn


def table_chain(input_source: str = "fused") -> FunctionChain:
    return FunctionChain(FunctionChainStage.POST_PROCESS, result_mode="table", input=input_source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--collection", default="function_chain_post_table_demo")
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse this example's existing collection"
    )
    parser.add_argument("--resource-name", help="Registered transform_demo 0.3 wheel resource")
    parser.add_argument(
        "--wheel-path", help="Object-storage wheel key to register; not a local file"
    )
    args = parser.parse_args()
    if args.wheel_path and not args.resource_name:
        parser.error("--wheel-path requires --resource-name")
    client = MilvusClient(args.uri, timeout=60)
    try:
        if args.wheel_path:
            client.add_file_resource(args.resource_name, args.wheel_path)
        if not args.reuse:
            schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("tenant", DataType.VARCHAR, max_length=32)
            schema.add_field("text", DataType.VARCHAR, max_length=256)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
            client.create_collection(args.collection, schema=schema)
            client.insert(
                args.collection,
                [
                    {"tenant": "a", "text": "alpha", "vector": [1.0, 0.0]},
                    {"tenant": "a", "text": "beta", "vector": [0.0, 1.0]},
                    {"tenant": "b", "text": "gamma", "vector": [1.0, 1.0]},
                ],
            )
            indexes = client.prepare_index_params()
            indexes.add_index("vector", index_type="FLAT", metric_type="COSINE")
            client.create_index(args.collection, indexes)
        client.load_collection(args.collection)
        queries = [[1.0, 0.0], [0.0, 1.0]]

        across_queries = (
            table_chain()
            .concat()
            .group_by("tenant")
            .aggregate(
                {
                    "count": agg.count(),
                    "texts": agg.collect_list(col("text")),
                    "queries": agg.collect_list(col("$query_index")),
                }
            )
            .sort("tenant", desc=False)
        )
        result = client.search(
            args.collection,
            data=queries,
            anns_field="vector",
            limit=3,
            search_params={"metric_type": "COSINE"},
            function_chains=across_queries,
        )
        assert result.row_count == 2
        assert [(row["tenant"], row["count"]) for row in result.rows] == [("a", 4), ("b", 2)]
        assert set(result.columns) == {"tenant", "count", "texts", "queries"}
        print("CROSS_NQ", result.schema, result.rows)

        per_route = (
            table_chain("ann_inputs")
            .concat()
            .group_by("$ann_index")
            .aggregate(
                {
                    "count": agg.count(),
                    "ids": agg.collect_list(col("$id")),
                    "queries": agg.collect_list(col("$query_index")),
                }
            )
            .sort("$ann_index", desc=False)
        )
        routes = [
            AnnSearchRequest(queries, "vector", {"metric_type": "COSINE"}, limit=3),
            AnnSearchRequest(queries[::-1], "vector", {"metric_type": "COSINE"}, limit=2),
        ]
        result = client.hybrid_search(args.collection, routes, limit=1, function_chains=per_route)
        assert [(row["$ann_index"], row["count"]) for row in result.rows] == [(0, 6), (1, 4)]
        assert len({pk for row in result.rows for pk in row["ids"]}) <= 3
        print("ANN_INPUTS", result.schema, result.rows)

        query_groups = (
            table_chain()
            .group_by("tenant")
            .aggregate(
                {
                    "count": agg.count(),
                    "texts": agg.collect_list(col("text")),
                }
            )
            .sort("tenant", desc=False)
        )
        result = client.query(
            args.collection,
            filter="",
            limit=10,
            function_chains=query_groups,
        )
        assert [(row["tenant"], row["count"]) for row in result.rows] == [("a", 2), ("b", 1)]
        print("QUERY", result.schema, result.rows)

        empty = client.query(
            args.collection,
            filter="id < 0",
            limit=10,
            function_chains=table_chain().aggregate({"count": agg.count()}),
        )
        assert empty.rows == [{"count": 0}]
        keyed_empty = client.query(
            args.collection,
            filter="id < 0",
            limit=10,
            function_chains=table_chain().group_by("tenant").aggregate({"count": agg.count()}),
        )
        assert keyed_empty.row_count == 0
        assert keyed_empty.columns == ["tenant", "count"]
        print("EMPTY", empty.rows, keyed_empty.schema)

        legacy = client.search(
            args.collection,
            data=queries,
            anns_field="vector",
            limit=3,
            search_params={"metric_type": "COSINE"},
        )
        assert len(legacy) == 2
        assert all(len(hits) == 3 for hits in legacy)
        print("LEGACY_HITS", [len(hits) for hits in legacy])

        if args.resource_name:
            summary = (
                table_chain()
                .concat()
                .reduce(
                    "summary",
                    fn.py_udf(
                        col("text"),
                        resource_name=args.resource_name,
                        udf_params={"mode": "summary"},
                    ),
                )
                .select(["summary"])
            )
            result = client.search(
                args.collection,
                data=queries,
                anns_field="vector",
                limit=3,
                search_params={"metric_type": "COSINE"},
                function_chains=summary,
            )
            assert result.row_count == 1
            assert result.rows[0]["summary"]["count"] == 6
            assert result.rows[0]["summary"]["query_count"] == 2
            print("REDUCE", result.schema, result.rows)
    finally:
        client.close()


if __name__ == "__main__":
    main()
