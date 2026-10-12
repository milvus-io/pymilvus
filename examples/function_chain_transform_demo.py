"""Exercise the transform_demo wheel from Milvus's PyUDF runtime examples.

Build/upload the wheel first, then register its object path as a file resource.
This example uses deterministic two-dimensional vectors, without a model API.
"""

# This runnable verification example prints results and checks its contract.
# ruff: noqa: INP001, S101, T201, PT017, PT018

import argparse

from pymilvus import DataType, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.exceptions import IngestionFunctionChainError
from pymilvus.function_chain import col, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--resource-name", default="transform_demo")
    parser.add_argument(
        "--wheel-path",
        help="Object path in Milvus storage; register before creating the collection",
    )
    parser.add_argument("--collection", default="function_chain_transform_demo")
    args = parser.parse_args()
    client = MilvusClient(args.uri, timeout=60)
    try:
        if args.wheel_path:
            client.add_file_resource(args.resource_name, args.wheel_path)
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("text", DataType.VARCHAR, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        schema.add_field(
            "tags", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=16, max_length=32
        )
        schema.add_function_chain(
            FunctionChain(FunctionChainStage.PRE_INSERT, "chunk_and_embed")
            .flat_map(
                "chunk",
                fn.py_udf(col("document", DataType.VARCHAR), resource_name=args.resource_name),
                position="chunk_index",
            )
            .select(["chunk.text", "chunk.vector", "chunk.tags"], ["text", "vector", "tags"])
        )
        client.create_collection(args.collection, schema=schema)
        result = client.insert(args.collection, [{"document": "alpha beta"}, {"document": "gamma"}])
        assert result["insert_count"] == len(result["ids"]) == 3
        assert result["ingestion_result"]["source_row_indexes"] == [0, 0, 1]
        print("INSERT", dict(result))

        index = client.prepare_index_params()
        index.add_index("vector", index_type="FLAT", metric_type="COSINE")
        client.create_index(args.collection, index)
        client.load_collection(args.collection)
        pre = (
            FunctionChain(FunctionChainStage.PRE_PROCESS, "translate_and_embed")
            .map(
                "query",
                fn.py_udf(
                    col("query"),
                    resource_name=args.resource_name,
                    udf_params={"mode": "translation"},
                ),
            )
            .map(
                "query",
                fn.py_udf(
                    col("query"), resource_name=args.resource_name, udf_params={"mode": "embedding"}
                ),
            )
        )
        post = (
            FunctionChain(FunctionChainStage.POST_PROCESS, "highlight")
            .map(
                "highlight",
                fn.py_udf(
                    col("text"), resource_name=args.resource_name, udf_params={"mode": "highlight"}
                ),
            )
            .select(["highlight"])
        )
        rerank = (
            FunctionChain(FunctionChainStage.L2_RERANK, "query_aware_rerank")
            .map(
                "$score",
                fn.py_udf(
                    col("text"), resource_name=args.resource_name, udf_params={"mode": "rerank"}
                ),
            )
            .sort("$score", desc=True)
            .limit(3)
        )
        results = client.search(
            args.collection,
            data=["hello", "world"],
            anns_field="vector",
            limit=3,
            function_chains=[pre, rerank, post],
        )
        assert len(results) == 2
        assert all(len(hits) == 3 for hits in results)
        assert all(set(hit["entity"]) == {"highlight"} for hits in results for hit in hits)
        for query_index, query in enumerate(["hello", "world"]):
            for hit in results[query_index]:
                highlight = hit["entity"]["highlight"]
                assert highlight["query"] == query and highlight["query_index"] == query_index
                assert highlight["query_count"] == 2 and highlight["total_hits"] == 6
        assert results[0][0]["entity"]["highlight"]["text"] == "alpha"
        assert results[1][0]["entity"]["highlight"]["text"] == "gamma"
        print("SEARCH", results)

        zero = client.insert(args.collection, [{"document": ""}])
        assert zero["insert_count"] == 0 and len(zero["ids"]) == 0
        assert zero["ingestion_result"]["source_offsets"] == [0]
        print("EMPTY", dict(zero))
        try:
            client.insert(args.collection, [{"document": "__FAIL__"}])
        except IngestionFunctionChainError as error:
            assert error.retriable is False and error.mutation_result.insert_count == 0
            assert error.unknown_outcome is False
            print("UDF_FAILURE", error.code, error.mutation_result.ingestion_result)
        else:
            message = "Expected the demo model to fail"
            raise AssertionError(message)
        rows = client.query(args.collection, filter="", output_fields=["text", "tags"], limit=10)
        assert len(rows) == 3
        print("STORED", rows)
    finally:
        client.close()


if __name__ == "__main__":
    main()
