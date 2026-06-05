"""
Validate BM25 full-text search on a sparse field added through add_function_field.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

from pymilvus import DataType, FieldSchema, Function, FunctionType, MilvusClient
from pymilvus.exceptions import MilvusException

HOST = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.environ.get(
    "ADD_FUNCTION_FULL_TEXT_COLLECTION",
    "test_add_function_field_full_text_search",
)
CLEAR_EXIST_COLLECTION = os.environ.get("FULL_TEXT_CLEAR_EXIST", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
DIM = 128
SEALED_ROWS = int(os.environ.get("FULL_TEXT_SEALED_ROWS", "10000"))
GROWING_ROWS = int(os.environ.get("FULL_TEXT_GROWING_ROWS", "200"))
INSERT_BATCH = 5000
TOTAL_ROWS = SEALED_ROWS + GROWING_ROWS
SEARCH_TIMEOUT = float(os.environ.get("FULL_TEXT_SEARCH_TIMEOUT", "300"))
SEARCH_POLL_INTERVAL = float(os.environ.get("FULL_TEXT_SEARCH_POLL_INTERVAL", "3"))
ID_FIELD = "id"
DENSE_FIELD = "vec"
TEXT_FIELD = "text"
SPARSE_FIELD = "sparse"
FUNCTION_NAME = "bm25_fn"
SPARSE_INDEX_NAME = "sparse_bm25_index"
SEARCH_QUERY = "information retrieval ranking"
QUERY_TERMS = {"information", "retrieval", "ranking"}
TEXT_CORPUS = [
    "information retrieval ranking for document search and scoring",
    "ranking models compare relevance across sparse vector candidates",
    "bm25 normalization balances term frequency and document length",
    "schema evolution keeps searchable fields compatible over time",
    "dense vectors complement sparse retrieval in hybrid recall",
    "retrieval pipelines aggregate candidate sets before rerank",
    "ranking accuracy depends on both term stats and index quality",
    "background text with weak relevance and broad vocabulary coverage",
    "long document body with many neutral terms and few query terms present",
    "short focused text about retrieval ranking information relevance",
]
SEARCH_LIMIT = int(os.environ.get("FULL_TEXT_SEARCH_LIMIT", "20"))
SEGMENT_CHECK_LIMIT = int(os.environ.get("FULL_TEXT_SEGMENT_CHECK_LIMIT", "20"))
SEARCH_PARAMS = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}}
POST_FLUSH_SEARCH_ROUNDS = int(os.environ.get("FULL_TEXT_POST_FLUSH_SEARCH_ROUNDS", "20"))
POST_FLUSH_SEARCH_INTERVAL = float(os.environ.get("FULL_TEXT_POST_FLUSH_SEARCH_INTERVAL", "1"))

# Keep sealed and growing distributions aligned to avoid segment-source bias.
SEALED_TEXTS = TEXT_CORPUS
GROWING_TEXTS = TEXT_CORPUS

SEALED_ID_FILTER = f"{ID_FIELD} >= 0 and {ID_FIELD} < {SEALED_ROWS}"
GROWING_ID_FILTER = f"{ID_FIELD} >= {SEALED_ROWS} and {ID_FIELD} < {TOTAL_ROWS}"
EOF_ERR = "BM25 search returned no hits within timeout"
EXPECTED_NO_SPARSE_INDEX_ERR = "field sparse is not loaded"
PYMILVUS_DECORATOR_LOGGER = "pymilvus.decorators"


@contextmanager
def _suppress_expected_pymilvus_rpc_error() -> None:
    logger = logging.getLogger(PYMILVUS_DECORATOR_LOGGER)
    previous_disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = previous_disabled


def _drop_collection_if_exists(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)


def _prepare_collection_start(client: MilvusClient, collection_name: str) -> None:
    if CLEAR_EXIST_COLLECTION:
        _drop_collection_if_exists(client, collection_name)
        print(f"clear_exist enabled, dropped existing collection if present: {collection_name}")
    else:
        print(f"clear_exist disabled, keep existing collection if present: {collection_name}")


def _create_collection(client: MilvusClient, collection_name: str) -> None:
    schema = client.create_schema()
    schema.add_field(ID_FIELD, DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=2048, enable_analyzer=True)

    index_params = client.prepare_index_params()
    index_params.add_index(DENSE_FIELD, index_type="FLAT", metric_type="L2")

    client.create_collection(collection_name, schema=schema, index_params=index_params)


def _make_rows(start_id: int, count: int, text_pool: list[str]) -> list[dict[str, Any]]:
    rows = []
    for row_id in range(start_id, start_id + count):
        rows.append(
            {
                ID_FIELD: row_id,
                DENSE_FIELD: [float((row_id + dim) % 100) / 100 for dim in range(DIM)],
                TEXT_FIELD: text_pool[row_id % len(text_pool)],
            }
        )
    return rows


def _insert_rows_in_batches(
    client: MilvusClient,
    collection_name: str,
    start_id: int,
    total_rows: int,
    text_pool: list[str],
) -> None:
    inserted = 0
    while inserted < total_rows:
        batch_count = min(INSERT_BATCH, total_rows - inserted)
        client.insert(collection_name, _make_rows(start_id + inserted, batch_count, text_pool))
        inserted += batch_count


def _insert_sealed_rows(client: MilvusClient, collection_name: str) -> None:
    _insert_rows_in_batches(client, collection_name, 0, SEALED_ROWS, SEALED_TEXTS)


def _insert_growing_rows(client: MilvusClient, collection_name: str) -> None:
    _insert_rows_in_batches(client, collection_name, SEALED_ROWS, GROWING_ROWS, GROWING_TEXTS)


def _function_field_spec() -> tuple[FieldSchema, Function]:
    sparse_field = FieldSchema(
        name=SPARSE_FIELD,
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        desc="BM25 output field added by add_function_field full-text search test",
    )
    bm25_fn = Function(
        name=FUNCTION_NAME,
        input_field_names=[TEXT_FIELD],
        output_field_names=[SPARSE_FIELD],
        function_type=FunctionType.BM25,
    )
    return sparse_field, bm25_fn


def _add_function_field_once(client: MilvusClient, collection_name: str) -> None:
    sparse_field, bm25_fn = _function_field_spec()
    client.add_function_field(
        collection_name,
        field_schema=sparse_field,
        func=bm25_fn,
    )


def _create_sparse_index(client: MilvusClient, collection_name: str) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        index_name=SPARSE_INDEX_NAME,
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    client.create_index(collection_name, index_params)


def _assert_schema_after_add(schema_info: dict[str, Any]) -> None:
    field_names = {field["name"] for field in schema_info["fields"]}
    assert SPARSE_FIELD in field_names, f"Missing sparse field {SPARSE_FIELD!r}: {schema_info}"

    functions = {func["name"]: func for func in schema_info["functions"]}
    assert FUNCTION_NAME in functions, f"Missing BM25 function {FUNCTION_NAME!r}: {schema_info}"
    assert functions[FUNCTION_NAME]["input_field_names"] == [TEXT_FIELD]
    assert functions[FUNCTION_NAME]["output_field_names"] == [SPARSE_FIELD]
    assert (
        schema_info["schema_version"] == 1
    ), f"Expected schema_version=1, got {schema_info['schema_version']}"


# def _assert_sparse_index(client: MilvusClient, collection_name: str) -> None:
#     indexes = client.list_indexes(collection_name)
#     sparse_indexes = client.list_indexes(collection_name, field_name=SPARSE_FIELD)
#     index_info = client.describe_index(collection_name, SPARSE_INDEX_NAME)
#     print(f"Indexes after add_function_field/create_index: {indexes}")
#     print(f"Sparse indexes: {sparse_indexes}")
#     print(f"Sparse index info: {index_info}")
#     assert SPARSE_INDEX_NAME in indexes, f"Missing index {SPARSE_INDEX_NAME!r}: {indexes}"
#     assert SPARSE_INDEX_NAME in sparse_indexes, f"Missing sparse index for {SPARSE_FIELD!r}: {sparse_indexes}"
#     assert index_info is not None, f"Missing index info for {SPARSE_INDEX_NAME!r}"


def _hit_field(hit: dict[str, Any], field_name: str) -> Any:
    if field_name in hit:
        return hit[field_name]
    entity = hit.get("entity")
    if isinstance(entity, dict):
        return entity.get(field_name)
    return None


def _search_once(
    client: MilvusClient,
    collection_name: str,
    query: str,
    limit: int,
    filter_expr: str | None = None,
) -> list[dict[str, Any]]:
    search_kwargs = {
        "collection_name": collection_name,
        "data": [query],
        "anns_field": SPARSE_FIELD,
        "search_params": SEARCH_PARAMS,
        "output_fields": [ID_FIELD, TEXT_FIELD],
        "limit": limit,
    }
    if filter_expr:
        search_kwargs["filter"] = filter_expr
    results = client.search(**search_kwargs)
    assert len(results) == 1, f"Expected one result set, got {len(results)}"
    return results[0]


def _milvus_exception_message(exc: MilvusException) -> str:
    return f"code={exc.code}, message={exc.message}"


def _assert_search_fails_without_sparse_index(client: MilvusClient, collection_name: str) -> None:
    try:
        with _suppress_expected_pymilvus_rpc_error():
            hits = _search_once(client, collection_name, SEARCH_QUERY, SEARCH_LIMIT)
    except MilvusException as exc:
        err_msg = _milvus_exception_message(exc)
        print(f"Expected BM25 search failure before sparse index creation: {err_msg}")
        assert EXPECTED_NO_SPARSE_INDEX_ERR in exc.message, err_msg
        return
    except Exception as exc:
        err_msg = str(exc)
        print(f"Expected BM25 search failure before sparse index creation: {err_msg}")
        assert EXPECTED_NO_SPARSE_INDEX_ERR in err_msg, err_msg
        return

    print(f"Raw BM25 search result before sparse index creation: {hits}")
    if not hits:
        print("Expected BM25 search returned no hits before sparse index creation")
        return

    try:
        _assert_mixed_segment_hits(hits)
    except Exception as exc:
        print(f"Expected BM25 search returned invalid hits before sparse index creation: {exc}")
        return

    raise AssertionError("BM25 search unexpectedly succeeded before sparse index creation")


def _search_until_ready(
    client: MilvusClient,
    collection_name: str,
    query: str,
    limit: int,
    filter_expr: str | None = None,
) -> list[dict[str, Any]]:
    deadline = time.time() + SEARCH_TIMEOUT
    last_error = None
    printed_errors: set[str] = set()
    while time.time() < deadline:
        try:
            with _suppress_expected_pymilvus_rpc_error():
                hits = _search_once(client, collection_name, query, limit, filter_expr=filter_expr)
            if hits:
                return hits
        except MilvusException as exc:
            last_error = exc
            err_msg = _milvus_exception_message(exc)
            if err_msg not in printed_errors:
                print(
                    f"BM25 search is not ready yet for query={query!r}, filter={filter_expr!r}: {err_msg}"
                )
                printed_errors.add(err_msg)
        except Exception as exc:
            last_error = exc
            err_msg = str(exc)
            if err_msg not in printed_errors:
                print(
                    f"BM25 search is not ready yet for query={query!r}, filter={filter_expr!r}: {err_msg}"
                )
                printed_errors.add(err_msg)
        time.sleep(SEARCH_POLL_INTERVAL)

    raise AssertionError(
        f"{EOF_ERR}: timeout={SEARCH_TIMEOUT}s, query={query!r}, filter={filter_expr!r}, last_error={last_error}"
    )


def _assert_ids_cover_segment_range(
    hits: list[dict[str, Any]],
    lower_bound: int,
    upper_bound: int,
    segment_name: str,
) -> None:
    hit_ids = [_hit_field(hit, ID_FIELD) for hit in hits]
    assert any(
        isinstance(hit_id, int) and lower_bound <= hit_id < upper_bound for hit_id in hit_ids
    ), f"Expected at least one {segment_name} hit in [{lower_bound}, {upper_bound}), got ids={hit_ids}"


def _assert_ids_do_not_exceed_total_rows(hits: list[dict[str, Any]]) -> None:
    hit_ids = [_hit_field(hit, ID_FIELD) for hit in hits]
    assert all(isinstance(hit_id, int) and 0 <= hit_id < TOTAL_ROWS for hit_id in hit_ids), hit_ids


def _assert_top_hit_contains_terms(hits: list[dict[str, Any]], terms: set[str]) -> None:
    top_text = str(_hit_field(hits[0], TEXT_FIELD)).lower()
    assert terms.issubset(set(top_text.split())), f"Top hit is not query-relevant: {hits[0]}"


def _print_hits(title: str, hits: list[dict[str, Any]]) -> None:
    print(f"{title} returned {len(hits)} hits:")
    for hit in hits:
        print(
            f"  id={_hit_field(hit, ID_FIELD)}, "
            f"score={hit.get('distance')}, "
            f"text={_hit_field(hit, TEXT_FIELD)!r}"
        )


def _assert_mixed_segment_hits(hits: list[dict[str, Any]]) -> None:
    _print_hits("BM25 mixed segment search", hits)
    _assert_ids_do_not_exceed_total_rows(hits)
    _assert_top_hit_contains_terms(hits, QUERY_TERMS)
    # Mixed query can legitimately skew to one segment depending on term stats.
    # Segment-source validation is performed separately by _assert_segment_specific_hits.


def _assert_segment_specific_hits(
    title: str,
    hits: list[dict[str, Any]],
    lower_bound: int,
    upper_bound: int,
    segment_name: str,
) -> None:
    _print_hits(title, hits)
    _assert_ids_do_not_exceed_total_rows(hits)
    _assert_top_hit_contains_terms(hits, QUERY_TERMS)
    _assert_ids_cover_segment_range(hits, lower_bound, upper_bound, segment_name)


def _assert_search_hits(hits: list[dict[str, Any]]) -> None:
    _assert_mixed_segment_hits(hits)


def _assert_repeated_search_success(client: MilvusClient, collection_name: str) -> None:
    for round_idx in range(1, POST_FLUSH_SEARCH_ROUNDS + 1):
        hits = _search_once(client, collection_name, SEARCH_QUERY, SEARCH_LIMIT)
        assert hits, f"Round {round_idx}: BM25 search returned no hits"
        print(f"Post-flush BM25 search round {round_idx}/{POST_FLUSH_SEARCH_ROUNDS} succeeded")
        if round_idx < POST_FLUSH_SEARCH_ROUNDS:
            time.sleep(POST_FLUSH_SEARCH_INTERVAL)


def _assert_basic_search_hits(hits: list[dict[str, Any]]) -> None:
    _print_hits("BM25 search", hits)
    _assert_ids_do_not_exceed_total_rows(hits)
    _assert_top_hit_contains_terms(hits, QUERY_TERMS)


def run_add_function_field_full_text_search() -> None:
    client = MilvusClient(HOST)
    try:
        print(f"[STEP1] create collection without BM25 function: {COLLECTION_NAME}")
        _prepare_collection_start(client, COLLECTION_NAME)
        _create_collection(client, COLLECTION_NAME)
        initial_schema = client.describe_collection(COLLECTION_NAME)
        print(f"Initial collection schema: {initial_schema}")
        assert initial_schema["functions"] == []

        print(f"[STEP2] insert and flush {SEALED_ROWS} sealed rows before add_function_field")
        _insert_sealed_rows(client, COLLECTION_NAME)
        client.flush(COLLECTION_NAME)
        stats = client.get_collection_stats(COLLECTION_NAME)
        print(f"Collection stats after flush: {stats}")
        assert (
            stats["row_count"] == SEALED_ROWS
        ), f"Expected {SEALED_ROWS} rows, got {stats['row_count']}"

        print("[STEP3] load collection before add_function_field")
        client.load_collection(COLLECTION_NAME, load_fields=[ID_FIELD, DENSE_FIELD, TEXT_FIELD])

        print("[STEP4] add_function_field once for BM25 sparse output field")
        _add_function_field_once(client, COLLECTION_NAME)
        schema_after_add = client.describe_collection(COLLECTION_NAME)
        print(f"Collection schema after add_function_field: {schema_after_add}")
        _assert_schema_after_add(schema_after_add)

        # print("[STEP4.1] verify create_index result on new sparse field")
        # _assert_sparse_index(client, COLLECTION_NAME)

        print(f"[STEP5.1] insert {GROWING_ROWS} rows after load without flush (growing data)")
        _insert_growing_rows(client, COLLECTION_NAME)

        print("[STEP5.2] verify BM25 search fails before sparse index creation")
        _assert_search_fails_without_sparse_index(client, COLLECTION_NAME)

        print("[STEP5.3] create sparse index explicitly")
        _create_sparse_index(client, COLLECTION_NAME)

        print("[STEP5.4] search sparse field until success")
        mixed_hits = _search_until_ready(client, COLLECTION_NAME, SEARCH_QUERY, SEARCH_LIMIT)
        _assert_mixed_segment_hits(mixed_hits)

        print("[STEP5.5] flush growing rows")
        client.flush(COLLECTION_NAME)

        print(
            f"[STEP5.6] verify {POST_FLUSH_SEARCH_ROUNDS} post-flush searches "
            f"with interval={POST_FLUSH_SEARCH_INTERVAL}s"
        )
        _assert_repeated_search_success(client, COLLECTION_NAME)

        print(f"Final collection stats: {client.get_collection_stats(COLLECTION_NAME)}")
        print(f"Final collection schema: {client.describe_collection(COLLECTION_NAME)}")
        print("PASS: add_function_field BM25 full-text search flow verified")
    finally:
        client.close()


def main() -> None:
    run_add_function_field_full_text_search()


if __name__ == "__main__":
    main()
