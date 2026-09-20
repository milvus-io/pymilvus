"""
Validate MinHash search on a binary field added through add_function_field.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Any

from pymilvus import DataType, FieldSchema, Function, FunctionType, MilvusClient
from pymilvus.exceptions import MilvusException

HOST = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.environ.get(
    "ADD_FUNCTION_MINHASH_COLLECTION",
    "test_add_function_field_minhash_search",
)
CLEAR_EXIST_COLLECTION = os.environ.get("MINHASH_CLEAR_EXIST", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
DIM = 128
NUM_HASHES = int(os.environ.get("MINHASH_NUM_HASHES", "16"))
MINHASH_DIM = NUM_HASHES * 32
MINHASH_LSH_BAND = int(os.environ.get("MINHASH_LSH_BAND", "8"))
SHINGLE_SIZE = int(os.environ.get("MINHASH_SHINGLE_SIZE", "3"))
SEALED_ROWS = int(os.environ.get("MINHASH_SEALED_ROWS", "10000"))
GROWING_ROWS = int(os.environ.get("MINHASH_GROWING_ROWS", "200"))
INSERT_BATCH = 5000
TOTAL_ROWS = SEALED_ROWS + GROWING_ROWS
SEARCH_TIMEOUT = float(os.environ.get("MINHASH_SEARCH_TIMEOUT", "300"))
SEARCH_POLL_INTERVAL = float(os.environ.get("MINHASH_SEARCH_POLL_INTERVAL", "3"))
ID_FIELD = "id"
DENSE_FIELD = "vec"
TEXT_FIELD = "text"
MINHASH_FIELD = "minhash_signature"
FUNCTION_NAME = "minhash_fn"
MINHASH_INDEX_NAME = "minhash_lsh_index"
SEARCH_QUERY = "information retrieval ranking document search scoring"
QUERY_CORE_TERMS = {"information", "retrieval", "ranking", "document", "search", "scoring"}
TEXT_RANDOM_SEED = int(os.environ.get("MINHASH_TEXT_RANDOM_SEED", "20240608"))
SIMILAR_TEXT_EVERY = max(1, int(os.environ.get("MINHASH_SIMILAR_TEXT_EVERY", "37")))
BACKGROUND_VOCABULARY = [
    "archive",
    "batch",
    "cache",
    "catalog",
    "cluster",
    "column",
    "compaction",
    "cursor",
    "dataset",
    "delta",
    "embedding",
    "filter",
    "graph",
    "index",
    "ingest",
    "journal",
    "latency",
    "manifest",
    "metadata",
    "partition",
    "pipeline",
    "query",
    "replica",
    "segment",
    "snapshot",
    "storage",
    "token",
    "transaction",
    "vector",
    "version",
]
SIMILAR_VARIANTS = [
    "information retrieval ranking document search scoring",
    "information retrieval ranking relevant document search scoring",
    "document search scoring improves information retrieval ranking",
    "retrieval ranking scores documents for information search",
]
SEARCH_LIMIT = int(os.environ.get("MINHASH_SEARCH_LIMIT", "20"))
SEARCH_REFINE_K = int(os.environ.get("MINHASH_SEARCH_REFINE_K", "50"))
SEARCH_PARAMS = {
    "metric_type": "MHJACCARD",
    "params": {
        "mh_search_with_jaccard": True,
        "refine_k": SEARCH_REFINE_K,
    },
}
POST_FLUSH_SEARCH_ROUNDS = int(os.environ.get("MINHASH_POST_FLUSH_SEARCH_ROUNDS", "20"))
POST_FLUSH_SEARCH_INTERVAL = float(os.environ.get("MINHASH_POST_FLUSH_SEARCH_INTERVAL", "1"))
SEALED_ID_FILTER = f"{ID_FIELD} >= 0 and {ID_FIELD} < {SEALED_ROWS}"
GROWING_ID_FILTER = f"{ID_FIELD} >= {SEALED_ROWS} and {ID_FIELD} < {TOTAL_ROWS}"

EOF_ERR = "MinHash search returned no hits within timeout"
EXPECTED_NO_MINHASH_INDEX_ERRS = (
    f"field {MINHASH_FIELD} is not loaded",
    f"field index of the field: {MINHASH_FIELD} is not loaded",
    "index not found",
    "index doesn't exist",
    "failed to get index",
)
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


def _make_background_text(row_id: int) -> str:
    rng = random.Random(TEXT_RANDOM_SEED + row_id)
    words = rng.sample(BACKGROUND_VOCABULARY, 12)
    return " ".join(words)


def _make_similar_text(row_id: int) -> str:
    rng = random.Random(TEXT_RANDOM_SEED * 2 + row_id)
    variant = SIMILAR_VARIANTS[row_id % len(SIMILAR_VARIANTS)]
    noise = " ".join(rng.sample(BACKGROUND_VOCABULARY, 3))
    return f"{variant} {noise} sample {row_id}"


def _make_text(row_id: int) -> str:
    if row_id % SIMILAR_TEXT_EVERY == 0:
        return _make_similar_text(row_id)
    return _make_background_text(row_id)


def _make_rows(start_id: int, count: int) -> list[dict[str, Any]]:
    rows = []
    for row_id in range(start_id, start_id + count):
        rows.append(
            {
                ID_FIELD: row_id,
                DENSE_FIELD: [float((row_id + dim) % 100) / 100 for dim in range(DIM)],
                TEXT_FIELD: _make_text(row_id),
            }
        )
    return rows


def _insert_rows_in_batches(
    client: MilvusClient,
    collection_name: str,
    start_id: int,
    total_rows: int,
) -> None:
    inserted = 0
    while inserted < total_rows:
        batch_count = min(INSERT_BATCH, total_rows - inserted)
        client.insert(collection_name, _make_rows(start_id + inserted, batch_count))
        inserted += batch_count


def _insert_sealed_rows(client: MilvusClient, collection_name: str) -> None:
    _insert_rows_in_batches(client, collection_name, 0, SEALED_ROWS)


def _insert_growing_rows(client: MilvusClient, collection_name: str) -> None:
    _insert_rows_in_batches(client, collection_name, SEALED_ROWS, GROWING_ROWS)


def _function_field_spec() -> tuple[FieldSchema, Function]:
    minhash_field = FieldSchema(
        name=MINHASH_FIELD,
        dtype=DataType.BINARY_VECTOR,
        dim=MINHASH_DIM,
        desc="MinHash output field added by add_function_field search test",
    )
    minhash_fn = Function(
        name=FUNCTION_NAME,
        input_field_names=[TEXT_FIELD],
        output_field_names=[MINHASH_FIELD],
        function_type=FunctionType.MINHASH,
        params={
            "num_hashes": NUM_HASHES,
            "shingle_size": SHINGLE_SIZE,
            "token_level": "word",
        },
    )
    return minhash_field, minhash_fn


def _add_function_field_once(client: MilvusClient, collection_name: str) -> None:
    minhash_field, minhash_fn = _function_field_spec()
    client.add_function_field(
        collection_name,
        field_schema=minhash_field,
        func=minhash_fn,
    )


def _create_minhash_index(client: MilvusClient, collection_name: str) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=MINHASH_FIELD,
        index_type="MINHASH_LSH",
        index_name=MINHASH_INDEX_NAME,
        metric_type="MHJACCARD",
        params={
            "mh_lsh_band": MINHASH_LSH_BAND,
            "mh_element_bit_width": 32,
            "with_raw_data": True,
        },
    )
    client.create_index(collection_name, index_params)


def _assert_schema_after_add(schema_info: dict[str, Any]) -> None:
    field_names = {field["name"] for field in schema_info["fields"]}
    assert MINHASH_FIELD in field_names, f"Missing MinHash field {MINHASH_FIELD!r}: {schema_info}"

    functions = {func["name"]: func for func in schema_info["functions"]}
    assert FUNCTION_NAME in functions, f"Missing MinHash function {FUNCTION_NAME!r}: {schema_info}"
    assert functions[FUNCTION_NAME]["input_field_names"] == [TEXT_FIELD]
    assert functions[FUNCTION_NAME]["output_field_names"] == [MINHASH_FIELD]


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
        "anns_field": MINHASH_FIELD,
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


def _assert_search_fails_without_minhash_index(client: MilvusClient, collection_name: str) -> None:
    try:
        with _suppress_expected_pymilvus_rpc_error():
            hits = _search_once(client, collection_name, SEARCH_QUERY, SEARCH_LIMIT)
    except MilvusException as exc:
        err_msg = _milvus_exception_message(exc)
        print(f"Expected MinHash search failure before index creation: {err_msg}")
        assert any(expected in exc.message for expected in EXPECTED_NO_MINHASH_INDEX_ERRS), err_msg
        return
    except Exception as exc:
        err_msg = str(exc)
        print(f"Expected MinHash search failure before index creation: {err_msg}")
        assert any(expected in err_msg for expected in EXPECTED_NO_MINHASH_INDEX_ERRS), err_msg
        return

    print(f"Raw MinHash search result before index creation: {hits}")
    if not hits:
        print("Expected MinHash search returned no hits before index creation")
        return

    try:
        _assert_search_hits("Unexpected pre-index MinHash search", hits)
    except Exception as exc:
        print(f"Expected MinHash search returned invalid hits before index creation: {exc}")
        return

    raise AssertionError("MinHash search unexpectedly succeeded before index creation")


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
                    f"MinHash search is not ready yet for query={query!r}, "
                    f"filter={filter_expr!r}: {err_msg}"
                )
                printed_errors.add(err_msg)
        except Exception as exc:
            last_error = exc
            err_msg = str(exc)
            if err_msg not in printed_errors:
                print(
                    f"MinHash search is not ready yet for query={query!r}, "
                    f"filter={filter_expr!r}: {err_msg}"
                )
                printed_errors.add(err_msg)
        time.sleep(SEARCH_POLL_INTERVAL)

    raise AssertionError(
        f"{EOF_ERR}: timeout={SEARCH_TIMEOUT}s, query={query!r}, "
        f"filter={filter_expr!r}, last_error={last_error}"
    )


def _assert_ids_do_not_exceed_total_rows(hits: list[dict[str, Any]]) -> None:
    hit_ids = [_hit_field(hit, ID_FIELD) for hit in hits]
    assert all(isinstance(hit_id, int) and 0 <= hit_id < TOTAL_ROWS for hit_id in hit_ids), hit_ids


def _assert_has_similar_hit(hits: list[dict[str, Any]]) -> None:
    for hit in hits:
        text = str(_hit_field(hit, TEXT_FIELD)).lower()
        overlap = QUERY_CORE_TERMS.intersection(set(text.split()))
        if len(overlap) >= 4:
            return
    raise AssertionError(f"No query-relevant hit found: {hits}")


def _assert_ids_in_range(
    hits: list[dict[str, Any]],
    lower_bound: int,
    upper_bound: int,
    range_name: str,
) -> None:
    hit_ids = [_hit_field(hit, ID_FIELD) for hit in hits]
    assert hit_ids, f"Expected {range_name} hits, got none"
    assert all(
        isinstance(hit_id, int) and lower_bound <= hit_id < upper_bound for hit_id in hit_ids
    ), f"Expected all {range_name} hits in [{lower_bound}, {upper_bound}), got ids={hit_ids}"


def _print_hits(title: str, hits: list[dict[str, Any]]) -> None:
    print(f"{title} returned {len(hits)} hits:")
    for hit in hits:
        print(
            f"  id={_hit_field(hit, ID_FIELD)}, "
            f"score={hit.get('distance')}, "
            f"text={_hit_field(hit, TEXT_FIELD)!r}"
        )


def _assert_search_hits(title: str, hits: list[dict[str, Any]]) -> None:
    _print_hits(title, hits)
    _assert_ids_do_not_exceed_total_rows(hits)
    _assert_has_similar_hit(hits)


def _assert_repeated_search_success(client: MilvusClient, collection_name: str) -> None:
    for round_idx in range(1, POST_FLUSH_SEARCH_ROUNDS + 1):
        hits = _search_once(client, collection_name, SEARCH_QUERY, SEARCH_LIMIT)
        assert hits, f"Round {round_idx}: MinHash search returned no hits"
        print(f"Post-flush MinHash search round {round_idx}/{POST_FLUSH_SEARCH_ROUNDS} succeeded")
        if round_idx < POST_FLUSH_SEARCH_ROUNDS:
            time.sleep(POST_FLUSH_SEARCH_INTERVAL)


def run_add_function_field_minhash_search() -> None:
    client = MilvusClient(HOST)
    try:
        print(f"[STEP1] create collection without MinHash function: {COLLECTION_NAME}")
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
        assert stats["row_count"] == SEALED_ROWS, f"Expected {SEALED_ROWS} rows, got {stats['row_count']}"

        print("[STEP3] load collection before add_function_field")
        client.load_collection(COLLECTION_NAME, load_fields=[ID_FIELD, DENSE_FIELD, TEXT_FIELD])

        print("[STEP4] add_function_field once for MinHash binary output field")
        _add_function_field_once(client, COLLECTION_NAME)
        schema_after_add = client.describe_collection(COLLECTION_NAME)
        print(f"Collection schema after add_function_field: {schema_after_add}")
        _assert_schema_after_add(schema_after_add)

        print(f"[STEP5.1] insert {GROWING_ROWS} rows after load without flush (growing data)")
        _insert_growing_rows(client, COLLECTION_NAME)

        print("[STEP5.2] verify MinHash search fails before index creation")
        _assert_search_fails_without_minhash_index(client, COLLECTION_NAME)

        print("[STEP5.3] create MinHash LSH index explicitly")
        _create_minhash_index(client, COLLECTION_NAME)

        print("[STEP5.4] search MinHash field until success")
        hits = _search_until_ready(client, COLLECTION_NAME, SEARCH_QUERY, SEARCH_LIMIT)
        _assert_search_hits("MinHash mixed search", hits)

        print("[STEP5.4.1] verify sealed rows are searchable")
        sealed_hits = _search_until_ready(
            client,
            COLLECTION_NAME,
            SEARCH_QUERY,
            SEARCH_LIMIT,
            filter_expr=SEALED_ID_FILTER,
        )
        _assert_search_hits("MinHash sealed search", sealed_hits)
        _assert_ids_in_range(sealed_hits, 0, SEALED_ROWS, "sealed")

        print("[STEP5.4.2] verify growing rows are searchable")
        growing_hits = _search_until_ready(
            client,
            COLLECTION_NAME,
            SEARCH_QUERY,
            SEARCH_LIMIT,
            filter_expr=GROWING_ID_FILTER,
        )
        _assert_search_hits("MinHash growing search", growing_hits)
        _assert_ids_in_range(growing_hits, SEALED_ROWS, TOTAL_ROWS, "growing")

        print("[STEP5.5] flush growing rows")
        client.flush(COLLECTION_NAME)

        print(
            f"[STEP5.6] verify {POST_FLUSH_SEARCH_ROUNDS} post-flush searches "
            f"with interval={POST_FLUSH_SEARCH_INTERVAL}s"
        )
        _assert_repeated_search_success(client, COLLECTION_NAME)

        print("[STEP5.7] verify post-flush rows remain searchable")
        post_flush_hits = _search_until_ready(
            client,
            COLLECTION_NAME,
            SEARCH_QUERY,
            SEARCH_LIMIT,
            filter_expr=GROWING_ID_FILTER,
        )
        _assert_search_hits("MinHash post-flush growing search", post_flush_hits)
        _assert_ids_in_range(post_flush_hits, SEALED_ROWS, TOTAL_ROWS, "post-flush growing")

        print(f"Final collection stats: {client.get_collection_stats(COLLECTION_NAME)}")
        print(f"Final collection schema: {client.describe_collection(COLLECTION_NAME)}")
        print("PASS: add_function_field MinHash search flow verified")
    finally:
        client.close()


def main() -> None:
    run_add_function_field_minhash_search()


if __name__ == "__main__":
    main()
