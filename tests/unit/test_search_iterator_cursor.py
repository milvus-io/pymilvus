from copy import deepcopy
from unittest.mock import Mock

import pytest
from pymilvus.client.constants import (
    GUARANTEE_TIMESTAMP,
    ITER_SEARCH_BATCH_SIZE_KEY,
    ITER_SEARCH_CURSOR_VERSION_KEY,
    ITER_SEARCH_ID_KEY,
    ITER_SEARCH_LAST_BOUND_KEY,
    ITER_SEARCH_LAST_PK_KEY,
    ITER_SEARCH_LAST_PK_TYPE_KEY,
)
from pymilvus.client.iterator import SearchIteratorV2
from pymilvus.client.prepare import Prepare
from pymilvus.client.search_result import SearchResult
from pymilvus.client.types import DataType
from pymilvus.exceptions import MilvusException, ParamError, ServerVersionIncompatibleException
from pymilvus.grpc_gen import common_pb2, schema_pb2


def page(
    ids,
    *,
    version="2",
    ts=100,
    token="token",  # noqa: S107
    bound=0.25,
    metadata=None,
    round_decimal=-1,
):
    string_pk = bool(ids) and isinstance(ids[0], str)
    raw_ids = (
        schema_pb2.IDs(str_id=schema_pb2.StringArray(data=ids))
        if string_pk
        else schema_pb2.IDs(int_id=schema_pb2.LongArray(data=ids))
    )
    data = schema_pb2.SearchResultData(
        num_queries=1, ids=raw_ids, topks=[len(ids)], scores=[bound] * len(ids)
    )
    data.search_iterator_v2_results.token = token
    data.search_iterator_v2_results.last_bound = bound
    extra = {} if version is None else {ITER_SEARCH_CURSOR_VERSION_KEY: version}
    if version == "2" and ids:
        extra[ITER_SEARCH_LAST_PK_TYPE_KEY] = "varchar" if string_pk else "int64"
        extra[ITER_SEARCH_LAST_PK_KEY] = str(ids[-1])
    if metadata is not None:
        extra.update(metadata)
    return SearchResult(data, round_decimal, common_pb2.Status(extra_info=extra), session_ts=ts)


def test_duplicate_pk_with_different_scores_does_not_consume_limit():
    it, handler = iterator(
        [page([1], bound=0), page([1], bound=1), page([2], bound=2), page([])],
        batch_size=1,
        limit=2,
    )

    assert it.next().ids() == [1]
    assert it.next().ids() == [2]
    assert handler.search.call_count == 3
    assert it.next() is None


def test_empty_varchar_pk_is_deduplicated_across_scores():
    it, handler = iterator(
        [page([""], bound=0), page([""], bound=1), page(["other"], bound=2)],
        pk_type=DataType.VARCHAR,
        batch_size=1,
        limit=2,
    )

    assert it.next().ids() == [""]
    assert it.next().ids() == ["other"]
    assert handler.search.call_count == 3
    assert it.next() is None


def test_filtered_cache_deduplicates_pending_and_emitted_pks():
    it, handler = iterator(
        [page([1, 2, 3]), page([3, 1, 4], bound=1), page([])],
        external_filter_func=lambda hits: hits,
        limit=4,
    )

    assert it.next().ids() == [1, 2]
    assert it.next().ids() == [3, 4]
    assert handler.search.call_count == 2
    assert it.next() is None


def test_filter_failure_does_not_commit_seen_pks_or_cursor():
    attempts = 0

    def fail_once(hits):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry this page")
        return hits

    it, handler = iterator(
        [page([1, 1]), page([2], bound=1), page([])],
        external_filter_func=fail_once,
        limit=2,
    )

    with pytest.raises(RuntimeError, match="retry this page"):
        it.next()
    assert handler.search.call_count == 1
    assert it.next().ids() == [1, 2]
    assert handler.search.call_count == 2
    assert it.next() is None


def iterator(replies, *, pk_type=DataType.INT64, pk_cursor=True, **kwargs):
    handler = Mock()
    handler.describe_collection.return_value = {
        "collection_id": 17,
        "fields": [{"name": "pk", "type": pk_type, "is_primary": True}],
    }
    handler.search.side_effect = replies
    rpc_options = dict(kwargs.pop("rpc_options", {}))
    if pk_cursor:
        rpc_options.setdefault(ITER_SEARCH_CURSOR_VERSION_KEY, "2")
    it = SearchIteratorV2(
        handler=handler,
        context=None,
        collection_name="c",
        data=[[1.0, 2.0]],
        batch_size=kwargs.pop("batch_size", 2),
        rpc_options=rpc_options,
        **kwargs,
    )
    return it, handler


def test_first_real_page_is_returned_once_and_cursor_sent_on_second_rpc():
    large = 2**53 + 1
    it, handler = iterator([page([large, large + 1]), page([large + 2]), page([])])
    assert handler.search.call_count == 1
    assert handler.search.call_args.kwargs["limit"] == 2
    assert handler.search.call_args.kwargs[ITER_SEARCH_BATCH_SIZE_KEY] == 2
    assert handler.search.call_args.kwargs[ITER_SEARCH_CURSOR_VERSION_KEY] == "2"
    assert it.next().ids() == [large, large + 1]
    assert handler.search.call_count == 1
    assert it.next().ids() == [large + 2]
    request = handler.search.call_args.kwargs
    assert request[ITER_SEARCH_LAST_PK_KEY] == str(large + 1)
    assert request[ITER_SEARCH_LAST_PK_TYPE_KEY] == "int64"
    assert request[ITER_SEARCH_ID_KEY] == "token"
    assert request[GUARANTEE_TIMESTAMP] == 100
    assert len(it.next()) == len(it.next()) == 0
    assert handler.search.call_count == 3


@pytest.mark.parametrize("pk", [-(2**63), 2**63 - 1, 2**53 + 1])
def test_int64_cursor_preserves_full_precision(pk):
    it, _ = iterator([page([pk])])
    assert it.next().ids() == [pk]
    assert it._params[ITER_SEARCH_LAST_PK_KEY] == str(pk)


@pytest.mark.parametrize("pk", ["", 'a"\\\n雪', "零\x00尾"])
def test_varchar_cursor_is_not_json_encoded(pk):
    it, _ = iterator([page([pk])], pk_type=DataType.VARCHAR)
    assert it.next().ids() == [pk]
    assert it._params[ITER_SEARCH_LAST_PK_KEY] == pk


@pytest.mark.parametrize(
    "metadata",
    [
        {ITER_SEARCH_LAST_PK_KEY: "9223372036854775808"},
        {ITER_SEARCH_LAST_PK_KEY: "1.0"},
        {ITER_SEARCH_LAST_PK_KEY: "+1"},
        {ITER_SEARCH_LAST_PK_KEY: "01"},
        {ITER_SEARCH_LAST_PK_KEY: "2"},
        {ITER_SEARCH_LAST_PK_TYPE_KEY: "varchar"},
    ],
)
def test_invalid_first_cursor_is_not_treated_as_unsupported_server(metadata):
    with pytest.raises(MilvusException) as exc:
        iterator([page([1], metadata=metadata)])
    assert not isinstance(exc.value, ServerVersionIncompatibleException)


def test_missing_cursor_and_schema_type_mismatch():
    missing = page([1])
    missing._status_extra_info.pop(ITER_SEARCH_LAST_PK_KEY)
    with pytest.raises(MilvusException):
        iterator([missing])
    with pytest.raises(MilvusException):
        iterator([page([1])], pk_type=DataType.VARCHAR)


@pytest.mark.parametrize(
    "bad_page", [page([2], version=None), page([2], version="3"), page([2], token="changed")]
)
def test_negotiated_protocol_cannot_change_and_failed_response_does_not_advance(bad_page):
    it, _ = iterator([page([1]), bad_page, page([2])])
    assert it.next().ids() == [1]
    before = deepcopy(it._params)
    with pytest.raises(MilvusException):
        it.next()
    assert it._params == before
    assert it.next().ids() == [2]


def test_rpc_failure_and_malformed_cursor_leave_pagination_unchanged():
    bad = page([2], metadata={ITER_SEARCH_LAST_PK_KEY: "3"})
    it, _ = iterator([page([1]), RuntimeError("rpc"), bad, page([2])])
    it.next()
    before = deepcopy(it._params)
    with pytest.raises(RuntimeError):
        it.next()
    assert it._params == before
    with pytest.raises(MilvusException):
        it.next()
    assert it._params == before
    assert it.next().ids() == [2]


def test_callback_failure_retries_same_unmodified_pending_page():
    calls = 0

    def callback(hits):
        nonlocal calls
        calls += 1
        if calls == 1:
            hits.clear()
            raise RuntimeError("callback")
        return hits

    it, handler = iterator([page([1, 2])], external_filter_func=callback)
    before = deepcopy(it._params)
    with pytest.raises(RuntimeError):
        it.next()
    assert it._params == before
    assert it.next().ids() == [1, 2]
    assert handler.search.call_count == 1


def test_filtered_cache_surplus_uses_no_unnecessary_rpc_and_limit_counts_output():
    it, handler = iterator(
        [page([1, 2, 3]), page([4, 5, 6]), page([])],
        external_filter_func=lambda hits: hits,
        batch_size=2,
        limit=5,
    )
    assert it.next().ids() == [1, 2]
    assert it.next().ids() == [3, 4]
    assert it.next().ids() == [5]
    assert it.next() is None
    assert handler.search.call_count == 2


def test_fully_filtered_first_page_advances_raw_cursor_without_losing_following_page():
    it, handler = iterator(
        [page([1, 2]), page([3, 4])],
        external_filter_func=lambda hits: [h for h in hits if h["id"] > 2],
    )
    assert it.next().ids() == [3, 4]
    assert handler.search.call_args.kwargs[ITER_SEARCH_LAST_PK_KEY] == "2"


def test_snapshot_is_required_for_pk_mode_and_explicit_snapshot_is_preserved():
    with pytest.raises(MilvusException):
        iterator([page([1], ts=0)])
    it, handler = iterator([page([1], ts=0), page([2])], rpc_options={GUARANTEE_TIMESTAMP: 42})
    it.next()
    it.next()
    assert handler.search.call_args.kwargs[GUARANTEE_TIMESTAMP] == 42


def test_legacy_v2_remains_distance_only_and_initial_page_is_reused():
    it, handler = iterator([page([1, 2], version=None), page([3], version=None)])
    assert it.next().ids() == [1, 2]
    assert it.next().ids() == [3]
    assert handler.search.call_count == 2
    assert ITER_SEARCH_CURSOR_VERSION_KEY not in handler.search.call_args.kwargs
    assert ITER_SEARCH_LAST_PK_KEY not in handler.search.call_args.kwargs


def test_legacy_iterator_cannot_silently_upgrade_halfway_through():
    it, handler = iterator([page([1], version=None), page([2])])
    it.next()
    before = deepcopy(it._params)
    with pytest.raises(MilvusException):
        it.next()
    assert it._params == before
    assert ITER_SEARCH_CURSOR_VERSION_KEY not in handler.search.call_args.kwargs


def test_manual_legacy_cursor_preserves_existing_bound_token_and_explicit_snapshot():
    supplied = {
        ITER_SEARCH_LAST_BOUND_KEY: 0.5,
        ITER_SEARCH_ID_KEY: "token",
        GUARANTEE_TIMESTAMP: 42,
    }
    it, handler = iterator(
        [page([2], version=None), page([3], version=None)], rpc_options=supplied, pk_cursor=False
    )
    first = handler.search.call_args.kwargs
    assert first[ITER_SEARCH_LAST_BOUND_KEY] == 0.5
    assert first[ITER_SEARCH_ID_KEY] == "token"
    assert first[GUARANTEE_TIMESTAMP] == 42
    assert ITER_SEARCH_CURSOR_VERSION_KEY not in first
    assert it.next().ids() == [2]
    assert it.next().ids() == [3]
    assert handler.search.call_args.kwargs[GUARANTEE_TIMESTAMP] == 42
    assert ITER_SEARCH_CURSOR_VERSION_KEY not in handler.search.call_args.kwargs
    assert supplied[ITER_SEARCH_LAST_BOUND_KEY] == 0.5


def test_manual_legacy_cursor_rejects_unsolicited_pk_mode_on_first_response():
    with pytest.raises(MilvusException):
        iterator(
            [page([2])], rpc_options={ITER_SEARCH_LAST_BOUND_KEY: 0.5, ITER_SEARCH_ID_KEY: "token"}
        )


def test_user_parameters_cannot_seed_a_pk_cursor_for_a_new_iterator():
    supplied = {
        ITER_SEARCH_CURSOR_VERSION_KEY: "2",
        ITER_SEARCH_LAST_PK_TYPE_KEY: "int64",
        ITER_SEARCH_LAST_PK_KEY: "99",
    }
    search_params = {"params": dict(supplied), **supplied}
    it, handler = iterator(
        [page([1], version=None), page([2], version=None)],
        rpc_options=supplied,
        search_params=search_params,
    )
    initial_request = handler.search.call_args.kwargs
    assert initial_request[ITER_SEARCH_CURSOR_VERSION_KEY] == "2"
    assert ITER_SEARCH_LAST_PK_KEY not in initial_request
    assert initial_request["param"] == {"params": {}}
    assert search_params["params"] == supplied
    it.next()
    it.next()
    assert ITER_SEARCH_LAST_PK_KEY not in handler.search.call_args.kwargs


def test_advertised_pk_mode_without_token_is_protocol_error_not_v1_fallback():
    with pytest.raises(MilvusException) as exc:
        iterator([page([1], token="")])
    assert not isinstance(exc.value, ServerVersionIncompatibleException)


def test_missing_v2_token_preserves_fallback_but_rpc_failure_does_not():
    with pytest.raises(ServerVersionIncompatibleException):
        iterator([page([1], version=None, token="")])
    with pytest.raises(RuntimeError):
        iterator([RuntimeError("permission or transport failure")])
    # Old servers can omit the result shape too: the absent V2 token is still
    # the compatibility signal, rather than a malformed negotiated PK page.
    absent_v2_result = page([], version=None, token="")
    absent_v2_result.clear()
    with pytest.raises(ServerVersionIncompatibleException):
        iterator([absent_v2_result])


def test_empty_initial_page_exhausts_without_another_rpc():
    it, handler = iterator([page([])])
    assert len(it.next()) == len(it.next()) == 0
    assert handler.search.call_count == 1


def test_rounding_does_not_change_cursor_bound_and_metadata_stays_private():
    response = page([1], bound=0.123456, round_decimal=2)
    it, _ = iterator([response])
    assert it.next()[0]["distance"] == 0.12
    assert (
        it._params[ITER_SEARCH_LAST_BOUND_KEY]
        == response.get_search_iterator_v2_results_info().last_bound
    )
    assert response.extra == {}


def test_cursor_kwargs_are_serialized_losslessly():
    request = Prepare.search_requests_with_expr(
        collection_name="c",
        anns_field="v",
        param={},
        limit=2,
        data=[[1.0, 2.0]],
        **{
            ITER_SEARCH_CURSOR_VERSION_KEY: "2",
            ITER_SEARCH_LAST_PK_TYPE_KEY: "varchar",
            ITER_SEARCH_LAST_PK_KEY: '雪"\\',
        },
    )
    params = {item.key: item.value for item in request.search_params}
    assert params[ITER_SEARCH_CURSOR_VERSION_KEY] == "2"
    assert params[ITER_SEARCH_LAST_PK_TYPE_KEY] == "varchar"
    assert params[ITER_SEARCH_LAST_PK_KEY] == '雪"\\'


def test_zero_batch_is_rejected_before_network_calls():
    with pytest.raises(ParamError):
        iterator([], batch_size=0)


def test_default_requests_keep_legacy_distance_cost_without_opt_in():
    it, handler = iterator([page([1], version=None), page([2], version=None)], pk_cursor=False)
    assert ITER_SEARCH_CURSOR_VERSION_KEY not in handler.search.call_args.kwargs
    it.next()
    it.next()
    assert ITER_SEARCH_CURSOR_VERSION_KEY not in handler.search.call_args.kwargs


@pytest.mark.parametrize("version", ["3", "0", "legacy"])
def test_unsupported_requested_versions_fail_before_describe_or_search(version):
    with pytest.raises(ParamError):
        iterator([], rpc_options={ITER_SEARCH_CURSOR_VERSION_KEY: version})


def test_unsolicited_pk_mode_is_protocol_error_when_default_is_legacy():
    with pytest.raises(MilvusException):
        iterator([page([1])], pk_cursor=False)


@pytest.mark.parametrize(
    "shape",
    [
        (2, (1,), 1, 1, 0.25, True),
        (1, (), 1, 1, 0.25, True),
        (1, (2,), 3, 3, 0.25, True),
        (1, (1,), 1, 2, 0.25, True),
        (1, (1,), 1, 1, 0.5, True),
        (1, (1,), 1, 1, float("nan"), False),
    ],
)
def test_malformed_raw_shapes_or_scores_do_not_advance_cursor_and_can_retry(shape):
    bad = page([2])
    bad._iterator_result_shape = shape
    it, _ = iterator([page([1]), bad, page([2])])
    it.next()
    before = deepcopy(it._params)
    with pytest.raises(MilvusException):
        it.next()
    assert it._params == before
    assert it.next().ids() == [2]
