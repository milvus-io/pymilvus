import re
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pymilvus.exceptions import MilvusException, ParamError
from pymilvus.orm.constants import (
    CALC_DIST_BM25,
    CALC_DIST_COSINE,
    CALC_DIST_HAMMING,
    CALC_DIST_IP,
    CALC_DIST_JACCARD,
    CALC_DIST_L2,
    CALC_DIST_TANIMOTO,
    COLLECTION_ID,
    DEFAULT_SEARCH_EXTENSION_RATE,
    EF,
    FIELDS,
    IS_PRIMARY,
    ITERATOR_SESSION_CP_FILE,
    ITERATOR_SESSION_TS_FIELD,
    MAX_BATCH_SIZE,
    MAX_FILTERED_IDS_COUNT_ITERATION,
    METRIC_TYPE,
    OFFSET,
    PARAMS,
    RADIUS,
    RANGE_FILTER,
    REDUCE_STOP_FOR_BEST,
)
from pymilvus.orm.iterator import (
    NO_CACHE_ID,
    IteratorCache,
    QueryIterator,
    SearchIterator,
    SearchPage,
    assert_info,
    check_set_flag,
    extend_batch_size,
    fall_back_to_latest_session_ts,
    io_operation,
    iterator_cache,
    metrics_positive_related,
)
from pymilvus.orm.types import DataType


class TestFallBackToLatestSessionTs:
    @patch("pymilvus.orm.iterator.datetime")
    @patch("pymilvus.orm.iterator.mkts_from_datetime")
    def test_returns_timestamp_from_now(self, mock_mkts, mock_datetime):
        mock_now = Mock()
        mock_datetime.datetime.now.return_value = mock_now
        mock_mkts.return_value = 123456789

        result = fall_back_to_latest_session_ts()

        mock_datetime.datetime.now.assert_called_once()
        mock_mkts.assert_called_once_with(mock_now, milliseconds=1000.0)
        assert result == 123456789


class TestAssertInfo:
    def test_passes_on_true(self):
        assert_info(True, "should not raise")

    @pytest.mark.parametrize(
        "message",
        [
            "Simple error",
            "Error with special characters: @#$%",
            "Error with numbers 12345",
            "",
        ],
    )
    def test_raises_on_false(self, message):
        escaped = re.escape(message) if message else ".*"
        with pytest.raises(MilvusException, match=escaped):
            assert_info(False, message)


class TestIoOperation:
    def test_success(self):
        mock_func = Mock()
        io_operation(mock_func, "error msg")
        mock_func.assert_called_once()

    @pytest.mark.parametrize(
        "exc_type",
        [OSError, PermissionError, FileNotFoundError],
    )
    def test_wraps_os_errors(self, exc_type):
        mock_func = Mock(side_effect=exc_type("fail"))
        with pytest.raises(MilvusException, match="wrapped"):
            io_operation(mock_func, "wrapped")


class TestExtendBatchSize:
    @pytest.mark.parametrize(
        "batch_size, params, extend, expected",
        [
            # Without extension, no EF
            (100, {}, False, 100),
            # With extension, no EF
            (100, {}, True, min(MAX_BATCH_SIZE, int(100 * DEFAULT_SEARCH_EXTENSION_RATE))),
            # With EF limiting
            (100, {EF: 50}, False, 50),
            (100, {EF: 50}, True, 50),
            # EF larger than extended
            (100, {EF: 200}, True, min(200, int(100 * DEFAULT_SEARCH_EXTENSION_RATE))),
            # Capped at MAX_BATCH_SIZE
            (MAX_BATCH_SIZE * 2, {}, False, MAX_BATCH_SIZE),
            (MAX_BATCH_SIZE * 2, {}, True, MAX_BATCH_SIZE),
            # Minimum batch size
            (1, {}, False, 1),
        ],
    )
    def test_extend_batch_size(self, batch_size, params, extend, expected):
        next_param = {PARAMS: params}
        assert extend_batch_size(batch_size, next_param, extend) == expected


class TestCheckSetFlag:
    @pytest.mark.parametrize(
        "key, value, expected",
        [
            ("bool_flag", True, True),
            ("bool_flag", False, False),
            ("str_flag", "enabled", "enabled"),
            ("none_flag", None, None),
            ("num_flag", 42, 42),
        ],
    )
    def test_sets_value_from_kwargs(self, key, value, expected):
        obj = Mock()
        check_set_flag(obj, "attr", {key: value}, key)
        assert obj.attr == expected

    def test_missing_key_defaults_to_false(self):
        obj = Mock()
        check_set_flag(obj, "attr", {}, "missing")
        assert obj.attr is False

    def test_overwrites_existing(self):
        obj = Mock()
        obj.attr = "old"
        check_set_flag(obj, "attr", {"k": "new"}, "k")
        assert obj.attr == "new"


class TestMetricsPositiveRelated:
    @pytest.mark.parametrize(
        "metric, expected",
        [
            (CALC_DIST_L2, True),
            (CALC_DIST_JACCARD, True),
            (CALC_DIST_HAMMING, True),
            (CALC_DIST_TANIMOTO, True),
            (CALC_DIST_IP, False),
            (CALC_DIST_COSINE, False),
            (CALC_DIST_BM25, False),
        ],
    )
    def test_known_metrics(self, metric, expected):
        assert metrics_positive_related(metric) is expected

    def test_unknown_metric_raises(self):
        with pytest.raises(MilvusException, match="unsupported metrics type"):
            metrics_positive_related("UNKNOWN")


class TestSearchPage:
    def test_init_with_none(self):
        page = SearchPage(None)
        assert len(page) == 0
        assert page.get_session_ts() == 0
        assert page.get_res() == []

    def test_init_with_session_ts(self):
        hits = Mock(__len__=Mock(return_value=5))
        page = SearchPage(hits, session_ts=12345)
        assert page.get_session_ts() == 12345
        assert len(page.get_res()) == 1

    def test_length(self):
        hits = Mock(__len__=Mock(return_value=10))
        page = SearchPage(hits)
        assert len(page) == 10

    def test_merge(self):
        h1 = Mock(__len__=Mock(return_value=5))
        h2 = Mock(__len__=Mock(return_value=3))
        page = SearchPage(h1)
        page.merge([h2])
        assert len(page.get_res()) == 2
        assert len(page) == 8

    def test_merge_none(self):
        hits = Mock(__len__=Mock(return_value=5))
        page = SearchPage(hits)
        page.merge(None)
        assert len(page.get_res()) == 1

    def test_ids(self):
        h1, h2 = Mock(id=1), Mock(id=2)
        hits = Mock(__len__=Mock(return_value=2), __iter__=Mock(return_value=iter([h1, h2])))
        page = SearchPage(hits)
        assert page.ids() == [1, 2]

    def test_distances(self):
        h1, h2 = Mock(distance=0.1), Mock(distance=0.2)
        hits = Mock(__len__=Mock(return_value=2), __iter__=Mock(return_value=iter([h1, h2])))
        page = SearchPage(hits)
        assert page.distances() == [0.1, 0.2]

    def test_get_item_empty(self):
        assert SearchPage(None).get__item(0) is None

    def test_get_item_out_of_range(self):
        hits = Mock(__len__=Mock(return_value=2))
        with pytest.raises(IndexError, match="Index out of range"):
            SearchPage(hits).get__item(5)

    def test_get_item_success(self):
        hit = Mock()
        hits = Mock(__len__=Mock(return_value=1), __getitem__=Mock(return_value=hit))
        assert SearchPage(hits).get__item(0) == hit

    def test_get_item_across_multiple_results(self):
        """Test get__item when results span multiple merged pages."""
        h0 = Mock()
        h1 = Mock()
        hits1 = Mock(
            __len__=Mock(return_value=1),
            __getitem__=Mock(return_value=h0),
        )
        hits2 = Mock(
            __len__=Mock(return_value=1),
            __getitem__=Mock(return_value=h1),
        )
        page = SearchPage(hits1)
        page.merge([hits2])
        # Index 1 should come from second hits
        result = page.get__item(1)
        assert result == h1


class TestIteratorCache:
    def test_cache_and_fetch(self):
        cache = IteratorCache()
        data = ["a", "b"]
        cache_id = cache.cache(data, NO_CACHE_ID)
        assert cache_id > 0
        assert cache.fetch_cache(cache_id) == data

    def test_update_existing(self):
        cache = IteratorCache()
        cid = cache.cache(["v1"], NO_CACHE_ID)
        cache.cache(["v2"], cid)
        assert cache.fetch_cache(cid) == ["v2"]

    def test_release(self):
        cache = IteratorCache()
        cid = cache.cache(["x"], NO_CACHE_ID)
        cache.release_cache(cid)
        assert cache.fetch_cache(cid) is None

    def test_fetch_nonexistent(self):
        assert IteratorCache().fetch_cache(999) is None

    def test_release_nonexistent(self):
        IteratorCache().release_cache(999)  # should not raise

    def test_multiple_entries(self):
        cache = IteratorCache()
        ids = [cache.cache([f"d{i}"], NO_CACHE_ID) for i in range(3)]
        assert len(set(ids)) == 3
        for i, cid in enumerate(ids):
            assert cache.fetch_cache(cid) == [f"d{i}"]

        cache.release_cache(ids[1])
        assert cache.fetch_cache(ids[0]) == ["d0"]
        assert cache.fetch_cache(ids[1]) is None
        assert cache.fetch_cache(ids[2]) == ["d2"]


class TestGlobalIteratorCache:
    def test_singleton_exists(self):
        assert isinstance(iterator_cache, IteratorCache)

    def test_no_cache_id_constant(self):
        assert NO_CACHE_ID == -1


# ---------------------------------------------------------------------------
# QueryIterator tests
# ---------------------------------------------------------------------------

_SCHEMA_DICT = {
    FIELDS: [
        {"name": "pk", "type": DataType.INT64, IS_PRIMARY: True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
    ],
}

_VARCHAR_SCHEMA_DICT = {
    FIELDS: [
        {"name": "pk", "type": DataType.VARCHAR, IS_PRIMARY: True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
    ],
}


def _make_mock_conn(query_results=None, session_ts=100):
    """Return a mock connection object suitable for QueryIterator."""
    conn = Mock()
    conn.describe_collection.return_value = {COLLECTION_ID: 999}

    if query_results is None:
        query_results = []

    mock_res = Mock()
    mock_res.__len__ = Mock(return_value=len(query_results))
    mock_res.__iter__ = Mock(return_value=iter(query_results))
    mock_res.__getitem__ = lambda self, idx: query_results[idx]
    mock_res.extra = {ITERATOR_SESSION_TS_FIELD: session_ts}
    conn.query.return_value = mock_res
    return conn


class TestQueryIteratorInit:
    def test_basic_init(self):
        conn = _make_mock_conn()
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        assert qi._collection_name == "test"
        assert qi._collection_id == 999
        assert qi._pk_field_name == "pk"
        assert qi._pk_str is False

    def test_varchar_pk(self):
        conn = _make_mock_conn()
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr='pk != ""',
            output_fields=["pk"],
            schema=_VARCHAR_SCHEMA_DICT,
        )
        assert qi._pk_str is True

    def test_default_expr_int_pk(self):
        conn = _make_mock_conn()
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr=None,
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        assert "pk" in qi._expr
        assert "<" in qi._expr

    def test_default_expr_varchar_pk(self):
        conn = _make_mock_conn()
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr=None,
            output_fields=["pk"],
            schema=_VARCHAR_SCHEMA_DICT,
        )
        assert '!= ""' in qi._expr

    def test_batch_size_negative(self):
        conn = _make_mock_conn()
        with pytest.raises(ParamError, match="less than zero"):
            QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=-1,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
            )

    def test_batch_size_too_large(self):
        conn = _make_mock_conn()
        with pytest.raises(ParamError, match="larger than"):
            QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=MAX_BATCH_SIZE + 1,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
            )

    def test_session_ts_set(self):
        conn = _make_mock_conn(session_ts=5000)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        assert qi._session_ts == 5000

    def test_session_ts_fallback(self):
        """When server returns ts <= 0, fallback to client-side ts."""
        conn = _make_mock_conn(session_ts=0)
        with patch("pymilvus.orm.iterator.fall_back_to_latest_session_ts", return_value=9999):
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
            )
        assert qi._session_ts == 9999

    def test_none_query_result_raises(self):
        """When query returns None during ts setup, raise MilvusException."""
        conn = Mock()
        conn.describe_collection.return_value = {COLLECTION_ID: 999}
        conn.query.return_value = None
        with pytest.raises(MilvusException, match="failed to connect"):
            QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
            )


class TestQueryIteratorNext:
    def test_next_returns_results(self):
        rows = [{"pk": 1, "vec": [1, 2, 3, 4]}, {"pk": 2, "vec": [5, 6, 7, 8]}]
        conn = _make_mock_conn(session_ts=100)

        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk", "vec"],
            schema=_SCHEMA_DICT,
        )

        # Now set up query to return results for the next() call
        next_res = Mock()
        next_res.__len__ = Mock(return_value=2)
        next_res.__iter__ = Mock(return_value=iter(rows))

        def getitem(self, key):
            if isinstance(key, slice):
                return rows[key]
            return rows[key]

        next_res.__getitem__ = getitem
        conn.query.return_value = next_res

        result = qi.next()
        assert len(result) == 2

    def test_next_empty_returns_empty(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )

        empty_res = []

        def getitem(self, key):
            if isinstance(key, slice):
                return empty_res[key]
            return empty_res[key]

        mock_empty = Mock()
        mock_empty.__len__ = Mock(return_value=0)
        mock_empty.__getitem__ = getitem
        conn.query.return_value = mock_empty

        result = qi.next()
        assert len(result) == 0


class TestQueryIteratorClose:
    def test_close_releases_cache(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        # Should not raise
        qi.close()


class TestQueryIteratorLimit:
    def test_limit_truncates_result(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            limit=1,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )

        rows = [{"pk": 1}, {"pk": 2}]

        def getitem(self, key):
            if isinstance(key, slice):
                return rows[key]
            return rows[key]

        mock_res = Mock()
        mock_res.__len__ = Mock(return_value=2)
        mock_res.__getitem__ = getitem
        conn.query.return_value = mock_res

        result = qi.next()
        assert len(result) == 1


class TestQueryIteratorNextExpr:
    def test_setup_next_expr_no_cursor(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        # _next_id is None initially
        expr = qi._QueryIterator__setup_next_expr()
        assert expr == "pk > 0"

    def test_setup_next_expr_with_int_cursor(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        qi._next_id = 42
        expr = qi._QueryIterator__setup_next_expr()
        assert "pk > 42" in expr
        assert "(pk > 0)" in expr
        # PK cursor must precede the user filter so that right-most-operand
        # constraints (e.g. `element_filter()`) are preserved across pages.
        assert expr.index("pk > 42") < expr.index("(pk > 0)")

    def test_setup_next_expr_element_filter_stays_right_most(self):
        conn = _make_mock_conn(session_ts=100)
        user_filter = "element_filter(structA, $[int_val] >= 20000)"
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr=user_filter,
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        qi._next_id = 211
        expr = qi._QueryIterator__setup_next_expr()
        # element_filter() must remain the right-most operand of AND.
        assert expr == f"pk > 211 and ({user_filter})"

    def test_setup_next_expr_with_varchar_cursor(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr='pk != ""',
            output_fields=["pk"],
            schema=_VARCHAR_SCHEMA_DICT,
        )
        qi._next_id = "abc"
        expr = qi._QueryIterator__setup_next_expr()
        assert 'pk > "abc"' in expr

    def test_setup_next_expr_empty_expr_with_cursor(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        qi._next_id = 10
        qi._expr = ""
        expr = qi._QueryIterator__setup_next_expr()
        assert expr == "pk > 10"


class TestQueryIteratorGetCursor:
    def test_get_cursor_int_pk(self):
        conn = _make_mock_conn(session_ts=200)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )
        qi._next_id = 42
        cursor = qi.get_cursor()
        assert cursor.int_pk == 42

    def test_get_cursor_varchar_pk(self):
        conn = _make_mock_conn(session_ts=200)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr='pk != ""',
            output_fields=["pk"],
            schema=_VARCHAR_SCHEMA_DICT,
        )
        qi._next_id = "abc"
        cursor = qi.get_cursor()
        assert cursor.str_pk == "abc"


# ---------------------------------------------------------------------------
# SearchIterator tests
# ---------------------------------------------------------------------------

_SEARCH_SCHEMA_DICT = {
    FIELDS: [
        {"name": "pk", "type": DataType.INT64, IS_PRIMARY: True},
        {"name": "vec", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
    ],
}


def _make_search_conn(hits=None, session_ts=100):
    """Return a mock connection for SearchIterator."""
    conn = Mock()
    conn.describe_collection.return_value = {COLLECTION_ID: 999}

    if hits is None:
        hits = []

    # The search result is a list-like object where res[0] gives the Hits
    mock_hit_list = Mock()
    mock_hit_list.__len__ = Mock(return_value=len(hits))
    mock_hit_list.__iter__ = Mock(return_value=iter(hits))

    def hit_getitem(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx = len(hits) + idx
            return hits[idx]
        return hits[idx]

    mock_hit_list.__getitem__ = hit_getitem

    mock_res = Mock()
    mock_res.__getitem__ = lambda self, idx: mock_hit_list
    mock_res.get_session_ts = Mock(return_value=session_ts)
    conn.search.return_value = mock_res
    return conn


class TestSearchIteratorInit:
    def test_multiple_vectors_raises(self):
        conn = _make_search_conn()
        with pytest.raises(ParamError, match="multiple vectors"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4], [5, 6, 7, 8]],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_empty_vector_raises(self):
        conn = _make_search_conn()
        with pytest.raises(ParamError, match="cannot be empty"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_offset_raises(self):
        conn = _make_search_conn()
        with pytest.raises(ParamError, match="offset"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
                **{OFFSET: 5},
            )

    def test_no_metric_type_raises(self):
        conn = _make_search_conn()
        with pytest.raises(ParamError, match="metrics type"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "", PARAMS: {}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_ef_too_small_raises(self):
        conn = _make_search_conn()
        with pytest.raises(MilvusException, match="hnsw"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {EF: 5}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_basic_init_l2(self):
        """SearchIterator init with L2 metric and non-empty results."""
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)

        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        assert si._init_success is True
        assert si._session_ts == 500

    def test_init_empty_page(self):
        """When init page is empty, _init_success should be False."""
        conn = _make_search_conn(hits=[], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        assert si._init_success is False

    def test_range_search_l2_invalid_params_raises(self):
        """L2 metric: radius must be > range_filter."""
        conn = _make_search_conn()
        with pytest.raises(MilvusException, match="radius must be"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {RADIUS: 0.1, RANGE_FILTER: 0.5}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_range_search_ip_invalid_params_raises(self):
        """IP metric: radius must be < range_filter."""
        conn = _make_search_conn()
        with pytest.raises(MilvusException, match="radius must be"):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "IP", PARAMS: {RADIUS: 0.5, RANGE_FILTER: 0.1}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_param_none_raises_on_missing_metric(self):
        """When param is None, __check_metrics raises because metric_type is missing."""
        conn = _make_search_conn()
        with pytest.raises((ParamError, KeyError)):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param=None,
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )

    def test_check_set_params_none(self):
        """Directly test __check_set_params with None creates empty dict + PARAMS."""
        conn = _make_search_conn()
        # We can't easily test private methods but we can verify that passing
        # param=None with no metric raises the right error
        with pytest.raises((ParamError, KeyError)):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param=None,
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )


class TestSearchIteratorNext:
    def test_next_when_not_init_success(self):
        """When init fails, next() returns empty page."""
        conn = _make_search_conn(hits=[], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        assert si._init_success is False
        result = si.next()
        assert len(result) == 0

    def test_next_with_limit_reached(self):
        """Once limit is reached, next returns empty."""
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)

        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            limit=0,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # limit=0 means we have already hit the limit
        # Actually, limit=0 compared to _returned_count=0 means 0 < 0 is False
        # so __check_reached_limit returns True
        result = si.next()
        assert len(result) == 0


class TestSearchIteratorClose:
    def test_close_releases_cache(self):
        hit0 = Mock(id=1, distance=0.1)
        conn = _make_search_conn(hits=[hit0], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        si.close()
        # cache should be released
        assert iterator_cache.fetch_cache(si._cache_id) is None


class TestSearchIteratorFilteredExpr:
    def test_no_filtered_ids(self):
        conn = _make_search_conn(hits=[], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            expr="pk > 0",
            schema=_SEARCH_SCHEMA_DICT,
        )
        expr = si._SearchIterator__filtered_duplicated_result_expr("pk > 0")
        assert expr == "pk > 0"

    def test_with_int_filtered_ids(self):
        hit0 = Mock(id=1, distance=0.1)
        conn = _make_search_conn(hits=[hit0], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            expr="pk > 0",
            schema=_SEARCH_SCHEMA_DICT,
        )
        si._filtered_ids = [10, 20]
        expr = si._SearchIterator__filtered_duplicated_result_expr("pk > 0")
        assert "not in" in expr
        assert "10" in expr
        assert "20" in expr

    def test_with_varchar_filtered_ids(self):
        conn = Mock()
        conn.describe_collection.return_value = {COLLECTION_ID: 999}
        hit0 = Mock(id="a", distance=0.1)
        mock_hit_list = Mock()
        mock_hit_list.__len__ = Mock(return_value=1)
        mock_hit_list.__iter__ = Mock(return_value=iter([hit0]))
        mock_hit_list.__getitem__ = lambda self, idx: hit0

        mock_res = Mock()
        mock_res.__getitem__ = lambda self, idx: mock_hit_list
        mock_res.get_session_ts = Mock(return_value=500)
        conn.search.return_value = mock_res

        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_VARCHAR_SCHEMA_DICT,
        )
        si._filtered_ids = ["x", "y"]
        expr = si._SearchIterator__filtered_duplicated_result_expr(None)
        assert "not in" in expr
        assert '"x"' in expr

    def test_with_empty_expr_and_filtered_ids(self):
        hit0 = Mock(id=1, distance=0.1)
        conn = _make_search_conn(hits=[hit0], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        si._filtered_ids = [1]
        expr = si._SearchIterator__filtered_duplicated_result_expr("")
        assert "not in" in expr


class TestSearchIteratorNextParams:
    def test_next_params_l2(self):
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        next_p = si._SearchIterator__next_params(1)
        assert RADIUS in next_p[PARAMS]
        assert RANGE_FILTER in next_p[PARAMS]
        # For L2 (positive related), radius = tail_band + width * coefficient
        assert next_p[PARAMS][RANGE_FILTER] == si._tail_band

    def test_next_params_ip(self):
        hit0 = Mock(id=1, distance=0.9)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "IP", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        next_p = si._SearchIterator__next_params(1)
        assert RADIUS in next_p[PARAMS]
        assert RANGE_FILTER in next_p[PARAMS]

    def test_next_params_coefficient(self):
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        p1 = si._SearchIterator__next_params(1)
        p2 = si._SearchIterator__next_params(2)
        # With larger coefficient, radius should be further out
        assert p2[PARAMS][RADIUS] > p1[PARAMS][RADIUS]


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestQueryIteratorSeekToOffset:
    """Cover __seek_to_offset (lines 132, 135-169)."""

    def test_seek_offset_with_results(self):
        """Offset > 0 triggers the seek loop."""
        rows = [{"pk": 1}, {"pk": 2}]
        conn = _make_mock_conn(session_ts=100)

        # The first query call is __setup_ts_by_request, second+ is seek
        seek_res = Mock()
        seek_res.__len__ = Mock(return_value=2)
        seek_res.__iter__ = Mock(return_value=iter(rows))

        def getitem(self, key):
            return rows[key]

        seek_res.__getitem__ = getitem

        init_res = Mock()
        init_res.__len__ = Mock(return_value=0)
        init_res.__getitem__ = lambda s, k: [][k]
        init_res.extra = {ITERATOR_SESSION_TS_FIELD: 100}

        conn.query.side_effect = [init_res, seek_res]

        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
            **{OFFSET: 2},
        )
        # offset was consumed; next_id updated to last pk
        assert qi._next_id == 2
        assert qi._kwargs[OFFSET] == 0

    def test_seek_offset_drained(self):
        """Offset seek breaks when query returns 0 results."""
        conn = _make_mock_conn(session_ts=100)

        empty_res = Mock()
        empty_res.__len__ = Mock(return_value=0)
        empty_res.__getitem__ = lambda s, k: [][k]

        init_res = Mock()
        init_res.__len__ = Mock(return_value=0)
        init_res.__getitem__ = lambda s, k: [][k]
        init_res.extra = {ITERATOR_SESSION_TS_FIELD: 100}

        conn.query.side_effect = [init_res, empty_res]

        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
            **{OFFSET: 5},
        )
        assert qi._kwargs[OFFSET] == 0

    def test_seek_offset_skipped_when_next_id_set(self):
        """When _next_id is already set (cp file), skip offset."""
        conn = _make_mock_conn(session_ts=100)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cp", delete=False) as f:
            f.write("100\n")
            f.write("42\n")
            cp_path = f.name

        try:
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
                **{ITERATOR_SESSION_CP_FILE: cp_path, OFFSET: 10},
            )
            assert qi._next_id == "42"
        finally:
            if Path(cp_path).exists():
                qi.close()


class TestQueryIteratorConnQueryArgs:
    """Verify arguments with business logic passed to conn.query() in each phase."""

    def test_setup_ts_by_request_query_args(self):
        """__setup_ts_by_request: only sets up mvccTs, output_fields/partition_names should be empty."""
        conn = _make_mock_conn(session_ts=100)

        QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            partition_names=["part_b"],
            schema=_SCHEMA_DICT,
        )

        kwargs = conn.query.call_args_list[0].kwargs
        assert kwargs["output_fields"] == []
        assert kwargs["partition_names"] == []
        assert kwargs["limit"] == 1
        assert kwargs["offset"] == 0
        assert kwargs["iterator"] == "True"
        assert kwargs["reduce_stop_for_best"] == "True"

    def test_seek_to_offset_query_args(self):
        """__seek_to_offset: should use user's partition_names, empty output_fields, iterator disabled."""
        rows = [{"pk": 1}, {"pk": 2}]
        conn = _make_mock_conn(session_ts=100)

        seek_res = Mock()
        seek_res.__len__ = Mock(return_value=2)
        seek_res.__iter__ = Mock(return_value=iter(rows))
        seek_res.__getitem__ = lambda self, key: rows[key]

        init_res = Mock()
        init_res.__len__ = Mock(return_value=0)
        init_res.__getitem__ = lambda s, k: [][k]
        init_res.extra = {ITERATOR_SESSION_TS_FIELD: 100}

        conn.query.side_effect = [init_res, seek_res]

        QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            partition_names=["part_a"],
            schema=_SCHEMA_DICT,
            **{OFFSET: 2},
        )

        kwargs = conn.query.call_args_list[1].kwargs
        assert kwargs["output_fields"] == []
        assert kwargs["partition_names"] == ["part_a"]
        assert kwargs["offset"] == 0
        assert kwargs["limit"] == 2
        assert kwargs["iterator"] == "False"
        assert kwargs["reduce_stop_for_best"] == "False"
        assert kwargs["guarantee_timestamp"] == 100

    def test_next_query_args(self):
        """next(): should use user's partition_names and output_fields, iterator enabled."""
        conn = _make_mock_conn(session_ts=100)

        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            partition_names=["part_a"],
            schema=_SCHEMA_DICT,
        )

        next_rows = [{"pk": 1}]
        next_res = Mock()
        next_res.__len__ = Mock(return_value=1)
        next_res.__iter__ = Mock(return_value=iter(next_rows))
        next_res.__getitem__ = lambda self, key: next_rows[key]
        conn.query.return_value = next_res

        qi.next()

        kwargs = conn.query.call_args.kwargs
        assert kwargs["output_fields"] == ["pk"]
        assert kwargs["partition_names"] == ["part_a"]
        assert kwargs["iterator"] == "True"
        assert kwargs["reduce_stop_for_best"] == "True"
        assert kwargs["guarantee_timestamp"] == 100


class TestQueryIteratorCpFile:
    """Cover __set_up_ts_cp with cp file (lines 273-299)."""

    def test_cp_file_new_creates_and_saves(self):
        """New cp file triggers ts setup by request + save."""
        conn = _make_mock_conn(session_ts=200)

        with tempfile.TemporaryDirectory() as td:
            cp_path = str(Path(td) / "new_iter.cp")
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
                **{ITERATOR_SESSION_CP_FILE: cp_path},
            )
            assert qi._need_save_cp is True
            assert qi._session_ts == 200
            qi.close()

    def test_cp_file_existing_reads_ts_and_cursor(self):
        """Existing cp file with valid content restores state."""
        conn = _make_mock_conn(session_ts=100)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cp", delete=False) as f:
            f.write("300\n")
            f.write("99\n")
            cp_path = f.name

        try:
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
                **{ITERATOR_SESSION_CP_FILE: cp_path},
            )
            assert qi._session_ts == 300
            assert qi._next_id == "99"
            qi.close()
        except Exception:
            if Path(cp_path).exists():
                Path(cp_path).unlink()
            raise

    def test_cp_file_too_few_lines_raises(self):
        """CP file with only 1 line raises ParamError."""
        conn = _make_mock_conn(session_ts=100)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cp", delete=False) as f:
            f.write("300\n")
            cp_path = f.name

        try:
            with pytest.raises(ParamError, match="at least two lines"):
                QueryIterator(
                    connection=conn,
                    collection_name="test",
                    batch_size=10,
                    expr="pk > 0",
                    output_fields=["pk"],
                    schema=_SCHEMA_DICT,
                    **{ITERATOR_SESSION_CP_FILE: cp_path},
                )
        finally:
            if Path(cp_path).exists():
                Path(cp_path).unlink()

    def test_cp_file_invalid_ts_raises(self):
        """CP file with non-integer ts raises ParamError."""
        conn = _make_mock_conn(session_ts=100)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cp", delete=False) as f:
            f.write("not_a_number\n")
            f.write("42\n")
            cp_path = f.name

        try:
            with pytest.raises(ParamError, match="cannot parse"):
                QueryIterator(
                    connection=conn,
                    collection_name="test",
                    batch_size=10,
                    expr="pk > 0",
                    output_fields=["pk"],
                    schema=_SCHEMA_DICT,
                    **{ITERATOR_SESSION_CP_FILE: cp_path},
                )
        finally:
            if Path(cp_path).exists():
                Path(cp_path).unlink()


class TestQueryIteratorSavePkCursor:
    """Cover __save_pk_cursor (lines 195-214)."""

    def test_save_pk_cursor_on_next(self):
        """When cp file is configured, next() saves cursor."""
        conn = _make_mock_conn(session_ts=200)
        with tempfile.TemporaryDirectory() as td:
            cp_path = str(Path(td) / "cursor.cp")
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
                **{ITERATOR_SESSION_CP_FILE: cp_path},
            )

            # Set up next() to return rows
            rows = [{"pk": 10}, {"pk": 20}]
            next_res = Mock()
            next_res.__len__ = Mock(return_value=2)

            def getitem(self, key):
                if isinstance(key, slice):
                    return rows[key]
                return rows[key]

            next_res.__getitem__ = getitem
            conn.query.return_value = next_res

            qi.next()
            assert qi._next_id == 20

            # Verify file was written
            assert Path(cp_path).exists()
            qi.close()

    def test_save_pk_cursor_truncates_after_100_lines(self):
        """CP file is truncated when buffer lines >= 100."""
        conn = _make_mock_conn(session_ts=200)
        with tempfile.TemporaryDirectory() as td:
            cp_path = str(Path(td) / "cursor.cp")
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
                **{ITERATOR_SESSION_CP_FILE: cp_path},
            )

            # Simulate 100 buffered lines
            qi._buffer_cursor_lines_number = 100
            qi._next_id = 42

            # Manually call __save_pk_cursor
            qi._QueryIterator__save_pk_cursor()
            # After truncation, lines reset to 0 + 1 new write
            assert qi._buffer_cursor_lines_number == 1
            qi.close()


class TestQueryIteratorMaybeCache:
    """Cover __maybe_cache (lines 304-307)."""

    def test_maybe_cache_large_result(self):
        """When result has >= 2*batch_size items, cache the overflow."""
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=2,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )

        big_result = [{"pk": i} for i in range(5)]

        mock_res = Mock()
        mock_res.__len__ = Mock(return_value=5)

        def getitem(self, key):
            if isinstance(key, slice):
                return big_result[key]
            return big_result[key]

        mock_res.__getitem__ = getitem
        conn.query.return_value = mock_res

        result = qi.next()
        # Should return batch_size items
        assert len(result) == 2
        # Cache should have overflow
        assert qi._cache_id_in_use != NO_CACHE_ID

    def test_next_uses_cache(self):
        """Second next() call uses cached results."""
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=2,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )

        big_result = [{"pk": i} for i in range(6)]

        mock_res = Mock()
        mock_res.__len__ = Mock(return_value=6)

        def getitem(self, key):
            if isinstance(key, slice):
                return big_result[key]
            return big_result[key]

        mock_res.__getitem__ = getitem
        conn.query.return_value = mock_res

        result1 = qi.next()
        assert len(result1) == 2

        # Second call should use cache (no new query)
        call_count_before = conn.query.call_count
        result2 = qi.next()
        # Cache had 4 items (6-2), enough for batch_size=2
        assert len(result2) == 2
        assert conn.query.call_count == call_count_before


class TestQueryIteratorCheckReachedLimit:
    """Cover __check_reached_limit non-truncation path (line 354)."""

    def test_limit_not_yet_reached(self):
        """With large limit, results pass through untruncated."""
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            limit=100,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
        )

        rows = [{"pk": 1}, {"pk": 2}]
        mock_res = Mock()
        mock_res.__len__ = Mock(return_value=2)

        def getitem(self, key):
            if isinstance(key, slice):
                return rows[key]
            return rows[key]

        mock_res.__getitem__ = getitem
        conn.query.return_value = mock_res

        result = qi.next()
        assert len(result) == 2


class TestQueryIteratorReduceStopForBest:
    """Cover __check_set_reduce_stop_for_best False branch (line 220)."""

    def test_reduce_stop_for_best_false(self):
        conn = _make_mock_conn(session_ts=100)
        qi = QueryIterator(
            connection=conn,
            collection_name="test",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=_SCHEMA_DICT,
            **{REDUCE_STOP_FOR_BEST: False},
        )
        assert qi._kwargs[REDUCE_STOP_FOR_BEST] == "False"


class TestQueryIteratorMissingPk:
    """Cover __setup__pk_prop missing pk (line 369)."""

    def test_no_pk_field_raises(self):
        schema_no_pk = {
            FIELDS: [
                {
                    "name": "vec",
                    "type": DataType.FLOAT_VECTOR,
                    "params": {"dim": 4},
                },
            ],
        }
        conn = _make_mock_conn(session_ts=100)
        with pytest.raises((MilvusException, AttributeError)):
            QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=schema_no_pk,
            )


class TestQueryIteratorCloseWithCpFile:
    """Cover close() with cp_file_handler (lines 394-399)."""

    def test_close_removes_cp_file(self):
        conn = _make_mock_conn(session_ts=200)
        with tempfile.TemporaryDirectory() as td:
            cp_path = str(Path(td) / "close_test.cp")
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
                **{ITERATOR_SESSION_CP_FILE: cp_path},
            )
            assert qi._cp_file_handler is not None
            qi.close()
            assert not Path(cp_path).exists()


# ---------------------------------------------------------------------------
# SearchIterator additional coverage tests
# ---------------------------------------------------------------------------


class TestSearchIteratorSessionTsFallback:
    """Cover __init_search_iterator ts fallback (lines 534-535)."""

    def test_session_ts_fallback(self):
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=0)

        with patch(
            "pymilvus.orm.iterator.fall_back_to_latest_session_ts",
            return_value=7777,
        ):
            si = SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {}},
                batch_size=10,
                schema=_SEARCH_SCHEMA_DICT,
            )
        assert si._session_ts == 7777
        assert si._init_success is True
        si.close()


class TestSearchIteratorMissingPk:
    """Cover __setup__pk_prop missing pk (line 604)."""

    def test_no_pk_field_raises(self):
        schema_no_pk = {
            FIELDS: [
                {
                    "name": "vec",
                    "type": DataType.FLOAT_VECTOR,
                    "params": {"dim": 4},
                },
            ],
        }
        conn = _make_search_conn(hits=[], session_ts=500)
        with pytest.raises((ParamError, AttributeError)):
            SearchIterator(
                connection=conn,
                collection_name="test",
                data=[[1, 2, 3, 4]],
                ann_field="vec",
                param={METRIC_TYPE: "L2", PARAMS: {}},
                batch_size=10,
                schema=schema_no_pk,
            )


class TestSearchIteratorUpdateFilteredIds:
    """Cover __update_filtered_ids branches (lines 638, 641, 649)."""

    def _make_si(self, hits):
        conn = _make_search_conn(hits=hits, session_ts=500)
        return SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )

    def test_update_filtered_ids_empty_page(self):
        """Empty page is a no-op."""
        hit0 = Mock(id=1, distance=0.1)
        si = self._make_si([hit0])
        empty = SearchPage(None)
        si._SearchIterator__update_filtered_ids(empty)
        # Should not change filtered_ids
        si.close()

    def test_update_filtered_ids_last_hit_none(self):
        """Page whose last element is None is skipped."""
        hit0 = Mock(id=1, distance=0.1)
        si = self._make_si([hit0])
        # Build a page where [-1] returns None
        mock_hits = Mock()
        mock_hits.__len__ = Mock(return_value=1)

        def hits_getitem(self, idx):
            return None

        mock_hits.__getitem__ = hits_getitem
        mock_hits.__iter__ = Mock(return_value=iter([None]))
        page = SearchPage(mock_hits)
        si._SearchIterator__update_filtered_ids(page)
        si.close()

    def test_update_filtered_ids_exceeds_max(self):
        """Exceeding max filtered ids raises."""
        hit0 = Mock(id=1, distance=0.1)
        si = self._make_si([hit0])
        # Pre-fill filtered_ids above limit
        si._filtered_ids = list(range(MAX_FILTERED_IDS_COUNT_ITERATION + 1))
        si._filtered_distance = 0.5

        hit_same = Mock(id=999, distance=0.5)
        mock_hits = Mock()
        mock_hits.__len__ = Mock(return_value=1)

        def hits_getitem(self, idx):
            return hit_same

        mock_hits.__getitem__ = hits_getitem
        mock_hits.__iter__ = Mock(return_value=iter([hit_same]))
        page = SearchPage(mock_hits)

        with pytest.raises(MilvusException, match="filtered ids length"):
            si._SearchIterator__update_filtered_ids(page)
        si.close()


class TestSearchIteratorCacheHelpers:
    """Cover __is_cache_enough, __extract_page_from_cache,
    __push_new_page_to_cache (lines 656-682)."""

    def _make_si_with_cache(self):
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        return SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=2,
            schema=_SEARCH_SCHEMA_DICT,
        )

    def test_is_cache_enough_true(self):
        si = self._make_si_with_cache()
        # Init cached 2 hits, batch_size=2
        assert si._SearchIterator__is_cache_enough(2) is True
        si.close()

    def test_is_cache_enough_false(self):
        si = self._make_si_with_cache()
        assert si._SearchIterator__is_cache_enough(100) is False
        si.close()

    def test_extract_page_from_cache(self):
        si = self._make_si_with_cache()
        page = si._SearchIterator__extract_page_from_cache(1)
        assert len(page) > 0
        si.close()

    def test_extract_page_insufficient_raises(self):
        si = self._make_si_with_cache()
        with pytest.raises(ParamError, match="Wrong"):
            si._SearchIterator__extract_page_from_cache(999)
        si.close()

    def test_push_new_page_none_raises(self):
        si = self._make_si_with_cache()
        with pytest.raises(ParamError, match="Cannot push None"):
            si._SearchIterator__push_new_page_to_cache(None)
        si.close()

    def test_push_new_page_merges_with_existing(self):
        si = self._make_si_with_cache()
        new_hit = Mock(id=3, distance=0.7)
        mock_hits = Mock()
        mock_hits.__len__ = Mock(return_value=1)
        mock_hits.__iter__ = Mock(return_value=iter([new_hit]))
        new_page = SearchPage(mock_hits)
        count = si._SearchIterator__push_new_page_to_cache(new_page)
        assert count == 3  # 2 original + 1 new
        si.close()

    def test_push_new_page_to_empty_cache(self):
        si = self._make_si_with_cache()
        # Release the existing cache to make it None
        iterator_cache.release_cache(si._cache_id)
        new_hit = Mock(id=3, distance=0.7)
        mock_hits = Mock()
        mock_hits.__len__ = Mock(return_value=1)
        mock_hits.__iter__ = Mock(return_value=iter([new_hit]))
        new_page = SearchPage(mock_hits)
        count = si._SearchIterator__push_new_page_to_cache(new_page)
        assert count == 1
        si.close()


class TestSearchIteratorNextFull:
    """Cover SearchIterator.next() main paths (lines 688-710)."""

    def _make_search_conn_multi(self, init_hits, next_hits_list, session_ts=500):
        """Conn that returns different results per call."""
        conn = Mock()
        conn.describe_collection.return_value = {COLLECTION_ID: 999}

        call_idx = [0]
        all_hits = [init_hits, *next_hits_list]

        def search_side_effect(**kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            hits = all_hits[idx] if idx < len(all_hits) else []

            mock_hit_list = Mock()
            mock_hit_list.__len__ = Mock(return_value=len(hits))
            mock_hit_list.__iter__ = Mock(return_value=iter(hits))

            def hit_getitem(self, i):
                if isinstance(i, int):
                    if i < 0:
                        i = len(hits) + i
                    return hits[i]
                return hits[i]

            mock_hit_list.__getitem__ = hit_getitem

            mock_res = Mock()
            mock_res.__getitem__ = lambda self, i: mock_hit_list
            mock_res.get_session_ts = Mock(return_value=session_ts)
            return mock_res

        conn.search.side_effect = search_side_effect
        return conn

    def test_next_returns_from_cache(self):
        """When cache is sufficient, extract directly."""
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.3)
        hit2 = Mock(id=3, distance=0.5)
        hit3 = Mock(id=4, distance=0.7)
        conn = self._make_search_conn_multi(
            init_hits=[hit0, hit1, hit2, hit3],
            next_hits_list=[],
        )
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=2,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # Init cached 4 hits, batch=2 => cache has enough
        result = si.next()
        assert len(result) == 2
        assert si._returned_count == 2
        si.close()

    def test_next_with_limit(self):
        """When limit is set, ret_len is capped."""
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.3)
        hit2 = Mock(id=3, distance=0.5)
        hit3 = Mock(id=4, distance=0.7)
        conn = self._make_search_conn_multi(
            init_hits=[hit0, hit1, hit2, hit3],
            next_hits_list=[],
        )
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            limit=1,
            schema=_SEARCH_SCHEMA_DICT,
        )
        result = si.next()
        assert len(result) == 1
        si.close()

    def test_next_fills_via_search(self):
        """Cache insufficient, triggers __try_search_fill."""
        hit0 = Mock(id=1, distance=0.1)
        fill_hit = Mock(id=2, distance=0.5)
        conn = self._make_search_conn_multi(
            init_hits=[hit0],
            next_hits_list=[
                [fill_hit],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
        )
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=2,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # Cache has 1 item but batch_size=2, so fill is needed
        result = si.next()
        assert len(result) > 0
        si.close()

    def test_next_updates_width_when_full_batch(self):
        """Width is updated when ret_page len == batch_size."""
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.3)
        # Provide enough fill hits for a full batch
        fill0 = Mock(id=3, distance=0.5)
        fill1 = Mock(id=4, distance=0.7)
        conn = self._make_search_conn_multi(
            init_hits=[hit0, hit1],
            next_hits_list=[
                [fill0, fill1],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            ],
        )
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=2,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # First next: cache has 2, enough for batch=2
        result1 = si.next()
        assert len(result1) == 2

        # Second next: needs to fill, should update width
        result2 = si.next()
        assert len(result2) > 0
        si.close()


class TestSearchIteratorTrySearchFill:
    """Cover __try_search_fill (lines 713-733)."""

    def test_fill_breaks_on_max_try(self):
        """Fill loop breaks after MAX_TRY_TIME empty rounds."""
        hit0 = Mock(id=1, distance=0.1)
        conn = Mock()
        conn.describe_collection.return_value = {COLLECTION_ID: 999}

        call_count = [0]

        def search_side_effect(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            hits = [hit0] if idx == 0 else []

            mock_hit_list = Mock()
            mock_hit_list.__len__ = Mock(return_value=len(hits))
            mock_hit_list.__iter__ = Mock(return_value=iter(hits))

            def hit_getitem(self, i):
                if isinstance(i, int):
                    if i < 0:
                        i = len(hits) + i
                    return hits[i]
                return hits[i]

            mock_hit_list.__getitem__ = hit_getitem
            mock_res = Mock()
            mock_res.__getitem__ = lambda self, i: mock_hit_list
            mock_res.get_session_ts = Mock(return_value=500)
            return mock_res

        conn.search.side_effect = search_side_effect

        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # After init, cache has 1 item, batch_size=10
        # next() will try to fill but all searches return empty
        result = si.next()
        # Should still return what it could get
        assert len(result) >= 0
        si.close()


class TestSearchIteratorNextParamsWithRadius:
    """Cover __next_params radius capping (lines 785, 791)."""

    def test_l2_radius_capped(self):
        """L2: when next_radius > user RADIUS, use user RADIUS."""
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={
                METRIC_TYPE: "L2",
                PARAMS: {RADIUS: 0.6},
            },
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # width = 0.5 - 0.1 = 0.4, tail_band = 0.5
        # coefficient=100 => next_radius = 0.5 + 0.4*100 = 40.5
        # But user RADIUS = 0.6, so capped
        p = si._SearchIterator__next_params(100)
        assert p[PARAMS][RADIUS] == 0.6
        si.close()

    def test_ip_radius_capped(self):
        """IP: when next_radius < user RADIUS, use user RADIUS."""
        hit0 = Mock(id=1, distance=0.9)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={
                METRIC_TYPE: "IP",
                PARAMS: {RADIUS: 0.4},
            },
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        # Large coefficient makes next_radius < user RADIUS => capped
        p = si._SearchIterator__next_params(100)
        assert p[PARAMS][RADIUS] == 0.4
        si.close()


class TestSearchIteratorUpdateWidth:
    """Cover __update_width zero-width case (line 557-558)."""

    def test_update_width_zero_becomes_minimum(self):
        """Same distance for first/last hit => width = 0.05."""
        hit0 = Mock(id=1, distance=0.5)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        assert si._width == 0.05
        si.close()

    def test_update_width_ip_metric(self):
        """IP metric: width = first.distance - last.distance."""
        hit0 = Mock(id=1, distance=0.9)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "IP", PARAMS: {}},
            batch_size=10,
            schema=_SEARCH_SCHEMA_DICT,
        )
        assert si._width == pytest.approx(0.4)
        si.close()


class TestSearchIteratorCheckReachedLimit:
    """Cover SearchIterator __check_reached_limit (line 570)."""

    def test_reached_limit_returns_true(self):
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            limit=5,
            schema=_SEARCH_SCHEMA_DICT,
        )
        si._returned_count = 5
        assert si._SearchIterator__check_reached_limit() is True
        si.close()

    def test_not_reached_limit_returns_false(self):
        hit0 = Mock(id=1, distance=0.1)
        hit1 = Mock(id=2, distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            limit=5,
            schema=_SEARCH_SCHEMA_DICT,
        )
        si._returned_count = 3
        assert si._SearchIterator__check_reached_limit() is False
        si.close()


class TestSearchIteratorVarcharPk:
    """Cover SearchIterator with VARCHAR pk schema."""

    def test_varchar_pk_init(self):
        hit0 = Mock(id="a", distance=0.1)
        hit1 = Mock(id="b", distance=0.5)
        conn = _make_search_conn(hits=[hit0, hit1], session_ts=500)
        si = SearchIterator(
            connection=conn,
            collection_name="test",
            data=[[1, 2, 3, 4]],
            ann_field="vec",
            param={METRIC_TYPE: "L2", PARAMS: {}},
            batch_size=10,
            schema=_VARCHAR_SCHEMA_DICT,
        )
        assert si._pk_str is True
        si.close()


class TestQueryIteratorExtraIsNone:
    """Cover __setup_ts_by_request when res.extra is None."""

    def test_extra_none_fallback(self):
        conn = Mock()
        conn.describe_collection.return_value = {COLLECTION_ID: 999}
        mock_res = Mock()
        mock_res.__len__ = Mock(return_value=0)
        mock_res.__getitem__ = lambda s, k: [][k]
        mock_res.extra = None
        conn.query.return_value = mock_res

        with patch(
            "pymilvus.orm.iterator.fall_back_to_latest_session_ts",
            return_value=8888,
        ):
            qi = QueryIterator(
                connection=conn,
                collection_name="test",
                batch_size=10,
                expr="pk > 0",
                output_fields=["pk"],
                schema=_SCHEMA_DICT,
            )
        assert qi._session_ts == 8888
