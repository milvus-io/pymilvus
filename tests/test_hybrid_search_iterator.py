"""Unit tests for the client-side hybrid search_iterator RRF fusion layer.

PR 5 added the NRA/RRF fuser (`_RrfHybridFuser`); PR 6a finished the SDK wiring --
`AnnSearchRequest` requests, `output_fields` passthrough through the fusion, and
`EmbeddingList` emb_list queries. Everything here exercises that over mock streams /
a fake connection -- no live milvus (the end-to-end run is the §10 acceptance stage).
"""

import pytest
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.constants import COLLECTION_ID
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.search_iterator import (
    HybridSearchIteratorV2,
    SearchIteratorV2,
    _RrfHybridFuser,
    _StreamCursor,
)
from pymilvus.client.search_result import Hit
from pymilvus.exceptions import ParamError

_SUB_TARGET = "pymilvus.client.search_iterator.SearchIteratorV2"


def make_source(items, batch_size=4):
    """A () -> [(doc_id, score)] batch source serving `items` in fixed-size batches."""
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    it = iter(batches)

    def fetch_batch():
        return next(it, [])

    return fetch_batch


def reference_rrf(streams, k):
    """Brute-force RRF: score(d) = sum over streams where d appears of 1/(k+rank)."""
    ranks = {}
    for si, stream in enumerate(streams):
        for rank, (doc, _score) in enumerate(stream, start=1):
            ranks.setdefault(doc, {})[si] = rank
    scored = [
        (doc, sum(1.0 / (k + r) for r in by_stream.values())) for doc, by_stream in ranks.items()
    ]
    scored.sort(key=lambda x: -x[1])
    return scored


def drain(fuser, batch_size):
    out = []
    while True:
        batch = fuser.next_batch(batch_size)
        if not batch:
            break
        out.extend(batch)
    return out


def assert_valid_rrf(fused, streams, k):
    """`fused` is a correct RRF fusion of `streams`.

    NRA emits each document with a *lower-bound* RRF score (the partial score of the
    streams it had been seen in when it settled -- see _RrfHybridFuser). So the checks
    are: complete, deduplicated; emitted in true-RRF-descending order; each emitted
    score a lower bound on the true RRF score; and the emitted lower bounds are
    themselves non-increasing.
    """
    ref_score = dict(reference_rrf(streams, k))

    ids = [doc for doc, _ in fused]
    assert len(ids) == len(set(ids)), "duplicate doc id in fused output"
    assert set(ids) == set(ref_score), "fused id set != reference"

    # emitted in true-RRF-descending order (the NRA emission rule guarantees this)
    true_seq = [ref_score[doc] for doc, _ in fused]
    for a, b in zip(true_seq, true_seq[1:]):
        assert a >= b - 1e-9, "fused output is not in true-RRF-descending order"

    # each emitted score is a lower bound on the true RRF score, and the emitted
    # lower bounds are non-increasing
    emitted = [s for _, s in fused]
    for a, b in zip(emitted, emitted[1:]):
        assert a >= b - 1e-9, "emitted scores are not non-increasing"
    for doc, score in fused:
        assert score <= ref_score[doc] + 1e-9, "emitted score exceeds true RRF score"


class TestRrfHybridFuser:
    K = 60

    def test_disjoint_streams(self):
        s0 = [(f"a{i}", 1.0 - i * 0.01) for i in range(10)]
        s1 = [(f"b{i}", 1.0 - i * 0.01) for i in range(10)]
        fuser = _RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K)
        assert_valid_rrf(drain(fuser, 5), [s0, s1], self.K)

    def test_overlapping_doc_sums_both_ranks(self):
        # "x" is rank 1 in s0 and rank 1 in s1 -> highest RRF (2/(K+1))
        s0 = [("x", 0.9), ("a", 0.8), ("b", 0.7)]
        s1 = [("x", 0.5), ("c", 0.4), ("d", 0.3)]
        fuser = _RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K)
        fused = drain(fuser, 10)
        assert_valid_rrf(fused, [s0, s1], self.K)
        assert fused[0][0] == "x"
        assert fused[0][1] == pytest.approx(2.0 / (self.K + 1), abs=1e-9)

    def test_full_overlap(self):
        # same docs, reversed order in each stream: doc d_i is rank i+1 in s0 and
        # rank 8-i in s1, so RRF pairs d_i with d_{7-i} at equal scores.
        s0 = [(f"d{i}", 1.0 - i * 0.1) for i in range(8)]
        s1 = [(f"d{i}", 1.0 - i * 0.1) for i in reversed(range(8))]
        fuser = _RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K)
        fused = drain(fuser, 3)
        assert_valid_rrf(fused, [s0, s1], self.K)
        assert len(fused) == 8

    def test_skewed_deep_shared_doc_emitted_once(self):
        # The blocker case (deep skewed hybrid iteration). s0 is a single-item
        # stream holding `shared`; once s0 is exhausted `shared` settles and is
        # emitted -- long before the long stream s1 reaches `shared` at rank 51.
        # Without the _emitted set, s1 re-yielding `shared` re-adds it to `seen`
        # and it is emitted a second time.
        shared = "shared"
        s0 = [(shared, 1.0)]
        s1 = (
            [(f"b{i}", 0.9 - i * 0.001) for i in range(50)]
            + [(shared, 0.5)]
            + [(f"b{i}", 0.4 - i * 0.001) for i in range(50, 90)]
        )
        fuser = _RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K)
        fused = drain(fuser, 5)

        ids = [d for d, _ in fused]
        assert ids.count(shared) == 1, "shared doc emitted more than once"
        assert len(ids) == len(set(ids)) == 91  # shared + 90 distinct b's

        # `shared` settled from its rank-1 sighting in s0 before s1 reached it,
        # so its emitted score is the lower bound 1/(K+1) -- not the true RRF
        # score 1/(K+1) + 1/(K+51)
        shared_score = dict(fused)[shared]
        assert shared_score == pytest.approx(1.0 / (self.K + 1), abs=1e-9)
        true_shared = 1.0 / (self.K + 1) + 1.0 / (self.K + 51)
        assert shared_score < true_shared

        assert_valid_rrf(fused, [s0, s1], self.K)

    def test_one_stream_empty(self):
        s0 = [(f"a{i}", 1.0 - i * 0.01) for i in range(12)]
        s1 = []
        fuser = _RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K)
        assert_valid_rrf(drain(fuser, 5), [s0, s1], self.K)

    def test_uneven_stream_lengths(self):
        s0 = [(f"a{i}", 1.0 - i * 0.001) for i in range(50)]
        s1 = [(f"b{i}", 1.0 - i * 0.001) for i in range(7)]
        s1[3] = ("a10", 0.5)  # one shared doc
        fuser = _RrfHybridFuser([make_source(s0, 8), make_source(s1, 3)], rrf_k=self.K)
        assert_valid_rrf(drain(fuser, 6), [s0, s1], self.K)

    def test_batching_is_consistent(self):
        s0 = [(f"a{i}", 1.0 - i * 0.01) for i in range(20)]
        s1 = [(f"b{i}", 1.0 - i * 0.01) for i in range(20)]
        s1[5] = ("a3", 0.5)
        s1[9] = ("a7", 0.4)
        one_shot = drain(_RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K), 10_000)
        small_batches = drain(_RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K), 3)
        assert one_shot == small_batches

    def test_force_flush_under_tiny_cap(self):
        # a tiny in-flight cap must still yield every doc exactly once, no error
        s0 = [(f"a{i}", 1.0 - i * 0.001) for i in range(40)]
        s1 = [(f"b{i}", 1.0 - i * 0.001) for i in range(40)]
        fuser = _RrfHybridFuser([make_source(s0), make_source(s1)], rrf_k=self.K, inflight_cap=4)
        fused = drain(fuser, 7)
        ids = [d for d, _ in fused]
        assert len(ids) == len(set(ids)) == 80

    def test_empty_both_streams(self):
        fuser = _RrfHybridFuser([make_source([]), make_source([])], rrf_k=self.K)
        assert fuser.next_batch(10) == []


class TestStreamCursor:
    def test_lazy_refill_and_exhaustion(self):
        cursor = _StreamCursor(make_source([(i, 1.0) for i in range(5)], batch_size=2))
        got = []
        while True:
            item = cursor.advance()
            if item is None:
                break
            got.append(item[0])
        assert got == [0, 1, 2, 3, 4]
        assert cursor.reads == 5
        assert cursor.exhausted
        assert cursor.advance() is None  # stays exhausted


# --- PR 6a: HybridSearchIteratorV2 SDK wiring over fake sub-iterators -----------


class _FakeSub:
    """Stand-in for SearchIteratorV2: replays the pages handed in as `data`.

    `HybridSearchIteratorV2.__init__` builds one SearchIteratorV2 per request; in
    these tests `SearchIteratorV2` is monkeypatched to this class and the request's
    `data` carries that modality's page stream (a list of pages, each a list of Hit).
    """

    def __init__(self, **kwargs):
        self._pages = iter(kwargs["data"] or [])
        self.output_fields = kwargs.get("output_fields")
        self.closed = False

    def next(self):
        return next(self._pages, None)

    def close(self):
        self.closed = True


def _page(items, pk_name="id"):
    """Build one result page (list of Hit) from (id, score) or (id, score, entity)."""
    page = []
    for it in items:
        doc, score = it[0], it[1]
        entity = it[2] if len(it) > 2 else {}
        page.append(Hit({pk_name: doc, "distance": score, "entity": entity}, pk_name=pk_name))
    return page


def _stream_to_pages(items, batch_size=4):
    """Chunk a flat (id, score[, entity]) stream into a list of result pages."""
    return [_page(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def _make_hybrid(monkeypatch, streams, *, as_ann_request=False, **kwargs):
    """Construct a HybridSearchIteratorV2 whose sub-iterators replay `streams`.

    `streams` is a list of page-stream lists, one per modality.
    """
    monkeypatch.setattr(_SUB_TARGET, _FakeSub)
    reqs = []
    for i, pages in enumerate(streams):
        if as_ann_request:
            reqs.append(AnnSearchRequest(data=pages, anns_field=f"f{i}", param={}, limit=10))
        else:
            reqs.append({"data": pages, "anns_field": f"f{i}", "param": {}})
    return HybridSearchIteratorV2(connection=None, collection_name="c", reqs=reqs, **kwargs)


def _drain_hybrid(it):
    """Drain a HybridSearchIteratorV2, returning the flat list of fused Hits."""
    out = []
    while True:
        page = it.next()
        if page is None or len(page) == 0:
            break
        out.extend(page)
    return out


class TestHybridSearchIteratorWiring:
    K = 60

    def test_dict_and_annsearchrequest_reqs_are_equivalent(self, monkeypatch):
        # the same streams, expressed as plain dicts vs AnnSearchRequest objects,
        # must fuse to the identical result -- _normalize_req unifies them
        s0 = [(f"a{i}", 1.0 - i * 0.01) for i in range(12)]
        s1 = [(f"b{i}", 1.0 - i * 0.01) for i in range(12)]
        s1[4] = ("a3", 0.5)

        as_dict = _drain_hybrid(
            _make_hybrid(monkeypatch, [_stream_to_pages(s0), _stream_to_pages(s1)], batch_size=5)
        )
        as_ann = _drain_hybrid(
            _make_hybrid(
                monkeypatch,
                [_stream_to_pages(s0), _stream_to_pages(s1)],
                as_ann_request=True,
                batch_size=5,
            )
        )
        assert [(h.id, h.distance) for h in as_dict] == [(h.id, h.distance) for h in as_ann]
        assert len(as_dict) == 23  # 12 + 12 - 1 shared

    def test_output_fields_carried_through_fusion(self, monkeypatch):
        # each emitted fused hit carries the entity of its source sub-iterator hit
        s0 = [(f"a{i}", 1.0 - i * 0.01, {"tag": f"a{i}", "src": "dense"}) for i in range(8)]
        s1 = [(f"b{i}", 1.0 - i * 0.01, {"tag": f"b{i}", "src": "sparse"}) for i in range(8)]
        it = _make_hybrid(
            monkeypatch,
            [_stream_to_pages(s0), _stream_to_pages(s1)],
            output_fields=["tag", "src"],
            batch_size=4,
        )
        fused = _drain_hybrid(it)
        assert len(fused) == 16
        for hit in fused:
            expected_src = "dense" if hit.id.startswith("a") else "sparse"
            assert hit["entity"] == {"tag": hit.id, "src": expected_src}
            assert hit["tag"] == hit.id  # Hit field access falls through to entity

    def test_output_fields_for_doc_shared_by_both_streams(self, monkeypatch):
        # a doc in both modalities still carries its entity exactly once
        ent = {"tag": "x", "body": "shared doc"}
        s0 = [("x", 0.9, ent), ("a", 0.8, {"tag": "a"})]
        s1 = [("x", 0.5, ent), ("c", 0.4, {"tag": "c"})]
        it = _make_hybrid(
            monkeypatch,
            [_stream_to_pages(s0), _stream_to_pages(s1)],
            output_fields=["tag", "body"],
            batch_size=10,
        )
        fused = _drain_hybrid(it)
        ids = [h.id for h in fused]
        assert ids.count("x") == 1
        x_hit = next(h for h in fused if h.id == "x")
        assert x_hit["entity"] == ent

    def test_hit_map_drains_fully_no_leak(self, monkeypatch):
        # after a full drain every stashed hit has been emitted or pruned
        s0 = [(f"a{i}", 1.0 - i * 0.001, {"tag": i}) for i in range(40)]
        s1 = [(f"b{i}", 1.0 - i * 0.001, {"tag": i}) for i in range(40)]
        s1[7] = ("a5", 0.5, {"tag": 5})
        it = _make_hybrid(monkeypatch, [_stream_to_pages(s0), _stream_to_pages(s1)], batch_size=6)
        _drain_hybrid(it)
        assert it._hits_by_id == {}

    def test_hit_map_bounded_when_doc_resurfaces_after_emission(self, monkeypatch):
        # the skewed-deep blocker (cf. test_skewed_deep_shared_doc_emitted_once):
        # `shared` settles from a short stream long before the deep stream reaches
        # it. The deep stream re-yields it post-emission; fetch_batch re-stashes the
        # hit, but next() prunes it against the fuser's _emitted set -> no leak.
        shared_ent = {"tag": "shared"}
        s0 = [("shared", 1.0, shared_ent)]
        s1 = (
            [(f"b{i}", 0.9 - i * 0.001, {"tag": i}) for i in range(50)]
            + [("shared", 0.5, shared_ent)]
            + [(f"b{i}", 0.4 - i * 0.001, {"tag": i}) for i in range(50, 90)]
        )
        it = _make_hybrid(monkeypatch, [_stream_to_pages(s0), _stream_to_pages(s1)], batch_size=5)
        fused = _drain_hybrid(it)
        ids = [h.id for h in fused]
        assert ids.count("shared") == 1
        assert len(ids) == len(set(ids)) == 91
        assert it._hits_by_id == {}

    def test_close_propagates_to_sub_iterators(self, monkeypatch):
        it = _make_hybrid(
            monkeypatch,
            [_stream_to_pages([("a", 0.9)]), _stream_to_pages([("b", 0.8)])],
        )
        it.close()
        assert all(sub.closed for sub in it._subs)


# --- PR 6a: SearchIteratorV2 accepts EmbeddingList emb_list queries ------------


class _FakeIterInfo:
    token = "iter-token"
    last_bound = 0.0


class _FakeProbeResult:
    def get_search_iterator_v2_results_info(self):
        return _FakeIterInfo()

    def get_session_ts(self):
        return 12345


class _FakeConn:
    """Minimal connection: enough for SearchIteratorV2 construction + the probe."""

    def __init__(self):
        self.last_search_params = None

    def describe_collection(self, collection_name, **kwargs):
        return {COLLECTION_ID: 7}

    def search(self, **params):
        self.last_search_params = params
        return _FakeProbeResult()


class TestSearchIteratorV2EmbList:
    def test_embedding_list_query_is_flattened_and_flagged(self):
        conn = _FakeConn()
        query = EmbeddingList([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        it = SearchIteratorV2(connection=conn, collection_name="c", data=[query], anns_field="emb")
        # the EmbeddingList was flattened to one wire vector and flagged emb_list
        assert it._params["is_embedding_list"] is True
        sent = it._params["data"]
        assert len(sent) == 1
        assert list(sent[0]) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        # and the flag reached the server call (the compatibility probe)
        assert conn.last_search_params["is_embedding_list"] is True

    def test_plain_single_vector_query_is_untouched(self):
        conn = _FakeConn()
        it = SearchIteratorV2(
            connection=conn, collection_name="c", data=[[0.1, 0.2]], anns_field="emb"
        )
        assert "is_embedding_list" not in it._params

    def test_multiple_embedding_lists_rejected_as_multi_query(self):
        conn = _FakeConn()
        two = [EmbeddingList([[0.1, 0.2]]), EmbeddingList([[0.3, 0.4]])]
        with pytest.raises(ParamError):
            SearchIteratorV2(connection=conn, collection_name="c", data=two, anns_field="emb")
