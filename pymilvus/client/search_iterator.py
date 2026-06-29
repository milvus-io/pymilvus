import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

from pymilvus.client import entity_helper, utils
from pymilvus.client.abstract import AnnSearchRequest
from pymilvus.client.constants import (
    COLLECTION_ID,
    GUARANTEE_TIMESTAMP,
    ITER_SEARCH_BATCH_SIZE_KEY,
    ITER_SEARCH_ID_KEY,
    ITER_SEARCH_LAST_BOUND_KEY,
    ITER_SEARCH_V2_KEY,
    ITERATOR_FIELD,
)
from pymilvus.client.embedding_list import EmbeddingList
from pymilvus.client.search_result import Hit, Hits
from pymilvus.exceptions import ExceptionsMessage, ParamError, ServerVersionIncompatibleException
from pymilvus.orm.connections import Connections
from pymilvus.orm.constants import MAX_BATCH_SIZE, OFFSET, UNLIMITED
from pymilvus.orm.iterator import SearchPage, fall_back_to_latest_session_ts

logger = logging.getLogger(__name__)


class SearchIteratorV2:
    # for compatibility, track the number of total results left
    _left_res_cnt = None

    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        data: Union[List, utils.SparseMatrixInputType],
        batch_size: int = 1000,
        limit: Optional[int] = UNLIMITED,
        filter: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        round_decimal: Optional[int] = -1,
        external_filter_func: Optional[Callable[[Hits], Union[Hits, List[Hit]]]] = None,
        **kwargs,
    ):
        self._check_params(batch_size, data, kwargs)

        # An emb_list (array-of-vector) query is one EmbeddingList holding that
        # query's multiple vectors. Flatten it for the wire and flag it -- exactly
        # as MilvusClient.search does -- so the server reads it as a single emb_list
        # query rather than as many single-vector queries.
        if isinstance(data, list) and data and isinstance(data[0], EmbeddingList):
            data = [emb_list.to_flat_array() for emb_list in data]
            kwargs["is_embedding_list"] = True

        # for compatibility, support limit, deprecate in future
        if limit != UNLIMITED:
            self._left_res_cnt = limit

        self._conn = connection
        self._set_up_collection_id(collection_name, **kwargs)
        kwargs[COLLECTION_ID] = self._collection_id
        self._params = {
            "collection_name": collection_name,
            "data": data,
            "anns_field": anns_field,
            "param": deepcopy(search_params),
            "limit": batch_size,
            "expression": filter,
            "partition_names": partition_names,
            "output_fields": output_fields,
            "timeout": timeout,
            "round_decimal": round_decimal,
            ITERATOR_FIELD: True,
            ITER_SEARCH_V2_KEY: True,
            ITER_SEARCH_BATCH_SIZE_KEY: batch_size,
            GUARANTEE_TIMESTAMP: 0,
            **kwargs,
        }
        self._external_filter_func = external_filter_func
        self._cache = []
        self._batch_size = batch_size
        self._probe_for_compability(self._params)

    def _set_up_collection_id(self, collection_name: str, **kwargs):
        res = self._conn.describe_collection(collection_name, **kwargs)
        self._collection_id = res[COLLECTION_ID]

    def _check_token_exists(self, token: Union[str, None]):
        if token is None or token == "":
            raise ServerVersionIncompatibleException(
                message=ExceptionsMessage.SearchIteratorV2FallbackWarning
            )

    # this detects whether the server supports search_iterator_v2 and is for compatibility only
    # if the server holds iterator states, this implementation needs to be reconsidered
    def _probe_for_compability(self, params: Dict):
        dummy_params = deepcopy(params)
        dummy_batch_size = 1
        dummy_params["limit"] = dummy_batch_size
        dummy_params[ITER_SEARCH_BATCH_SIZE_KEY] = dummy_batch_size
        probe_result = self._conn.search(**dummy_params)
        iter_info = probe_result.get_search_iterator_v2_results_info()
        self._check_token_exists(iter_info.token)
        # Pin GUARANTEE_TIMESTAMP from probe call's session_ts so that all subsequent
        # next() calls (including the very first) see a consistent MVCC snapshot.
        # Without this, the first next() runs with GUARANTEE_TIMESTAMP=0 (latest available),
        # which means a segment reload triggered by add_collection_field can shift distances
        # by 1 ULP, causing last_bound items to pass the dist > last_bound filter again
        # and produce duplicate PKs. See: https://github.com/milvus-io/pymilvus/issues/3421
        if params[GUARANTEE_TIMESTAMP] <= 0:
            session_ts = probe_result.get_session_ts()
            if session_ts > 0:
                params[GUARANTEE_TIMESTAMP] = session_ts
            else:
                logger.warning(
                    "failed to set up mvccTs from probe call, use client-side ts instead"
                )
                params[GUARANTEE_TIMESTAMP] = fall_back_to_latest_session_ts()

    # internal next function, do not use this outside of this class
    def _next(self):
        res = self._conn.search(**self._params)
        iter_info = res.get_search_iterator_v2_results_info()
        self._check_token_exists(iter_info.token)
        self._params[ITER_SEARCH_LAST_BOUND_KEY] = iter_info.last_bound

        # patch token and guarantee timestamp for the first next() call
        if ITER_SEARCH_ID_KEY not in self._params:
            # the token should not change during the lifetime of the iterator
            self._params[ITER_SEARCH_ID_KEY] = iter_info.token
        if self._params[GUARANTEE_TIMESTAMP] <= 0:
            if res.get_session_ts() > 0:
                self._params[GUARANTEE_TIMESTAMP] = res.get_session_ts()
            else:
                logger.warning(
                    "failed to set up mvccTs from milvus server, use client-side ts instead"
                )
                self._params[GUARANTEE_TIMESTAMP] = fall_back_to_latest_session_ts()
        return res

    def next(self):
        if self._left_res_cnt is not None and self._left_res_cnt <= 0:
            return None

        if self._external_filter_func is None:
            # return SearchPage for compability
            return self._wrap_return_res(self._next()[0])
        # the length of the results should be `batch_size` if no limit is set,
        # otherwise it should be the number of results left if less than `batch_size`
        target_len = (
            self._batch_size
            if self._left_res_cnt is None
            else min(self._batch_size, self._left_res_cnt)
        )
        while True:
            hits = self._next()[0]

            # no more results from server
            if len(hits) == 0:
                break

            # apply external filter
            if self._external_filter_func is not None:
                hits = self._external_filter_func(hits)

            self._cache.extend(hits)
            if len(self._cache) >= target_len:
                break

        # if the number of elements in cache is less than or equal to target_len,
        #   return all results we could possibly return
        # if the number of elements in cache is more than target_len,
        #   return target_len results and keep the rest for next call
        ret = self._cache[:target_len]
        del self._cache[:target_len]
        # return SearchPage for compability
        return self._wrap_return_res(ret)

    def close(self):
        pass

    def _check_params(
        self,
        batch_size: int,
        data: Union[List, utils.SparseMatrixInputType],
        kwargs: Dict,
    ):
        # metric_type can be empty, deduced at server side
        # anns_field can be empty, deduced at server side

        # check batch size
        if batch_size < 0:
            raise ParamError(message="batch size cannot be less than zero")
        if batch_size > MAX_BATCH_SIZE:
            raise ParamError(message=f"batch size cannot be larger than {MAX_BATCH_SIZE}")

        # check offset
        if kwargs.get(OFFSET, 0) != 0:
            raise ParamError(message="Offset is not supported for search_iterator_v2")

        # check num queries, heavy to check at server side
        rows = entity_helper.get_input_num_rows(data)
        if rows > 1:
            raise ParamError(
                message="search_iterator_v2 does not support processing multiple vectors simultaneously"
            )
        if rows == 0:
            raise ParamError(message="The vector data for search cannot be empty")

    def _wrap_return_res(self, res: Hits) -> SearchPage:
        if len(res) == 0:
            return SearchPage(None)

        if self._left_res_cnt is None:
            return SearchPage(res)

        # When we have a limit, ensure we don't return more results than requested
        cur_len = len(res)
        if cur_len > self._left_res_cnt:
            res = res[: self._left_res_cnt]
        self._left_res_cnt -= cur_len
        return SearchPage(res)


# Reciprocal-rank-fusion constant (Cormack et al. 2009); see SPEC 6.6 / Appendix A.3.
_DEFAULT_RRF_K = 60
# Soft bound on the NRA in-flight (seen-but-not-emitted) map. The map is bounded by
# dense/sparse stream skew (SPEC R4). On overflow the fuser force-flushes its best
# in-flight document rather than failing the iteration -- client-side memory is cheap.
_DEFAULT_INFLIGHT_CAP = 100000


class _StreamCursor:
    """Item-by-item, lazily-refilled view over one batched, score-descending stream.

    `fetch_batch` returns the next batch as a list of (doc_id, score) pairs, or an
    empty list once the stream is exhausted.
    """

    def __init__(self, fetch_batch: Callable[[], List]):
        self._fetch_batch = fetch_batch
        self._buf: List = []
        self._pos = 0
        self.reads = 0
        self.exhausted = False

    def advance(self):
        """Consume and return the next (doc_id, score), or None when exhausted."""
        if self._pos >= len(self._buf):
            if self.exhausted:
                return None
            batch = self._fetch_batch() or []
            self._buf, self._pos = list(batch), 0
            if not self._buf:
                self.exhausted = True
                return None
        item = self._buf[self._pos]
        self._pos += 1
        self.reads += 1
        return item


class _RrfHybridFuser:
    """Incremental RRF fusion of N score-descending streams via the NRA algorithm.

    Fagin/Lotem/Naor 2003 (NRA threshold algorithm) + Cormack et al. 2009 (RRF).
    See SPEC 6.6 and Appendix A.3. RRF is rank-based: a stream's contribution to any
    not-yet-seen document is 1 / (K + reads_so_far + 1), independent of raw scores.
    The fuser advances the least-read live stream and settles a document once no
    in-flight or unseen document can overtake it. State (the `seen` map, the `emitted`
    set and per-stream read counts) persists across next_batch() calls for the
    lifetime of the iterator.

    Emission order is exact: each emitted document provably out-ranks every
    not-yet-emitted document by true RRF score. The emitted *score*, however, is a
    lower bound -- a document is settled with the partial RRF score (`worst`) of the
    streams it has been seen in so far; if it later resurfaces in a not-yet-consumed
    stream, that document is already emitted and the extra rank is dropped (it cannot
    change the ranking). Callers that need the exact RRF score must re-score.
    """

    def __init__(
        self,
        fetch_batches: List[Callable[[], List]],
        rrf_k: int = _DEFAULT_RRF_K,
        inflight_cap: int = _DEFAULT_INFLIGHT_CAP,
    ):
        self._cursors = [_StreamCursor(fb) for fb in fetch_batches]
        self._k = rrf_k
        self._inflight_cap = inflight_cap
        # doc_id -> {stream_idx: rank}; the NRA in-flight (seen-but-not-emitted) map
        self._seen: Dict = {}
        # doc_ids already emitted -- a stream resurfacing one must not re-add it to
        # `seen` (that would re-emit it; deep skewed streams hit this readily)
        self._emitted: set = set()

    def _contribution(self, cursor: _StreamCursor) -> float:
        # an exhausted stream can never produce another rank -> contributes 0
        if cursor.exhausted:
            return 0.0
        return 1.0 / (self._k + cursor.reads + 1)

    def _worst(self, ranks: Dict) -> float:
        # RRF score from the ranks already known for a document
        return sum(1.0 / (self._k + rank) for rank in ranks.values())

    def _advance_least_read(self):
        """Advance the least-read live stream by one item (largest RRF contribution)."""
        live = [(i, c) for i, c in enumerate(self._cursors) if not c.exhausted]
        if not live:
            return
        idx, cursor = min(live, key=lambda ic: ic[1].reads)
        item = cursor.advance()
        if item is not None:
            doc_id = item[0]
            # an already-emitted doc resurfacing in another stream is dropped --
            # it is ranked; re-adding it to `seen` would emit a duplicate
            if doc_id not in self._emitted:
                self._seen.setdefault(doc_id, {}).setdefault(idx, cursor.reads)

    def _settled(self):
        """Return (doc_id, rrf_score) if the best in-flight doc is provably the next
        global result, else None."""
        if not self._seen:
            return None
        tau = sum(self._contribution(c) for c in self._cursors)
        worsts = {doc: self._worst(ranks) for doc, ranks in self._seen.items()}
        cand = max(worsts, key=worsts.get)
        cand_worst = worsts[cand]
        # an unseen document could still outrank cand
        if cand_worst < tau:
            return None
        # an in-flight document could still outrank cand
        for doc, ranks in self._seen.items():
            if doc == cand:
                continue
            best = worsts[doc] + sum(
                self._contribution(self._cursors[i])
                for i in range(len(self._cursors))
                if i not in ranks
            )
            if best > cand_worst:
                return None
        return cand, cand_worst

    def _emit(self, out: List, doc_id: Union[int, str], score: float):
        del self._seen[doc_id]
        self._emitted.add(doc_id)
        out.append((doc_id, score))

    def next_batch(self, batch_size: int) -> List:
        """Fuse and return up to batch_size (doc_id, rrf_score) pairs.

        Emitted RRF-descending; the score is a lower bound on the true RRF score
        (see the class docstring).
        """
        out: List = []
        while len(out) < batch_size:
            settled = self._settled()
            if settled is not None:
                self._emit(out, settled[0], settled[1])
                continue
            if all(c.exhausted for c in self._cursors):
                # no stream can advance; _settled() has already drained `seen`
                break
            self._advance_least_read()
            # adjustment 1: force-flush the best in-flight doc on overflow -- bound
            # memory on skewed streams without failing the iteration
            if self._inflight_cap and len(self._seen) > self._inflight_cap:
                worsts = {doc: self._worst(r) for doc, r in self._seen.items()}
                cand = max(worsts, key=worsts.get)
                self._emit(out, cand, worsts[cand])
        return out


class HybridSearchIteratorV2:
    """Hybrid search_iterator: client-side RRF fusion of per-modality search_iterators.

    Per SPEC 6.6 (corrected by the R3 spike), hybrid iteration is done in the SDK:
    each request becomes a stateless single-modality SearchIteratorV2, and their
    score-descending streams are fused with RRF via the NRA threshold algorithm
    (_RrfHybridFuser). One guarantee_timestamp is pinned for the whole hybrid
    iterator and shared by every sub-iterator, so the fusion is over a single
    consistent snapshot.

    `reqs` is a list of per-modality requests -- each an `AnnSearchRequest` or a plain
    dict {"data", "anns_field", "param", "expr"}. A request's `data` may be an
    `EmbeddingList` (one emb_list query); the sub-iterator flattens it (see
    SearchIteratorV2). A per-request `limit` is not used -- the iterator streams in
    `batch_size` chunks.

    next() returns the fused batch as a SearchPage of Hits, RRF-descending. The emitted
    score is the NRA lower bound (see _RrfHybridFuser); the `output_fields` requested on
    the hybrid iterator are forwarded to every sub-iterator and carried back through the
    fusion onto each fused Hit's `entity`.
    """

    def __init__(
        self,
        connection: Connections,
        collection_name: str,
        reqs: List[Union[Dict, AnnSearchRequest]],
        batch_size: int = 1000,
        limit: Optional[int] = UNLIMITED,
        rrf_k: int = _DEFAULT_RRF_K,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        pk_name: str = "id",
        guarantee_timestamp: Optional[int] = None,
        **kwargs,
    ):
        if not reqs:
            raise ParamError(message="hybrid search_iterator requires at least one request")
        if batch_size < 0 or batch_size > MAX_BATCH_SIZE:
            raise ParamError(message=f"batch_size must be in [0, {MAX_BATCH_SIZE}]")
        reqs = [self._normalize_req(req) for req in reqs]

        self._batch_size = batch_size
        self._pk_name = pk_name
        self._left_res_cnt = None if limit == UNLIMITED else limit
        # doc_id -> the sub-iterator Hit it was last fetched in, so next() can carry
        # the requested output_fields back through the fusion (see _make_fetch_batch)
        self._hits_by_id: Dict = {}

        # adjustment 3: pin ONE snapshot timestamp and share it with every
        # sub-iterator, so the fused result is over one consistent snapshot
        pinned_ts = guarantee_timestamp if guarantee_timestamp else fall_back_to_latest_session_ts()

        self._subs: List[SearchIteratorV2] = []
        for req in reqs:
            sub = SearchIteratorV2(
                connection=connection,
                collection_name=collection_name,
                data=req["data"],
                batch_size=batch_size,
                filter=req.get("expr"),
                output_fields=output_fields,
                search_params=req.get("param") or {},
                timeout=timeout,
                partition_names=partition_names,
                anns_field=req.get("anns_field") or "",
                **{GUARANTEE_TIMESTAMP: pinned_ts},
                **kwargs,
            )
            self._subs.append(sub)

        self._fuser = _RrfHybridFuser(
            [self._make_fetch_batch(sub) for sub in self._subs], rrf_k=rrf_k
        )

    @staticmethod
    def _normalize_req(req: Union[Dict, AnnSearchRequest]) -> Dict:
        """Accept an AnnSearchRequest or a plain dict; return the internal dict form.

        A per-request `limit` is intentionally dropped -- a hybrid sub-iterator
        streams in `batch_size` chunks, governed by the hybrid iterator's batch_size.
        """
        if isinstance(req, AnnSearchRequest):
            return {
                "data": req.data,
                "anns_field": req.anns_field,
                "param": req.param,
                "expr": req.expr,
            }
        return req

    def _make_fetch_batch(self, sub: "SearchIteratorV2") -> Callable[[], List]:
        """Adapt a SearchIteratorV2 into a () -> [(doc_id, score)] batch source.

        Each hit's full record is stashed in `_hits_by_id` so next() can carry the
        requested output_fields back through the fusion; the stash entry is dropped
        when the doc is emitted (see next()).
        """

        def fetch_batch() -> List:
            page = sub.next()
            if page is None or len(page) == 0:
                return []
            out = []
            for hit in page:
                pk = hit[self._pk_name]
                self._hits_by_id[pk] = hit
                out.append((pk, float(hit["distance"])))
            return out

        return fetch_batch

    def next(self) -> SearchPage:
        """Return the next fused batch as a SearchPage, RRF-descending."""
        if self._left_res_cnt is not None and self._left_res_cnt <= 0:
            return SearchPage(None)
        target = self._batch_size
        if self._left_res_cnt is not None:
            target = min(self._batch_size, self._left_res_cnt)

        fused = self._fuser.next_batch(target)
        if self._left_res_cnt is not None:
            self._left_res_cnt -= len(fused)

        pks = [doc for doc, _ in fused]
        scores = [score for _, score in fused]
        hits = Hits(len(fused), pks, scores, {}, [], self._pk_name)
        # carry the requested output_fields back through the fusion: copy each
        # emitted doc's entity from its stashed sub-iterator hit, then drop the stash
        for hit, doc_id in zip(hits, pks):
            src = self._hits_by_id.pop(doc_id, None)
            if src is not None:
                entity = src.get("entity")
                if entity:
                    hit["entity"] = dict(entity)
        # A doc can resurface in another stream after it was emitted: the fuser skips
        # it (its _emitted set) but fetch_batch still stashed the hit. Drop those
        # stragglers so _hits_by_id stays bounded by dense/sparse stream skew.
        emitted = self._fuser._emitted
        self._hits_by_id = {pk: hit for pk, hit in self._hits_by_id.items() if pk not in emitted}
        return SearchPage(hits)

    def close(self):
        for sub in self._subs:
            sub.close()
