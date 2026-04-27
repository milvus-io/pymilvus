# Copyright (C) 2019-2026 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

"""Tests for pymilvus/client/search_aggregation.py and its prepare.py integration."""

from unittest.mock import patch

import pytest
from pymilvus import AggregationBucket, SearchAggregation, TopHits
from pymilvus.client.prepare import Prepare
from pymilvus.client.search_result import SearchResult
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import schema_pb2


def _basic_agg(**overrides):
    kw = {
        "fields": ["brand"],
        "size": 5,
        "metrics": {"avg_p": {"avg": "price"}},
        "order": [{"avg_p": "desc"}],
        "top_hits": TopHits(size=2),
    }
    kw.update(overrides)
    return SearchAggregation(**kw)


def _prepare_search(**kwargs):
    base = {
        "collection_name": "c",
        "anns_field": "emb",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 10,
        "data": [[0.1] * 4],
    }
    base.update(kwargs)
    return Prepare.search_requests_with_expr(**base)


class TestTopHits:
    def test_minimal(self):
        t = TopHits(size=3)
        assert t.size == 3
        assert t.sort == []

    def test_with_sort(self):
        t = TopHits(size=3, sort=[{"price": "asc"}, {"_score": "desc"}])
        assert t.sort == [{"price": "asc"}, {"_score": "desc"}]
        spec = t.to_proto()
        assert spec.size == 3
        assert [(s.field_name, s.direction) for s in spec.sort] == [
            ("price", "asc"),
            ("_score", "desc"),
        ]

    @pytest.mark.parametrize("bad", [0, -1, 1.0, True, "3"])
    def test_bad_size(self, bad):
        with pytest.raises(ParamError):
            TopHits(size=bad)

    def test_bad_sort_direction(self):
        with pytest.raises(ParamError):
            TopHits(size=3, sort=[{"price": "upward"}])

    def test_sort_multi_key_dict(self):
        with pytest.raises(ParamError):
            TopHits(size=3, sort=[{"a": "asc", "b": "desc"}])

    def test_sort_not_list(self):
        with pytest.raises(ParamError):
            TopHits(size=3, sort={"price": "asc"})


class TestSearchAggregation:
    def test_minimal(self):
        a = SearchAggregation(fields=["brand"], size=3)
        spec = a.to_proto()
        assert list(spec.fields) == ["brand"]
        assert spec.size == 3
        assert len(spec.metrics) == 0
        assert len(spec.order) == 0
        assert not spec.HasField("top_hits")
        assert not spec.HasField("sub_aggregation")

    def test_composite_fields(self):
        a = SearchAggregation(fields=["brand", "color"], size=3)
        assert list(a.to_proto().fields) == ["brand", "color"]

    def test_metrics_all_ops(self):
        a = SearchAggregation(
            fields=["x"],
            size=3,
            metrics={
                "a": {"avg": "price"},
                "s": {"sum": "rev"},
                "c": {"count": "*"},
                "mn": {"min": "price"},
                "mx": {"max": "_score"},
            },
        )
        spec = a.to_proto()
        assert {k: (v.op, v.field_name) for k, v in spec.metrics.items()} == {
            "a": ("avg", "price"),
            "s": ("sum", "rev"),
            "c": ("count", "*"),
            "mn": ("min", "price"),
            "mx": ("max", "_score"),
        }

    def test_order_and_special_keys(self):
        a = SearchAggregation(
            fields=["x"],
            size=3,
            metrics={"a": {"avg": "price"}},
            order=[{"a": "desc"}, {"_count": "desc"}, {"_key": "asc"}],
        )
        order = [(o.key, o.direction) for o in a.to_proto().order]
        assert order == [("a", "desc"), ("_count", "desc"), ("_key", "asc")]

    def test_nested_sub_aggregation(self):
        a = SearchAggregation(
            fields=["category"],
            size=3,
            sub_aggregation=SearchAggregation(
                fields=["brand"],
                size=2,
                top_hits=TopHits(size=1, sort=[{"price": "asc"}]),
            ),
        )
        spec = a.to_proto()
        assert list(spec.sub_aggregation.fields) == ["brand"]
        assert spec.sub_aggregation.size == 2
        assert spec.sub_aggregation.top_hits.size == 1
        assert spec.sub_aggregation.top_hits.sort[0].field_name == "price"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fields": [], "size": 5},
            {"fields": ["x"], "size": 0},
            {"fields": ["x"], "size": -1},
            {"fields": ["x"], "size": True},
            {"fields": "brand", "size": 5},
            {"fields": ["", "brand"], "size": 5},
        ],
    )
    def test_bad_basic_args(self, kwargs):
        with pytest.raises(ParamError):
            SearchAggregation(**kwargs)

    @pytest.mark.parametrize("bad_field", ["meta['region']", 'meta["x"]', "a[0]"])
    def test_json_path_field_rejected(self, bad_field):
        with pytest.raises(ParamError, match="JSON path"):
            SearchAggregation(fields=[bad_field], size=3)

    def test_bad_metric_op(self):
        with pytest.raises(ParamError):
            SearchAggregation(fields=["x"], size=3, metrics={"a": {"median": "price"}})

    def test_star_only_for_count(self):
        SearchAggregation(fields=["x"], size=3, metrics={"c": {"count": "*"}})  # ok
        with pytest.raises(ParamError):
            SearchAggregation(fields=["x"], size=3, metrics={"a": {"avg": "*"}})

    def test_order_key_unknown(self):
        with pytest.raises(ParamError):
            SearchAggregation(
                fields=["x"],
                size=3,
                metrics={"a": {"avg": "price"}},
                order=[{"nonexistent": "desc"}],
            )

    def test_order_direction_bad(self):
        with pytest.raises(ParamError):
            SearchAggregation(
                fields=["x"],
                size=3,
                metrics={"a": {"avg": "price"}},
                order=[{"a": "sideways"}],
            )

    def test_order_multi_key_dict(self):
        with pytest.raises(ParamError):
            SearchAggregation(
                fields=["x"],
                size=3,
                metrics={"a": {"avg": "p"}, "b": {"sum": "p"}},
                order=[{"a": "desc", "b": "asc"}],
            )

    def test_top_hits_type(self):
        with pytest.raises(ParamError):
            SearchAggregation(fields=["x"], size=3, top_hits={"size": 3})

    def test_sub_aggregation_type(self):
        with pytest.raises(ParamError):
            SearchAggregation(fields=["x"], size=3, sub_aggregation={"fields": ["y"]})

    def test_empty_order_list(self):
        a = SearchAggregation(fields=["x"], size=3, order=[])
        assert list(a.to_proto().order) == []

    def test_to_proto_idempotent(self):
        a = SearchAggregation(
            fields=["category"],
            size=3,
            metrics={"tot": {"sum": "price"}},
            order=[{"tot": "desc"}],
            top_hits=TopHits(size=2, sort=[{"_score": "desc"}]),
            sub_aggregation=SearchAggregation(fields=["brand"], size=2),
        )
        first = a.to_proto()
        second = a.to_proto()
        assert first.SerializeToString() == second.SerializeToString()

    def test_deep_nesting_proto_builds(self):
        inner = SearchAggregation(fields=["a"], size=1)
        for _ in range(5):
            inner = SearchAggregation(fields=["a"], size=1, sub_aggregation=inner)
        proto = inner.to_proto()
        depth = 1
        cursor = proto
        while cursor.HasField("sub_aggregation"):
            depth += 1
            cursor = cursor.sub_aggregation
        assert depth == 6


class TestPrepareIntegration:
    def test_search_aggregation_fills_proto_field(self):
        agg = _basic_agg()
        req = _prepare_search(search_aggregation=agg)
        assert req.HasField("search_aggregation")
        assert list(req.search_aggregation.fields) == ["brand"]
        assert req.search_aggregation.size == 5
        assert req.search_aggregation.metrics["avg_p"].op == "avg"
        assert req.search_aggregation.top_hits.size == 2

    def test_no_search_aggregation(self):
        req = _prepare_search()
        assert not req.HasField("search_aggregation")

    def test_mutex_with_group_by_field(self):
        with pytest.raises(ParamError, match="mutually exclusive"):
            _prepare_search(search_aggregation=_basic_agg(), group_by_field="brand")

    def test_wrong_type(self):
        with pytest.raises(ParamError, match="SearchAggregation instance"):
            _prepare_search(search_aggregation={"fields": ["brand"], "size": 3})

    def test_hybrid_search_rejects(self):
        with pytest.raises(ParamError, match="not supported in hybrid_search"):
            Prepare.hybrid_search_request_with_ranker(
                collection_name="c",
                reqs=[],
                rerank=None,
                limit=10,
                search_aggregation=_basic_agg(),
            )

    def test_nested_proto_roundtrip(self):
        agg = SearchAggregation(
            fields=["category"],
            size=3,
            metrics={"tot": {"sum": "price"}, "avg_s": {"avg": "_score"}},
            order=[{"tot": "desc"}, {"_count": "desc"}],
            top_hits=TopHits(size=2, sort=[{"_score": "desc"}]),
            sub_aggregation=SearchAggregation(
                fields=["brand", "color"],
                size=2,
                metrics={"ar": {"avg": "rating"}},
                order=[{"ar": "desc"}],
                top_hits=TopHits(size=3, sort=[{"price": "asc"}]),
            ),
        )
        req = _prepare_search(search_aggregation=agg)
        sub = req.search_aggregation.sub_aggregation
        assert list(sub.fields) == ["brand", "color"]
        assert sub.metrics["ar"].field_name == "rating"
        assert sub.top_hits.sort[0].field_name == "price"


def _make_bucket_proto(
    key_entries=(),
    count=0,
    metrics=None,
    hits=(),
    sub_groups=(),
):
    """Build a synthetic AggBucket. key_entries / hit.fields tuples are
    (field_id, value) or (field_id, field_name, value)."""
    b = schema_pb2.AggBucket(count=count)
    for entry in key_entries:
        fid, name, val = entry if len(entry) == 3 else (entry[0], "", entry[1])
        e = b.key.add()
        e.field_id = fid
        if name:
            e.field_name = name
        if isinstance(val, bool):
            e.bool_val = val
        elif isinstance(val, int):
            e.int_val = val
        elif isinstance(val, str):
            e.string_val = val
    if metrics:
        for alias, val in metrics.items():
            m = b.metrics[alias]
            if isinstance(val, bool):
                m.bool_val = val
            elif isinstance(val, int):
                m.int_val = val
            elif isinstance(val, float):
                m.double_val = val
            elif isinstance(val, str):
                m.string_val = val
    for h in hits:
        hp = b.hits.add()
        pk = h["pk"]
        if isinstance(pk, int):
            hp.int_pk = pk
        else:
            hp.str_pk = pk
        hp.score = h.get("score", 0.0)
        for entry in h.get("fields", ()):
            fid, name, val = entry if len(entry) == 3 else (entry[0], "", entry[1])
            fp = hp.fields.add()
            fp.field_id = fid
            if name:
                fp.field_name = name
            if isinstance(val, bool):
                fp.bool_val = val
            elif isinstance(val, int):
                fp.int_val = val
            elif isinstance(val, float):
                fp.double_val = val
            elif isinstance(val, str):
                fp.string_val = val
            elif isinstance(val, bytes):
                fp.bytes_val = val
    for sb in sub_groups:
        b.sub_groups.add().CopyFrom(sb)
    return b


class TestResponseParsing:
    def test_minimal_bucket(self):
        p = _make_bucket_proto(key_entries=[(100, "category", "cat1")], count=5)
        bkt = AggregationBucket(p)
        assert bkt.key == [{"field_name": "category", "field_id": 100, "value": "cat1"}]
        assert bkt.count == 5
        assert bkt.metrics == {}
        assert bkt.hits == []
        assert bkt.sub_groups == []

    def test_key_without_field_name_falls_through(self):
        # Legacy servers or malformed responses — field_name empty string, SDK
        # keeps field_id visible so consumers can still identify the field.
        p = _make_bucket_proto(key_entries=[(100, "cat1")], count=1)
        bkt = AggregationBucket(p)
        assert bkt.key == [{"field_name": "100", "field_id": 100, "value": "cat1"}]

    def test_composite_key_mixed_types(self):
        p = _make_bucket_proto(
            key_entries=[
                (100, "color", "red"),
                (101, "age", 42),
                (102, "active", True),
            ],
            count=1,
        )
        key = AggregationBucket(p).key
        assert [k["value"] for k in key] == ["red", 42, True]
        assert [k["field_name"] for k in key] == ["color", "age", "active"]

    def test_metrics_all_oneof_branches(self):
        p = _make_bucket_proto(
            count=1,
            metrics={"c": 10, "s": 99.5, "label": "hot", "flag": True},
        )
        assert AggregationBucket(p).metrics == {"c": 10, "s": 99.5, "label": "hot", "flag": True}

    def test_hits_int_and_str_pk(self):
        p = _make_bucket_proto(
            count=2,
            hits=[
                {
                    "pk": 123,
                    "score": 0.9,
                    "fields": [(200, "name", "name_a"), (201, "price", 3.14)],
                },
                {"pk": "abc", "score": 0.5, "fields": []},
            ],
        )
        hits = AggregationBucket(p).hits
        assert hits[0].pk == 123
        assert abs(hits[0].score - 0.9) < 1e-6
        assert hits[0].fields == {"name": "name_a", "price": 3.14}
        assert hits[0].field_ids() == {"name": 200, "price": 201}
        assert hits[1].pk == "abc"
        assert hits[1].fields == {}

    def test_hit_fields_all_branches(self):
        p = _make_bucket_proto(
            count=1,
            hits=[
                {
                    "pk": 1,
                    "fields": [
                        (1, "ivar", 10),
                        (2, "bvar", True),
                        (3, "fvar", 1.5),
                        (4, "svar", "s"),
                        (5, "bytesvar", b"\x00\x01"),
                    ],
                }
            ],
        )
        f = AggregationBucket(p).hits[0].fields
        assert f == {"ivar": 10, "bvar": True, "fvar": 1.5, "svar": "s", "bytesvar": b"\x00\x01"}

    def test_hit_field_without_name_uses_id_string(self):
        p = _make_bucket_proto(count=1, hits=[{"pk": 1, "fields": [(200, "found")]}])
        f = AggregationBucket(p).hits[0].fields
        assert f == {"200": "found"}
        assert AggregationBucket(p).hits[0].field_ids() == {"200": 200}

    def test_nested_sub_groups(self):
        leaf = _make_bucket_proto(key_entries=[(101, "brand", "brand_a")], count=3)
        root = _make_bucket_proto(
            key_entries=[(100, "category", "cat1")], count=5, sub_groups=[leaf]
        )
        bkt = AggregationBucket(root)
        assert len(bkt.sub_groups) == 1
        assert bkt.sub_groups[0].key[0]["field_name"] == "brand"
        assert bkt.sub_groups[0].key[0]["value"] == "brand_a"
        assert bkt.sub_groups[0].count == 3

    def test_deep_nesting(self):
        l3 = _make_bucket_proto(key_entries=[(102, "sku", "sku1")], count=1)
        l2 = _make_bucket_proto(key_entries=[(101, "brand", "b1")], count=2, sub_groups=[l3])
        l1 = _make_bucket_proto(key_entries=[(100, "category", "c1")], count=3, sub_groups=[l2])
        bkt = AggregationBucket(l1)
        leaf = bkt.sub_groups[0].sub_groups[0]
        assert leaf.key[0]["field_name"] == "sku"
        assert leaf.key[0]["value"] == "sku1"

    def test_search_result_exposes_agg_buckets_nq1(self):
        res = schema_pb2.SearchResultData(num_queries=1, primary_field_name="id")
        res.agg_buckets.add().CopyFrom(
            _make_bucket_proto(key_entries=[(100, "category", "x")], count=7)
        )
        sr = SearchResult(res)
        # outer list = nq, inner = buckets
        assert len(sr.agg_buckets) == 1
        assert len(sr.agg_buckets[0]) == 1
        assert sr.agg_buckets[0][0].count == 7

    def test_search_result_empty_agg_buckets(self):
        res = schema_pb2.SearchResultData(num_queries=1, primary_field_name="id")
        assert SearchResult(res).agg_buckets == []

    def test_search_result_splits_agg_buckets_by_nq(self):
        # nq=3: 2 buckets for query 0, 1 bucket for query 1, 3 for query 2
        res = schema_pb2.SearchResultData(num_queries=3, primary_field_name="id")
        labels = [
            ("q0_a", 1),
            ("q0_b", 2),
            ("q1_a", 3),
            ("q2_a", 4),
            ("q2_b", 5),
            ("q2_c", 6),
        ]
        for name, count in labels:
            res.agg_buckets.add().CopyFrom(
                _make_bucket_proto(key_entries=[(100, "cat", name)], count=count)
            )
        res.agg_topks.extend([2, 1, 3])
        sr = SearchResult(res)
        assert [len(b) for b in sr.agg_buckets] == [2, 1, 3]
        assert [b.count for b in sr.agg_buckets[0]] == [1, 2]
        assert sr.agg_buckets[1][0].count == 3
        assert [b.count for b in sr.agg_buckets[2]] == [4, 5, 6]

    def test_search_result_missing_agg_topks_falls_back_to_single_nq(self):
        # Older server responses may omit agg_topks; SDK groups all buckets under nq 0.
        res = schema_pb2.SearchResultData(num_queries=1, primary_field_name="id")
        for i in range(3):
            res.agg_buckets.add().CopyFrom(
                _make_bucket_proto(key_entries=[(100, "cat", f"x{i}")], count=i + 1)
            )
        sr = SearchResult(res)
        assert len(sr.agg_buckets) == 1
        assert [b.count for b in sr.agg_buckets[0]] == [1, 2, 3]

    def test_search_result_missing_agg_topks_multi_nq_warns_and_returns_empty_groups(self):
        res = schema_pb2.SearchResultData(num_queries=3, primary_field_name="id")
        for i in range(2):
            res.agg_buckets.add().CopyFrom(
                _make_bucket_proto(key_entries=[(100, "cat", f"x{i}")], count=i + 1)
            )
        with patch("pymilvus.client.search_result.logger.warning") as mock_warning:
            sr = SearchResult(res)
        assert sr.agg_buckets == [[], [], []]
        mock_warning.assert_called_once()
        assert "missing agg_topks" in mock_warning.call_args[0][0]

    def test_search_result_mismatched_agg_topks_warns_and_preserves_trailing_buckets(self):
        res = schema_pb2.SearchResultData(num_queries=2, primary_field_name="id")
        for i in range(4):
            res.agg_buckets.add().CopyFrom(
                _make_bucket_proto(key_entries=[(100, "cat", f"x{i}")], count=i + 1)
            )
        res.agg_topks.extend([2, 1])
        with patch("pymilvus.client.search_result.logger.warning") as mock_warning:
            sr = SearchResult(res)
        assert [len(buckets) for buckets in sr.agg_buckets] == [2, 2]
        assert [b.count for b in sr.agg_buckets[1]] == [3, 4]
        mock_warning.assert_called_once()
        assert "bucket count mismatch" in mock_warning.call_args[0][0]

    def test_accessors_return_copies(self):
        p = _make_bucket_proto(
            key_entries=[(1, "f1", "a")],
            count=1,
            metrics={"m": 1},
            hits=[{"pk": 1, "fields": [(2, "fv", "v")]}],
        )
        bkt = AggregationBucket(p)
        bkt.key.append({"field_name": "x", "field_id": 99, "value": "injected"})
        bkt.metrics["injected"] = 42
        bkt.hits.append("junk")
        bkt.sub_groups.append("junk")
        assert len(bkt.key) == 1
        assert "injected" not in bkt.metrics
        assert len(bkt.hits) == 1
        assert len(bkt.sub_groups) == 0
        bkt.hits[0].fields["injected"] = 1
        assert "injected" not in bkt.hits[0].fields
