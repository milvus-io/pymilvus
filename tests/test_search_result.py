import logging
import os
import random
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import orjson
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.search_result import (
    Hit,
    Hits,
    HybridHits,
    MilvusException,
    SearchResult,
    apply_valid_data,
    extract_array_row_data,
    extract_struct_field_value,
    get_field_data,
)
from pymilvus.client.types import DataType, HybridExtraList
from pymilvus.grpc_gen import common_pb2, schema_pb2

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field-builder helpers
# ---------------------------------------------------------------------------


def _make_scalar_field(dtype, name, data, field_id=0, is_dynamic=False, valid_data=None):
    """Build a schema_pb2.FieldData for scalar types."""
    kwargs = {"type": dtype, "field_name": name, "is_dynamic": is_dynamic}
    if field_id:
        kwargs["field_id"] = field_id
    if valid_data is not None:
        kwargs["valid_data"] = valid_data
    scalar_map = {
        DataType.BOOL: lambda d: schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=d)),
        DataType.INT8: lambda d: schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=d)),
        DataType.INT16: lambda d: schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=d)),
        DataType.INT32: lambda d: schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=d)),
        DataType.INT64: lambda d: schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=d)),
        DataType.FLOAT: lambda d: schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=d)),
        DataType.DOUBLE: lambda d: schema_pb2.ScalarField(
            double_data=schema_pb2.DoubleArray(data=d)
        ),
        DataType.VARCHAR: lambda d: schema_pb2.ScalarField(
            string_data=schema_pb2.StringArray(data=d)
        ),
        DataType.JSON: lambda d: schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=d)),
        DataType.GEOMETRY: lambda d: schema_pb2.ScalarField(
            geometry_wkt_data=schema_pb2.GeometryWktArray(data=d)
        ),
    }
    kwargs["scalars"] = scalar_map[dtype](data)
    return schema_pb2.FieldData(**kwargs)


def _make_array_field(element_dtype, name, row_scalars, field_id=0):
    """Build a schema_pb2.FieldData for ARRAY types."""
    kwargs = {"type": DataType.ARRAY, "field_name": name}
    if field_id:
        kwargs["field_id"] = field_id
    kwargs["scalars"] = schema_pb2.ScalarField(
        array_data=schema_pb2.ArrayArray(data=row_scalars, element_type=element_dtype)
    )
    return schema_pb2.FieldData(**kwargs)


def _make_vector_field(dtype, name, dim, data, field_id=0):
    """Build a schema_pb2.FieldData for vector types."""
    kwargs = {"type": dtype, "field_name": name}
    if field_id:
        kwargs["field_id"] = field_id
    vec_kwargs = {"dim": dim}
    if dtype == DataType.FLOAT_VECTOR:
        vec_kwargs["float_vector"] = schema_pb2.FloatArray(data=data)
    elif dtype == DataType.BINARY_VECTOR:
        vec_kwargs["binary_vector"] = data
    elif dtype == DataType.FLOAT16_VECTOR:
        vec_kwargs["float16_vector"] = data
    elif dtype == DataType.BFLOAT16_VECTOR:
        vec_kwargs["bfloat16_vector"] = data
    elif dtype == DataType.INT8_VECTOR:
        vec_kwargs["int8_vector"] = data
    kwargs["vectors"] = schema_pb2.VectorField(**vec_kwargs)
    return schema_pb2.FieldData(**kwargs)


class TestHit:
    @pytest.mark.parametrize(
        "pk_dist",
        [
            {"id": 1, "distance": 0.1, "entity": {}},
            {"id": 2, "distance": 0.3, "entity": {}},
            {"id": "a", "distance": 0.4, "entity": {}},
        ],
    )
    def test_hit_no_fields(self, pk_dist: Dict):
        h = Hit(pk_dist, pk_name="id")
        assert h.id == h["id"] == h.get("id") == pk_dist["id"]
        assert h.score == h.distance == h["distance"] == h.get("distance") == pk_dist["distance"]
        assert h.entity == h
        assert h["entity"] == h.get("entity") == {}
        assert hasattr(h, "id") is True
        assert hasattr(h, "distance") is True
        assert hasattr(h, "a_random_attribute") is False

    @pytest.mark.parametrize(
        "pk_dist_fields",
        [
            {
                "id": 1,
                "distance": 0.1,
                "entity": {
                    "vector": [1.0, 2.0, 3.0, 4.0],
                    "description": "This is a test",
                    "d_a": "dynamic a",
                },
            },
            {
                "id": 2,
                "distance": 0.3,
                "entity": {
                    "vector": [3.0, 4.0, 5.0, 6.0],
                    "description": "This is a test too",
                    "d_b": "dynamic b",
                },
            },
            {
                "id": "a",
                "distance": 0.4,
                "entity": {
                    "vector": [4.0, 4.0, 4.0, 4.0],
                    "description": "This is a third test",
                    "d_a": "dynamic a twice",
                },
            },
        ],
    )
    def test_hit_with_fields(self, pk_dist_fields: Dict):
        h = Hit(pk_dist_fields, pk_name="id")

        # fixed attributes
        assert h.id == pk_dist_fields["id"]
        assert h.id == h.get("id") == h["id"]
        assert h.score == pk_dist_fields["distance"]
        assert h.distance == h.score
        assert h.distance == h.get("distance") == h["distance"]
        assert h.entity == pk_dist_fields
        assert pk_dist_fields["entity"] == h.get("entity") == h["entity"]
        assert hasattr(h, "id") is True
        assert hasattr(h, "distance") is True

        # dynamic attributes
        assert h.description == pk_dist_fields["entity"].get("description")
        assert h.vector == pk_dist_fields["entity"].get("vector")
        assert hasattr(h, "description") is True
        assert hasattr(h, "vector") is True

        assert hasattr(h, "a_random_attribute") is False

        with pytest.raises(AttributeError):
            _ = h.field_not_exits

        LOGGER.info(h)


class TestSearchResult:
    @pytest.mark.parametrize(
        "pk",
        [
            schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6)))),
            schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i * 10) for i in range(6)])),
        ],
    )
    @pytest.mark.parametrize(
        "round_decimal",
        [
            None,
            -1,
            4,
        ],
    )
    def test_search_result_no_fields_data(self, pk, round_decimal):
        result = schema_pb2.SearchResultData(
            num_queries=2,
            top_k=3,
            scores=[1.0 * i for i in range(6)],
            ids=pk,
            topks=[3, 3],
        )
        r = SearchResult(result, round_decimal)

        # Iterable
        assert len(r) == 2
        for hits in r:
            assert isinstance(hits, (Hits, HybridHits))
            assert len(hits.ids) == 3
            assert len(hits.distances) == 3

        # slicable
        assert len(r[1:]) == 1
        first_q, _ = r[0], r[1]
        assert len(first_q) == 3
        assert len(first_q[:]) == 3
        assert len(first_q[1:]) == 2
        assert len(first_q[2:]) == 1
        assert len(first_q[3:]) == 0
        LOGGER.info(first_q[:])
        LOGGER.info(first_q[1:])
        LOGGER.info(first_q[2:])

        first_hit = first_q[0]
        LOGGER.info(first_hit)
        assert first_hit["distance"] == 0.0
        assert first_hit["entity"] == {}

    @pytest.mark.parametrize(
        "pk",
        [
            schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6)))),
            schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i * 10) for i in range(6)])),
        ],
    )
    def test_search_result_with_fields_data(self, pk):
        _int_rows = [schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=list(range(10))))]
        _long_rows = [schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=list(range(10))))]
        _float_rows = [
            schema_pb2.ScalarField(
                float_data=schema_pb2.FloatArray(data=[j * 1.0 for j in range(10)])
            )
        ]
        _str_rows = [
            schema_pb2.ScalarField(
                string_data=schema_pb2.StringArray(data=[str(j * 1.0) for j in range(10)])
            )
        ]
        fields_data = [
            _make_scalar_field(DataType.BOOL, "bool_field", [True] * 6, field_id=100),
            _make_scalar_field(DataType.INT8, "int8_field", list(range(6)), field_id=101),
            _make_scalar_field(DataType.INT16, "int16_field", list(range(6)), field_id=102),
            _make_scalar_field(DataType.INT32, "int32_field", list(range(6)), field_id=103),
            _make_scalar_field(DataType.INT64, "int64_field", list(range(6)), field_id=104),
            _make_scalar_field(
                DataType.FLOAT, "float_field", [i * 1.0 for i in range(6)], field_id=105
            ),
            _make_scalar_field(
                DataType.DOUBLE, "double_field", [i * 1.0 for i in range(6)], field_id=106
            ),
            _make_scalar_field(
                DataType.VARCHAR, "varchar_field", [str(i * 10) for i in range(6)], field_id=107
            ),
            _make_array_field(DataType.INT16, "int16_array_field", _int_rows * 6, field_id=108),
            _make_array_field(DataType.INT64, "int64_array_field", _long_rows * 6, field_id=109),
            _make_array_field(DataType.FLOAT, "float_array_field", _float_rows * 6, field_id=110),
            _make_array_field(DataType.VARCHAR, "varchar_array_field", _str_rows * 6, field_id=110),
            _make_scalar_field(
                DataType.JSON,
                "normal_json_field",
                [orjson.dumps({str(i): i for i in range(3)}) for i in range(6)],
                field_id=111,
            ),
            _make_scalar_field(
                DataType.JSON,
                "$meta",
                [orjson.dumps({str(i * 100): i}) for i in range(6)],
                field_id=112,
                is_dynamic=True,
            ),
            _make_vector_field(
                DataType.FLOAT_VECTOR,
                "float_vector_field",
                4,
                [random.random() for i in range(24)],
                field_id=113,
            ),
            _make_vector_field(
                DataType.BINARY_VECTOR, "binary_vector_field", 8, os.urandom(6), field_id=114
            ),
            _make_vector_field(
                DataType.FLOAT16_VECTOR, "float16_vector_field", 16, os.urandom(32), field_id=115
            ),
            _make_vector_field(
                DataType.BFLOAT16_VECTOR, "bfloat16_vector_field", 16, os.urandom(32), field_id=116
            ),
            _make_vector_field(
                DataType.INT8_VECTOR, "int8_vector_field", 16, os.urandom(32), field_id=117
            ),
        ]
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.0 * i for i in range(6)],
            ids=pk,
            topks=[3, 3],
            output_fields=["$meta"],
        )
        r = SearchResult(result)
        LOGGER.info(r[0])
        assert len(r) == 2
        assert 3 == len(r[0]) == len(r[1])
        assert r[0][0].get("entity").get("normal_json_field") == {"0": 0, "1": 1, "2": 2}
        # dynamic field
        assert r[0][1].get("entity").get("100") == 1

        assert r[0][0].get("entity").get("int32_field") == 0
        assert r[0][1].get("entity").get("int8_field") == 1
        assert r[0][2].get("entity").get("int16_field") == 2
        assert r[0][1].get("entity").get("int64_array_field") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert len(r[0][0].get("entity").get("bfloat16_vector_field")) == 32
        assert len(r[0][0].get("entity").get("float16_vector_field")) == 32
        assert len(r[0][0].get("entity").get("int8_vector_field")) == 16

    @pytest.mark.parametrize(
        "element_indices_data,expected_offsets",
        [
            ([10, 20, 30, 40, 50, 60], [[10, 20, 30], [40, 50, 60]]),  # with indices
            (None, None),  # without indices
        ],
        ids=["with_element_indices", "without_element_indices"],
    )
    def test_search_result_element_indices(self, element_indices_data, expected_offsets):
        """Test that element_indices are properly handled (present and absent)."""
        pk = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6))))
        result_kwargs = {
            "num_queries": 2,
            "top_k": 3,
            "scores": [1.0 * i for i in range(6)],
            "ids": pk,
            "topks": [3, 3],
        }
        if element_indices_data is not None:
            result_kwargs["element_indices"] = schema_pb2.LongArray(data=element_indices_data)
        r = SearchResult(schema_pb2.SearchResultData(**result_kwargs))

        assert len(r) == 2
        first_q = r[0]
        assert len(first_q) == 3
        hit = first_q[0]
        assert "id" in hit
        assert "distance" in hit
        assert "entity" in hit

        if expected_offsets is not None:
            assert "offset" in hit
            for i, off in enumerate(expected_offsets[0]):
                assert first_q[i]["offset"] == off
            second_q = r[1]
            for i, off in enumerate(expected_offsets[1]):
                assert second_q[i]["offset"] == off
        else:
            assert "offset" not in hit

        LOGGER.info(f"Hit: {hit}")


class TestHybridExtraList:
    def test_query_result_with_element_indices(self):
        """Test that pre-expanded rows with _original_idx are handled correctly."""
        # Simulate handler-level expansion: entity 0 has 2 matches, entity 1 has 1 match
        results = [
            {"pk": 1, "name": "entity1", "offset": 0, "_original_idx": 0},
            {"pk": 1, "name": "entity1", "offset": 2, "_original_idx": 0},
            {"pk": 2, "name": "entity2", "offset": 5, "_original_idx": 1},
        ]
        hybrid_list = HybridExtraList([], results, extra={"session_ts": 12345})

        assert len(hybrid_list) == 3
        row0 = hybrid_list[0]
        assert row0["pk"] == 1
        assert row0["offset"] == 0
        assert "_original_idx" not in row0

        row1 = hybrid_list[1]
        assert row1["pk"] == 1
        assert row1["offset"] == 2

        row2 = hybrid_list[2]
        assert row2["pk"] == 2
        assert row2["offset"] == 5

    def test_query_result_without_element_indices(self):
        """Test normal rows without element_indices expansion."""
        results = [
            {"pk": 1, "name": "entity1"},
            {"pk": 2, "name": "entity2"},
        ]
        hybrid_list = HybridExtraList([], results, extra={"session_ts": 12345})

        assert len(hybrid_list) == 2
        row0 = hybrid_list[0]
        assert row0["pk"] == 1
        assert "offset" not in row0
        assert "_original_idx" not in row0

    def test_original_idx_used_for_lazy_field_extraction(self):
        """Test that _original_idx is used (not the list index) for lazy field extraction."""
        # Simulate: 2 physical entities, expanded to 3 rows
        # Row 0 and 1 come from entity 0, row 2 comes from entity 1
        mock_field_data = MagicMock()
        results = [
            {"pk": None, "offset": 0, "_original_idx": 0},
            {"pk": None, "offset": 2, "_original_idx": 0},
            {"pk": None, "offset": 5, "_original_idx": 1},
        ]
        hybrid_list = HybridExtraList([mock_field_data], results, extra={})

        # Access row at index 2; lazy extraction should use _original_idx=1, not index=2
        _ = hybrid_list[2]
        assert hybrid_list._materialized_bitmap[2] is True
        assert hybrid_list[2]["offset"] == 5
        assert "_original_idx" not in hybrid_list[2]


class TestSearchResultExtended:
    def _create_base_result(self, fields_data=None):
        if fields_data is None:
            fields_data = []

        pk = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2, 3]))
        return schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=3,
            scores=[0.1, 0.2, 0.3],
            ids=pk,
            topks=[3],
            output_fields=["*"],
        )

    def test_sparse_float_vector(self, monkeypatch):
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.SPARSE_FLOAT_VECTOR,
                field_name="sparse_vec",
                field_id=101,
                vectors=schema_pb2.VectorField(
                    dim=100, sparse_float_vector=schema_pb2.SparseFloatArray()
                ),
            )
        ]

        res_data = self._create_base_result(fields_data)
        expected_rows = [{0: 1.0}, {1: 0.5}, {2: 0.2}]

        monkeypatch.setattr(
            entity_helper, "sparse_proto_to_rows", lambda x, start, end: expected_rows[start:end]
        )

        sr = SearchResult(res_data)
        assert len(sr) == 1
        hits = sr[0]
        assert len(hits) == 3
        assert hits[0].entity["sparse_vec"] == {0: 1.0}
        assert hits[1].entity["sparse_vec"] == {1: 0.5}
        assert hits[2].entity["sparse_vec"] == {2: 0.2}

        # Test with validity mask
        res_data.fields_data[0].valid_data.extend([True, False, True])
        valid_rows = [{0: 1.0}, {2: 0.2}]
        monkeypatch.setattr(
            entity_helper, "sparse_proto_to_rows", lambda x, start, end: valid_rows[start:end]
        )

        sr = SearchResult(res_data)
        hits = sr[0]
        assert hits[0].entity["sparse_vec"] == {0: 1.0}
        assert hits[1].entity["sparse_vec"] is None
        assert hits[2].entity["sparse_vec"] == {2: 0.2}

    def test_array_of_struct(self, monkeypatch):
        fields_data = [
            schema_pb2.FieldData(
                type=DataType._ARRAY_OF_STRUCT,
                field_name="struct_arr",
                field_id=102,
                struct_arrays=schema_pb2.StructArrayField(),
            )
        ]

        res_data = self._create_base_result(fields_data)
        expected_data = [
            [{"name": "a", "age": 10}],
            [{"name": "b", "age": 20}],
            [{"name": "c", "age": 30}],
        ]

        monkeypatch.setattr(
            entity_helper,
            "extract_struct_array_from_column_data",
            lambda x, idx: expected_data[idx],
        )

        sr = SearchResult(res_data)
        hits = sr[0]
        assert hits[0].entity["struct_arr"] == [{"name": "a", "age": 10}]
        assert hits[1].entity["struct_arr"] == [{"name": "b", "age": 20}]
        assert hits[2].entity["struct_arr"] == [{"name": "c", "age": 30}]

    def test_array_of_vector(self):
        vec_arr = schema_pb2.VectorArray()

        # Row 0: 2 vectors of dim 2
        v0 = schema_pb2.VectorField(dim=2)
        v0.float_vector.data.extend([1.0, 2.0, 3.0, 4.0])
        vec_arr.data.append(v0)

        # Row 1: 1 vector of dim 2
        v1 = schema_pb2.VectorField(dim=2)
        v1.float_vector.data.extend([5.0, 6.0])
        vec_arr.data.append(v1)

        # Row 2: empty
        v2 = schema_pb2.VectorField(dim=2)
        vec_arr.data.append(v2)

        fields_data = [
            schema_pb2.FieldData(
                type=DataType._ARRAY_OF_VECTOR,
                field_name="vec_arr",
                field_id=103,
                vectors=schema_pb2.VectorField(vector_array=vec_arr),
            )
        ]

        res_data = self._create_base_result(fields_data)

        sr = SearchResult(res_data)
        hits = sr[0]

        assert hits[0].entity["vec_arr"] == [[1.0, 2.0], [3.0, 4.0]]
        assert hits[1].entity["vec_arr"] == [[5.0, 6.0]]
        assert hits[2].entity["vec_arr"] == []

    def test_geometry(self):
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.GEOMETRY,
                field_name="geom",
                field_id=104,
                scalars=schema_pb2.ScalarField(
                    geometry_wkt_data=schema_pb2.GeometryWktArray(
                        data=["POINT(1 1)", "POINT(2 2)", "POINT(3 3)"]
                    )
                ),
            )
        ]

        res_data = self._create_base_result(fields_data)
        sr = SearchResult(res_data)
        hits = sr[0]

        assert hits[0].entity["geom"] == "POINT(1 1)"
        assert hits[1].entity["geom"] == "POINT(2 2)"

    def test_search_result_extra_and_status(self):
        res_data = self._create_base_result()

        status = common_pb2.Status(
            code=0,
            extra_info={
                "report_value": "100",  # cost
                "scanned_remote_bytes": "2000",
                "scanned_total_bytes": "3000",
                "cache_hit_ratio": "0.5",
            },
        )

        res_data.recalls.extend([0.9])

        sr = SearchResult(res_data, status=status, session_ts=12345)

        assert sr.extra["cost"] == 100
        assert sr.extra["scanned_remote_bytes"] == 2000
        assert sr.extra["scanned_total_bytes"] == 3000
        assert sr.extra["cache_hit_ratio"] == 0.5
        assert sr.recalls == pytest.approx([0.9])
        assert sr.get_session_ts() == 12345

        # Test __str__ output with extras
        s = str(sr)
        assert "cost" in s
        assert "recalls" in s

    def test_null_values_all_scalar_types(self):
        # Create fields with some null values (valid_data has False)
        # Using a subset of types representing different branches

        valid_data = [True, False, True]

        # INT32
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.INT32,
                field_name="int_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(
                    int_data=schema_pb2.IntArray(data=[10, 0, 30])
                ),  # 0 is placeholder for null
            ),
            # FLOAT
            schema_pb2.FieldData(
                type=DataType.FLOAT,
                field_name="float_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(
                    float_data=schema_pb2.FloatArray(data=[1.1, 0.0, 3.3])
                ),
            ),
            # VARCHAR
            schema_pb2.FieldData(
                type=DataType.VARCHAR,
                field_name="str_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(
                    string_data=schema_pb2.StringArray(data=["a", "", "c"])
                ),
            ),
            # JSON
            schema_pb2.FieldData(
                type=DataType.JSON,
                field_name="json_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(
                    json_data=schema_pb2.JSONArray(data=[b'{"a":1}', b"", b'{"c":3}'])
                ),
            ),
        ]

        res_data = self._create_base_result(fields_data)
        sr = SearchResult(res_data)
        hits = sr[0]

        # Row 1 (valid)
        assert hits[0].entity["int_field"] == 10
        assert hits[0].entity["float_field"] == pytest.approx(1.1)
        assert hits[0].entity["str_field"] == "a"
        assert hits[0].entity["json_field"] == {"a": 1}

        # Row 2 (null)
        assert hits[1].entity["int_field"] is None
        assert hits[1].entity["float_field"] is None
        assert hits[1].entity["str_field"] is None
        assert hits[1].entity["json_field"] is None

        # Row 3 (valid)
        assert hits[2].entity["int_field"] == 30

    def test_json_error_handling(self):
        # Test malformed JSON
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.JSON,
                field_name="bad_json",
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b"{bad"])),
            )
        ]
        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=1,
            scores=[0.1],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            topks=[1],
        )

        sr = SearchResult(res_data)
        with pytest.raises(orjson.JSONDecodeError):
            _ = sr[0][0].entity  # Trigger materialization

    def test_large_result_str(self):
        # Create result with > 10 items
        count = 15
        pk = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(count))))
        res_data = schema_pb2.SearchResultData(
            num_queries=1, top_k=count, scores=[0.1] * count, ids=pk, topks=[count]
        )

        sr = SearchResult(res_data)
        s = str(sr[0])
        assert "entities remaining" in s
        assert str(count - 10) in s

    def test_vectors_optimization(self):
        # Test the "direct return" optimization for FLOAT_VECTOR

        data = [1.0, 2.0, 3.0, 4.0]
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.FLOAT_VECTOR,
                field_name="vec",
                field_id=101,
                vectors=schema_pb2.VectorField(
                    dim=2, float_vector=schema_pb2.FloatArray(data=data)
                ),
            )
        ]
        # 2 vectors of dim 2
        pk = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2]))
        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data, num_queries=1, top_k=2, scores=[0.1, 0.2], ids=pk, topks=[2]
        )

        sr = SearchResult(res_data)
        # Accessing .entity will materialize, but we want to verify the internal optimization?
        # It's hard to verify internal optimization reference without mocking or inspecting memory
        # But we can at least ensure it works correctly.
        hits = sr[0]
        assert hits[0].entity["vec"] == [1.0, 2.0]
        assert hits[1].entity["vec"] == [3.0, 4.0]


class TestHitsLegacy:
    """Test the Hits and Hit classes (legacy path coverage)"""

    def test_hit_legacy_properties(self):

        h = Hit({"id": 1, "distance": 0.5, "entity": {"a": 1}}, pk_name="id")

        # Test legacy patches
        assert h.id == 1
        assert h.pk == 1
        assert h.distance == 0.5
        assert h.score == 0.5
        assert h.fields == {"a": 1}
        assert h.to_dict() == h

        # Test __getitem__
        assert h["a"] == 1
        assert h["distance"] == 0.5

        # Test __getattr__
        assert h.entity == h
        with pytest.raises(AttributeError):
            _ = h.invalid_attr

        # Test weak highlight
        assert h.highlight is None

    def test_hits_init_and_access(self):
        # Manual construction of arguments for Hits

        field_meta = schema_pb2.FieldData(type=DataType.INT32, field_name="age")
        fields = {"age": ([10, 20], field_meta)}

        hits = Hits(
            topk=2,
            pks=[1, 2],
            distances=[0.1, 0.2],
            fields=fields,
            output_fields=["age", "extra"],
            pk_name="id",
        )

        assert len(hits) == 2
        assert hits[0].id == 1
        assert hits[0].entity["age"] == 10
        assert hits[1].id == 2
        assert hits[1].entity["age"] == 20

        # Check repr
        assert "entities remaining" not in str(hits)

        # Test dynamic fields logic
        # If we have JSON dynamic field
        json_meta = schema_pb2.FieldData(type=DataType.JSON, field_name="meta", is_dynamic=True)
        fields = {"meta": ([{"d1": 1}, {"d1": 2}], json_meta)}
        hits = Hits(
            topk=2,
            pks=[1, 2],
            distances=[0.1, 0.2],
            fields=fields,
            output_fields=["d1"],  # d1 requested
            pk_name="id",
        )
        assert hits[0].entity["d1"] == 1


class TestGetFieldsByRange:
    """Test SearchResult._get_fields_by_range"""

    def test_get_fields_all_types(self):
        # It's stateless regarding instance data, it just uses the args
        res = SearchResult(schema_pb2.SearchResultData())
        count = 5

        _int_row = schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[1]))
        all_fields_data = [
            _make_scalar_field(DataType.BOOL, "bool", [True] * count),
            _make_scalar_field(DataType.INT32, "int32", list(range(count))),
            _make_scalar_field(DataType.INT64, "int64", list(range(count))),
            _make_scalar_field(DataType.FLOAT, "float", [float(i) for i in range(count)]),
            _make_scalar_field(DataType.DOUBLE, "double", [float(i) for i in range(count)]),
            _make_scalar_field(DataType.VARCHAR, "varchar", [str(i) for i in range(count)]),
            _make_scalar_field(DataType.JSON, "json", [b"{}"] * count),
            _make_scalar_field(DataType.GEOMETRY, "geom", ["POINT(0 0)"] * count),
            _make_array_field(DataType.INT32, "array", [_int_row] * count),
            # ARRAY_OF_STRUCT — mocked helper
            schema_pb2.FieldData(type=DataType._ARRAY_OF_STRUCT, field_name="aos"),
            _make_vector_field(DataType.FLOAT_VECTOR, "fv", 2, [0.0] * (count * 2)),
            _make_vector_field(DataType.BINARY_VECTOR, "bv", 8, b"\x00" * count),
            _make_vector_field(DataType.FLOAT16_VECTOR, "f16v", 2, b"\x00" * (count * 2 * 2)),
            _make_vector_field(DataType.BFLOAT16_VECTOR, "bf16v", 2, b"\x00" * (count * 2 * 2)),
            _make_vector_field(DataType.INT8_VECTOR, "i8v", 2, b"\x00" * (count * 2)),
            schema_pb2.FieldData(
                type=DataType.SPARSE_FLOAT_VECTOR,
                field_name="sv",
                vectors=schema_pb2.VectorField(
                    sparse_float_vector=schema_pb2.SparseFloatArray(contents=[b""] * count)
                ),
            ),
        ]

        original_struct_func = entity_helper.extract_struct_array_from_column_data
        original_sparse_func = entity_helper.sparse_proto_to_rows

        try:
            entity_helper.extract_struct_array_from_column_data = lambda x, y: {}
            entity_helper.sparse_proto_to_rows = lambda x, start, end: [{}] * (end - start)

            result = res._get_fields_by_range(0, count, all_fields_data)

            expected_keys = [
                "bool",
                "int32",
                "int64",
                "float",
                "double",
                "varchar",
                "json",
                "geom",
                "array",
                "aos",
                "fv",
                "bv",
                "f16v",
                "bf16v",
                "i8v",
                "sv",
            ]
            for key in expected_keys:
                assert key in result
            assert len(result["bool"][0]) == count

        finally:
            entity_helper.extract_struct_array_from_column_data = original_struct_func
            entity_helper.sparse_proto_to_rows = original_sparse_func

    def test_get_fields_optimized_float_vector(self):
        """Test the 25% perf optimization path for float vector"""
        res = SearchResult(schema_pb2.SearchResultData())
        fd = schema_pb2.FieldData(
            type=DataType.FLOAT_VECTOR,
            field_name="fv",
            vectors=schema_pb2.VectorField(
                dim=2, float_vector=schema_pb2.FloatArray(data=[1.0, 2.0, 3.0, 4.0])
            ),
        )

        # Full range
        result = res._get_fields_by_range(0, 2, [fd])
        # It calls directly return data
        assert result["fv"][0] == [1.0, 2.0, 3.0, 4.0]


class TestHelpers:
    """Test standalone helper functions in search_result.py"""

    def test_extract_array_row_data_none(self):
        assert extract_array_row_data([None, None], DataType.INT64) == [None, None]

    @pytest.mark.parametrize(
        "scalar_field,dtype,expected",
        [
            (
                schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[1, 2])),
                DataType.INT64,
                [[1, 2]],
            ),
            (
                schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True, False])),
                DataType.BOOL,
                [[True, False]],
            ),
            (
                schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[1, 2])),
                DataType.INT32,
                [[1, 2]],
            ),
            (
                schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[1.0, 2.0])),
                DataType.FLOAT,
                [[1.0, 2.0]],
            ),
            (
                schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[1.0, 2.0])),
                DataType.DOUBLE,
                [[1.0, 2.0]],
            ),
            (
                schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=["a", "b"])),
                DataType.VARCHAR,
                [["a", "b"]],
            ),
        ],
    )
    def test_extract_array_row_data(self, scalar_field, dtype, expected):
        assert extract_array_row_data([scalar_field], dtype) == expected

    def test_apply_valid_data(self):
        # Both data and valid_data are pre-sliced by caller; same length required.
        data = [1, 2, 3]
        valid_data = [True, False, True]

        res = apply_valid_data(data, valid_data)
        assert res == [1, None, 3]

        # Partial range: caller passes pre-sliced data and valid_data
        data = [2, 3]
        valid_data_slice = [False, True]
        res = apply_valid_data(data, valid_data_slice)
        assert res == [None, 3]

        # None valid_data
        res = apply_valid_data([1, 2], None)
        assert res == [1, 2]

    @pytest.mark.parametrize(
        "fd,idx,expected",
        [
            (
                schema_pb2.FieldData(
                    type=DataType.INT32,
                    scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[10, 20])),
                ),
                0,
                10,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.INT32,
                    scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[10, 20])),
                ),
                1,
                20,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.INT32,
                    scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[10, 20])),
                ),
                2,
                None,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.INT64,
                    scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[10])),
                ),
                0,
                10,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.DOUBLE,
                    scalars=schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[1.0])),
                ),
                0,
                1.0,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.BOOL,
                    scalars=schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True])),
                ),
                0,
                True,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.VARCHAR,
                    scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=["s"])),
                ),
                0,
                "s",
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.JSON,
                    scalars=schema_pb2.ScalarField(
                        json_data=schema_pb2.JSONArray(data=[b'{"a":1}'])
                    ),
                ),
                0,
                {"a": 1},
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.FLOAT_VECTOR,
                    vectors=schema_pb2.VectorField(
                        dim=2, float_vector=schema_pb2.FloatArray(data=[1.0, 2.0, 3.0, 4.0])
                    ),
                ),
                0,
                [1.0, 2.0],
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.FLOAT_VECTOR,
                    vectors=schema_pb2.VectorField(
                        dim=2, float_vector=schema_pb2.FloatArray(data=[1.0, 2.0, 3.0, 4.0])
                    ),
                ),
                1,
                [3.0, 4.0],
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.FLOAT_VECTOR,
                    vectors=schema_pb2.VectorField(
                        dim=2, float_vector=schema_pb2.FloatArray(data=[1.0, 2.0, 3.0, 4.0])
                    ),
                ),
                2,
                None,
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.BINARY_VECTOR,
                    vectors=schema_pb2.VectorField(dim=8, binary_vector=b"\x01\x02"),
                ),
                0,
                b"\x01",
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.BINARY_VECTOR,
                    vectors=schema_pb2.VectorField(dim=8, binary_vector=b"\x01\x02"),
                ),
                1,
                b"\x02",
            ),
            (
                schema_pb2.FieldData(
                    type=DataType.BINARY_VECTOR,
                    vectors=schema_pb2.VectorField(dim=8, binary_vector=b"\x01\x02"),
                ),
                2,
                None,
            ),
            (schema_pb2.FieldData(type=DataType.INT8), 0, None),
        ],
    )
    def test_extract_struct_field_value(self, fd, idx, expected):
        assert extract_struct_field_value(fd, idx) == expected

    def test_extract_struct_field_value_float_type(self):
        # FLOAT returns np.single — verify the type separately
        fd = schema_pb2.FieldData(
            type=DataType.FLOAT,
            scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[1.0])),
        )
        val = extract_struct_field_value(fd, 0)
        assert val == 1.0
        assert isinstance(val, (float, np.floating))


class TestCoverageEdgeCases:
    """Targeted tests for remaining coverage gaps"""

    def test_hit_get_exception(self):
        # target 805-807: KeyError handling in get

        h = Hit({}, pk_name="id")
        # Access key that doesn't exist to trigger KeyError caught by get
        assert h.get("non_existent", "default") == "default"

    def test_search_result_materialize(self):
        # target 360-361: SearchResult.materialize
        res_data = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=1,
            topks=[1],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            scores=[0.1],
        )
        sr = SearchResult(res_data)
        sr.materialize()  # Just call it
        assert sr[0].has_materialized

    def test_get_field_data_exception(self):
        # target 620-621

        fd = schema_pb2.FieldData(type=999)
        with pytest.raises(MilvusException):
            get_field_data(fd)

    def test_hybrid_hits_unsupported_type(self):
        # target 96-97

        fd = schema_pb2.FieldData(type=999, field_name="f", field_id=1)

        with pytest.raises(MilvusException):
            HybridHits(0, 1, [1], [0.1], [fd], [], [], "id")

    def test_materialize_struct_else(self):
        # target 251-253
        # DataType._ARRAY_OF_STRUCT but get_field_data returns something without fields? or None?
        # get_field_data returns struct_arrays.
        # If struct_arrays doesn't have fields (empty), it enters loop but what?
        # The check is: if hasattr(struct_arrays, "fields")
        # So we pass a mock that doesn't have "fields"

        fd = schema_pb2.FieldData(type=DataType._ARRAY_OF_STRUCT, field_name="aos", field_id=1)
        # Mock get_field_data to return object without fields

        with patch("pymilvus.client.search_result.get_field_data") as mock_get:
            mock_get.return_value = MagicMock(spec=[])  # No fields attr

            hh = HybridHits(0, 1, [1], [0.1], [fd], [], [], "id")
            hh.materialize()
            assert hh[0].entity["aos"] is None

    def test_materialize_vector_else(self):
        # target 274: idx < len(vector_array.data) is False

        vec_arr = schema_pb2.VectorArray()
        # Empty vector array

        fd = schema_pb2.FieldData(
            type=DataType._ARRAY_OF_VECTOR,
            field_name="aov",
            field_id=1,
            vectors=schema_pb2.VectorField(vector_array=vec_arr),
        )

        hh = HybridHits(0, 1, [1], [0.1], [fd], [], [], "id")
        hh.materialize()
        # idx=0, len=0 -> branch else
        assert hh[0].entity["aov"] == []

    def test_get_fields_by_range_json_error(self):
        # target 488-494
        res = SearchResult(schema_pb2.SearchResultData())

        # Valid data mask true, but content invalid json
        fd = schema_pb2.FieldData(
            type=DataType.JSON,
            field_name="json",
            valid_data=[True],
            scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b"{bad"])),
        )

        with pytest.raises(orjson.JSONDecodeError):
            res._get_fields_by_range(0, 1, [fd])

    def test_iterator_v2_info(self):
        # target 582
        res = SearchResult(schema_pb2.SearchResultData())
        info = res.get_search_iterator_v2_results_info()

        # It returns the raw protobuf message SearchIteratorV2Results
        assert info.token == ""
        assert info.last_bound == 0.0

    def test_hits_legacy_iteration(self):

        json_fd = schema_pb2.FieldData(type=DataType.JSON, field_name="meta", is_dynamic=True)
        vec_fd = schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, field_name="emb")
        vec_fd.vectors.dim = 2
        vec_fd.vectors.float_vector.data.extend([0.1, 0.2])

        fields = {"meta": ([{"a": 1}], json_fd), "emb": ([0.1, 0.2], vec_fd)}

        # Initialize Hits
        hits = Hits(topk=1, pks=[1], distances=[0.0], fields=fields, output_fields=[], pk_name="id")

        for h in hits:
            # Check dynamic merge (should happen if type matches)
            # If it fails to merge, it puts 'meta' in entity.
            # Use get() because Hit might not support __contains__ properly
            if h.get("meta") is not None:
                # It failed to merge, so we are covering the 'else' branch (698)
                assert h.get("meta") == {"a": 1}
            else:
                # It merged, covering 675
                assert h.get("a") == 1

            # Check vector branch (684)
            # If logic works, 'emb' is in entity.
            assert h.get("emb") is not None
            assert h.get("emb") == [0.1, 0.2]


class TestLegacyDetails:
    def test_hybrid_iter(self):
        # Cover __iter__ implicit materialization
        # Create a simple result that needs materialization
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.JSON,
                field_name="j",
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b"{}"])),
            )
        ]
        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=1,
            scores=[0.1],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            topks=[1],
        )
        sr = SearchResult(res_data)
        hits = sr[0]
        assert hits.has_materialized is False

        # Iteration should trigger materialization
        for _hit in hits:
            pass
        assert hits.has_materialized is True

    def test_physical_index_cache(self):
        # Cover _get_physical_index cache logic
        # We need a field that has validity
        # JSON does NOT use _get_physical_index in current implementation (it assumes dense).
        # We MUST use a vector type to test this cache.

        fields_data = [
            schema_pb2.FieldData(
                type=DataType.FLOAT_VECTOR,
                field_name="fv",
                valid_data=[True, False, True],  # 3 rows, 2 valid
                vectors=schema_pb2.VectorField(
                    dim=2,
                    float_vector=schema_pb2.FloatArray(
                        data=[1.0, 1.0, 2.0, 2.0]
                    ),  # 2 vectors (sparse storage)
                ),
            )
        ]

        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=3,
            scores=[0.1, 0.2, 0.3],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2, 3])),
            topks=[3],
        )

        sr = SearchResult(res_data)
        hits = sr[0]

        # Trigger use of _get_physical_index via materialize
        hits.materialize()

        # Check if cache is created
        assert hasattr(hits, "_prefix_sum_cache")
        assert len(hits._prefix_sum_cache) > 0

        # Verify correctness
        # Logical 0 (True) -> Physical 0
        # Logical 1 (False) -> Physical 0 (but shouldn't be accessed for data)
        # Logical 2 (True) -> Physical 1

        field_id = id(hits.lazy_field_data[0])

        prefix_sum = hits._prefix_sum_cache[field_id]
        assert prefix_sum[0] == 0
        assert prefix_sum[2] == 1

        # Verify data access correct
        assert hits[0].entity["fv"] == pytest.approx([1.0, 1.0])
        assert hits[1].entity["fv"] is None
        assert hits[2].entity["fv"] == pytest.approx([2.0, 2.0])
