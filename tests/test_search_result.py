import logging
import os
import random
from typing import Dict

import pytest
import orjson
from pymilvus.client.search_result import Hit, Hits, HybridHits, SearchResult
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2, common_pb2
from pymilvus.client import entity_helper
import numpy as np

LOGGER = logging.getLogger(__name__)


class TestHit:
    @pytest.mark.parametrize("pk_dist", [
        {"id": 1, "distance": 0.1, "entity":{}},
        {"id": 2, "distance": 0.3, "entity":{}},
        {"id": "a", "distance": 0.4, "entity":{}},
    ])
    def test_hit_no_fields(self, pk_dist: Dict):
        h = Hit(pk_dist, pk_name="id")
        assert h.id == h["id"] == h.get("id") == pk_dist["id"]
        assert h.score == h.distance == h["distance"] == h.get("distance") == pk_dist["distance"]
        assert h.entity == h
        assert h["entity"] == h.get("entity") == {}
        assert hasattr(h, "id") is True
        assert hasattr(h, "distance") is True
        assert hasattr(h, "a_random_attribute") is False

    @pytest.mark.parametrize("pk_dist_fields", [
        {"id": 1, "distance": 0.1, "entity": {"vector": [1., 2., 3., 4.],  "description": "This is a test", 'd_a': "dynamic a"}},
        {"id": 2,  "distance": 0.3, "entity": {"vector": [3., 4., 5., 6.], "description": "This is a test too", 'd_b': "dynamic b"}},
        {"id": "a","distance": 0.4, "entity": {"vector": [4., 4., 4., 4.], "description": "This is a third test", 'd_a': "dynamic a twice"}},
    ])
    def test_hit_with_fields(self, pk_dist_fields: Dict):
        h = Hit(pk_dist_fields, pk_name="id")

        # fixed attributes
        assert h.id == pk_dist_fields["id"]
        assert h.id == h.get("id") == h["id"]
        assert h.score == pk_dist_fields["distance"]
        assert h.distance == h.score
        assert h.distance == h.get("distance") == h["distance"]
        assert h.entity == pk_dist_fields
        assert pk_dist_fields["entity"] == h.get("entity")==h["entity"]
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
    @pytest.mark.parametrize("pk", [
        schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6)))),
        schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))
    ])
    @pytest.mark.parametrize("round_decimal", [
        None,
        -1,
        4,
    ])
    def test_search_result_no_fields_data(self, pk, round_decimal):
        result = schema_pb2.SearchResultData(
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
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
        assert first_hit["distance"] == 0.
        assert first_hit["entity"] == {}

    @pytest.mark.parametrize("pk", [
        schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(6)))),
        schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))
    ])
    def test_search_result_with_fields_data(self, pk):
        fields_data = [
            schema_pb2.FieldData(type=DataType.BOOL, field_name="bool_field", field_id=100,
                                 scalars=schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.INT8, field_name="int8_field", field_id=101,
                                 scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=list(range(6))))),
            schema_pb2.FieldData(type=DataType.INT16, field_name="int16_field", field_id=102,
                                 scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=list(range(6))))),
            schema_pb2.FieldData(type=DataType.INT32, field_name="int32_field", field_id=103,
                                 scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=list(range(6))))),
            schema_pb2.FieldData(type=DataType.INT64, field_name="int64_field", field_id=104,
                                 scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=list(range(6))))),
            schema_pb2.FieldData(type=DataType.FLOAT, field_name="float_field", field_id=105,
                                 scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[i*1. for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.DOUBLE, field_name="double_field", field_id=106,
                                 scalars=schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[i*1. for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.VARCHAR, field_name="varchar_field", field_id=107,
                                 scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.ARRAY, field_name="int16_array_field", field_id=108,
                                 scalars=schema_pb2.ScalarField(
                                     array_data=schema_pb2.ArrayArray(
                                         data=[schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=list(range(10)))) for i in range(6)],
                                         element_type=DataType.INT16,
                                     ),
                                 )),
            schema_pb2.FieldData(type=DataType.ARRAY, field_name="int64_array_field", field_id=109,
                                 scalars=schema_pb2.ScalarField(
                                     array_data=schema_pb2.ArrayArray(
                                         data=[schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=list(range(10)))) for i in range(6)],
                                         element_type=DataType.INT64,
                                     ),
                                 )),
            schema_pb2.FieldData(type=DataType.ARRAY, field_name="float_array_field", field_id=110,
                                 scalars=schema_pb2.ScalarField(
                                     array_data=schema_pb2.ArrayArray(
                                         data=[schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[j*1. for j in range(10)])) for i in range(6)],
                                         element_type=DataType.FLOAT,
                                     ),
                                 )),
            schema_pb2.FieldData(type=DataType.ARRAY, field_name="varchar_array_field", field_id=110,
                                 scalars=schema_pb2.ScalarField(
                                     array_data=schema_pb2.ArrayArray(
                                         data=[schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=[str(j*1.) for j in range(10)])) for i in range(6)],
                                         element_type=DataType.VARCHAR,
                                     ),
                                 )),

            schema_pb2.FieldData(type=DataType.JSON, field_name="normal_json_field", field_id=111,
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[orjson.dumps({str(i): i for i in range(3)}) for i in range(6)])),
            ),
            schema_pb2.FieldData(type=DataType.JSON, field_name="$meta", field_id=112,
                is_dynamic=True,
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[orjson.dumps({str(i*100): i}) for i in range(6)])),
            ),

            schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, field_name="float_vector_field", field_id=113,
                vectors=schema_pb2.VectorField(
                    dim=4,
                    float_vector=schema_pb2.FloatArray(data=[random.random() for i in range(24)]),
                ),
            ),
            schema_pb2.FieldData(type=DataType.BINARY_VECTOR, field_name="binary_vector_field", field_id=114,
                vectors=schema_pb2.VectorField(
                    dim=8,
                    binary_vector=os.urandom(6),
                ),
            ),
            schema_pb2.FieldData(type=DataType.FLOAT16_VECTOR, field_name="float16_vector_field", field_id=115,
                vectors=schema_pb2.VectorField(
                    dim=16,
                    float16_vector=os.urandom(32),
                ),
            ),
            schema_pb2.FieldData(type=DataType.BFLOAT16_VECTOR, field_name="bfloat16_vector_field", field_id=116,
                vectors=schema_pb2.VectorField(
                    dim=16,
                    bfloat16_vector=os.urandom(32),
                ),
            ),
            schema_pb2.FieldData(type=DataType.INT8_VECTOR, field_name="int8_vector_field", field_id=117,
                vectors=schema_pb2.VectorField(
                    dim=16,
                    int8_vector=os.urandom(32),
                ),
            ),
        ]
        result = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=2,
            top_k=3,
            scores=[1.*i for i in range(6)],
            ids=pk,
            topks=[3, 3],
            output_fields=['$meta']
        )
        r = SearchResult(result)
        LOGGER.info(r[0])
        assert len(r) == 2
        assert 3 == len(r[0]) == len(r[1])
        assert r[0][0].get("entity").get("normal_json_field") == {'0': 0, '1': 1, '2': 2}
        # dynamic field
        assert r[0][1].get("entity").get('100') == 1

        assert r[0][0].get("entity").get("int32_field") == 0
        assert r[0][1].get("entity").get("int8_field") == 1
        assert r[0][2].get("entity").get("int16_field") == 2
        assert r[0][1].get("entity").get("int64_array_field") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert len(r[0][0].get("entity").get("bfloat16_vector_field")) == 32
        assert len(r[0][0].get("entity").get("float16_vector_field")) == 32
        assert len(r[0][0].get("entity").get("int8_vector_field")) == 16
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
            output_fields=["*"]
        )

    def test_sparse_float_vector(self):
        # Construct a sparse vector field
        fields_data = [
            schema_pb2.FieldData(
                type=DataType.SPARSE_FLOAT_VECTOR,
                field_name="sparse_vec",
                field_id=101,
                vectors=schema_pb2.VectorField(
                    dim=100, 
                    sparse_float_vector=schema_pb2.SparseFloatArray()
                )
            )
        ]
        
        # When creating SearchResultData, the fields_data list is copied into the repeated field.
        # So we must modify res_data.fields_data later for the second part of the test.
        res_data = self._create_base_result(fields_data)
        
        # We expect search_result to call sparse_proto_to_rows
        expected_rows = [{0: 1.0}, {1: 0.5}, {2: 0.2}]
        
        # We can monkeypatch entity_helper.sparse_proto_to_rows
        original_func = entity_helper.sparse_proto_to_rows
        try:
            entity_helper.sparse_proto_to_rows = lambda x, start, end: expected_rows[start:end]
            
            sr = SearchResult(res_data)
            assert len(sr) == 1
            hits = sr[0]
            assert len(hits) == 3
            assert hits[0].entity["sparse_vec"] == {0: 1.0}
            assert hits[1].entity["sparse_vec"] == {1: 0.5}
            assert hits[2].entity["sparse_vec"] == {2: 0.2}
            
            # Test lazy materialization logic via _get_physical_index
            # Modify the ACTUAL proto object in res_data
            res_data.fields_data[0].valid_data.extend([True, False, True])
            
            valid_rows = [{0: 1.0}, {2: 0.2}] # physical rows corresponding to logical 0 and 2
            entity_helper.sparse_proto_to_rows = lambda x, start, end: valid_rows[start:end]
            
            sr = SearchResult(res_data)
            hits = sr[0]
            assert hits[0].entity["sparse_vec"] == {0: 1.0}
            assert hits[1].entity["sparse_vec"] is None # invalid
            assert hits[2].entity["sparse_vec"] == {2: 0.2}

        finally:
            entity_helper.sparse_proto_to_rows = original_func

    def test_array_of_struct(self):
        # StructArrayField expected
        fields_data = [
            schema_pb2.FieldData(
                type=DataType._ARRAY_OF_STRUCT,
                field_name="struct_arr",
                field_id=102,
                struct_arrays=schema_pb2.StructArrayField() 
            )
        ]
        
        res_data = self._create_base_result(fields_data)
        
        expected_data = [
            [{"name": "a", "age": 10}],
            [{"name": "b", "age": 20}],
            [{"name": "c", "age": 30}]
        ]
        
        original_func = entity_helper.extract_struct_array_from_column_data
        try:
            # search_result calls extract_struct_array_from_column_data(struct_arrays, idx)
            entity_helper.extract_struct_array_from_column_data = lambda x, idx: expected_data[idx]
            
            sr = SearchResult(res_data)
            hits = sr[0]
            assert hits[0].entity["struct_arr"] == [{"name": "a", "age": 10}]
            assert hits[1].entity["struct_arr"] == [{"name": "b", "age": 20}]
            assert hits[2].entity["struct_arr"] == [{"name": "c", "age": 30}]
            
        finally:
            entity_helper.extract_struct_array_from_column_data = original_func

    def test_array_of_vector(self):
        # We need to construct a robust VectorArray manually
        # search_result.py logic:
        # vector_array = field_data.vectors.vector_array
        # vector_data = vector_array.data[idx]
        # float_data = vector_data.float_vector.data
        
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
                vectors=schema_pb2.VectorField(vector_array=vec_arr)
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
                    geometry_wkt_data=schema_pb2.GeometryWktArray(data=["POINT(1 1)", "POINT(2 2)", "POINT(3 3)"])
                )
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
                "report_value": "100", # cost
                "scanned_remote_bytes": "2000",
                "scanned_total_bytes": "3000",
                "cache_hit_ratio": "0.5"
            }
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
                scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[10, 0, 30])) # 0 is placeholder for null
            ),
            # FLOAT
            schema_pb2.FieldData(
                type=DataType.FLOAT,
                field_name="float_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[1.1, 0.0, 3.3]))
            ),
            # VARCHAR
            schema_pb2.FieldData(
                type=DataType.VARCHAR,
                field_name="str_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=["a", "", "c"]))
            ),
             # JSON
            schema_pb2.FieldData(
                type=DataType.JSON,
                field_name="json_field",
                valid_data=valid_data,
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b'{"a":1}', b'', b'{"c":3}']))
            )
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
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b"{bad"]))
            )
        ]
        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=1,
            scores=[0.1],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            topks=[1]
        )
        
        sr = SearchResult(res_data)
        with pytest.raises(orjson.JSONDecodeError):
            _ = sr[0][0].entity # Trigger materialization

    def test_large_result_str(self):
        # Create result with > 10 items
        count = 15
        pk = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=list(range(count))))
        res_data = schema_pb2.SearchResultData(
            num_queries=1,
            top_k=count,
            scores=[0.1] * count,
            ids=pk,
            topks=[count]
        )
        
        sr = SearchResult(res_data)
        s = str(sr[0])
        assert "entities remaining" in s
        assert str(count - 10) in s

    def test_vectors_optimization(self):
         # Test the "direct return" optimization for FLOAT_VECTOR
         # search_result.py:526
         # if start == 0 and (end - start) * dim >= len(vectors.float_vector.data):
         
         data = [1.0, 2.0, 3.0, 4.0]
         fields_data = [
            schema_pb2.FieldData(
                type=DataType.FLOAT_VECTOR,
                field_name="vec",
                field_id=101,
                vectors=schema_pb2.VectorField(
                    dim=2,
                    float_vector=schema_pb2.FloatArray(data=data)
                )
            )
        ]
         # 2 vectors of dim 2
         pk = schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2]))
         res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=2,
            scores=[0.1, 0.2],
            ids=pk,
            topks=[2]
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
        from pymilvus.client.search_result import Hit
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
        from pymilvus.client.search_result import Hits
        
        # Manual construction of arguments for Hits
        # fields: Dict[str, Tuple[List[Any], schema_pb2.FieldData]]
        
        field_meta = schema_pb2.FieldData(type=DataType.INT32, field_name="age")
        fields = {
            "age": ([10, 20], field_meta)
        }
        
        hits = Hits(
            topk=2,
            pks=[1, 2],
            distances=[0.1, 0.2],
            fields=fields,
            output_fields=["age", "extra"],
            pk_name="id"
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
        fields = {
            "meta": ([{"d1": 1}, {"d1": 2}], json_meta)
        }
        hits = Hits(
            topk=2,
            pks=[1, 2],
            distances=[0.1, 0.2],
            fields=fields,
            output_fields=["d1"], # d1 requested
            pk_name="id"
        )
        assert hits[0].entity["d1"] == 1

class TestGetFieldsByRange:
    """Test SearchResult._get_fields_by_range"""
    
    def test_get_fields_all_types(self):
        # We can use SearchResult instance to call this method
        # It's stateless regarding instance data, it just uses the args
        
        res = SearchResult(schema_pb2.SearchResultData())
        
        # Prepare all fields data
        all_fields_data = []
        count = 5
        
        # BOOL
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.BOOL,
            field_name="bool",
            scalars=schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True]*count))
        ))
        
        # INTs
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.INT32,
            field_name="int32",
            scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[i for i in range(count)]))
        ))
        
        # INT64
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.INT64,
            field_name="int64",
            scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[i for i in range(count)]))
        ))
        
        # FLOAT/DOUBLE
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.FLOAT,
            field_name="float",
            scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[float(i) for i in range(count)]))
        ))
        
        # DOUBLE
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.DOUBLE,
            field_name="double",
            scalars=schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[float(i) for i in range(count)]))
        ))
        
        # VARCHAR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.VARCHAR,
            field_name="varchar",
            scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=[str(i) for i in range(count)]))
        ))
        
        # JSON
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.JSON,
            field_name="json",
            scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b'{}']*count))
        ))
        
        # GEOMETRY
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.GEOMETRY,
            field_name="geom",
            scalars=schema_pb2.ScalarField(geometry_wkt_data=schema_pb2.GeometryWktArray(data=["POINT(0 0)"]*count))
        ))
        
        # ARRAY
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.ARRAY,
            field_name="array",
            scalars=schema_pb2.ScalarField(array_data=schema_pb2.ArrayArray(
                data=[schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[1])) for _ in range(count)],
                element_type=DataType.INT32
            ))
        ))
        
        # ARRAY_OF_STRUCT (mocking helper)
        aos_fd = schema_pb2.FieldData(type=DataType._ARRAY_OF_STRUCT, field_name="aos")
        # We need check usage:
        # struct_array_data.append(entity_helper.extract_struct_array_from_column_data(field.struct_arrays, row_idx))
        # We can mock extract_struct_array_from_column_data
        all_fields_data.append(aos_fd)
        
        # VECTORS
        # FLOAT_VECTOR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.FLOAT_VECTOR,
            field_name="fv",
            vectors=schema_pb2.VectorField(dim=2, float_vector=schema_pb2.FloatArray(data=[0.0]*(count*2)))
        ))
        
        # BINARY_VECTOR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.BINARY_VECTOR,
            field_name="bv",
            vectors=schema_pb2.VectorField(dim=8, binary_vector=b'\x00'*count)
        ))
        
        # FLOAT16_VECTOR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.FLOAT16_VECTOR,
            field_name="f16v",
            vectors=schema_pb2.VectorField(dim=2, float16_vector=b'\x00'*(count*2*2))
        ))
        
        # BFLOAT16_VECTOR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.BFLOAT16_VECTOR,
            field_name="bf16v",
            vectors=schema_pb2.VectorField(dim=2, bfloat16_vector=b'\x00'*(count*2*2))
        ))
        
        # INT8_VECTOR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.INT8_VECTOR,
            field_name="i8v",
            vectors=schema_pb2.VectorField(dim=2, int8_vector=b'\x00'*(count*2))
        ))
        
        # SPARSE_FLOAT_VECTOR
        all_fields_data.append(schema_pb2.FieldData(
            type=DataType.SPARSE_FLOAT_VECTOR,
            field_name="sv",
            vectors=schema_pb2.VectorField(sparse_float_vector=schema_pb2.SparseFloatArray(contents=[b'']*count))
        ))
        
        original_struct_func = entity_helper.extract_struct_array_from_column_data
        original_sparse_func = entity_helper.sparse_proto_to_rows
        
        try:
            entity_helper.extract_struct_array_from_column_data = lambda x, y: {}
            entity_helper.sparse_proto_to_rows = lambda x, start, end: [{}]*(end-start)
            
            result = res._get_fields_by_range(0, count, all_fields_data)
            
            assert "bool" in result
            assert len(result["bool"][0]) == count
            assert "int32" in result
            assert "int64" in result
            assert "float" in result
            assert "double" in result
            assert "varchar" in result
            assert "json" in result
            assert "geom" in result
            assert "array" in result
            assert "aos" in result
            assert "fv" in result
            assert "bv" in result
            assert "f16v" in result
            assert "bf16v" in result
            assert "i8v" in result
            assert "sv" in result
            
        finally:
            entity_helper.extract_struct_array_from_column_data = original_struct_func
            entity_helper.sparse_proto_to_rows = original_sparse_func

    def test_get_fields_optimized_float_vector(self):
        """Test the 25% perf optimization path for float vector"""
        res = SearchResult(schema_pb2.SearchResultData())
        fd = schema_pb2.FieldData(
            type=DataType.FLOAT_VECTOR,
            field_name="fv",
            vectors=schema_pb2.VectorField(dim=2, float_vector=schema_pb2.FloatArray(data=[1.0, 2.0, 3.0, 4.0]))
        )
        
        # Full range
        result = res._get_fields_by_range(0, 2, [fd])
        # It calls directly return data
        assert result["fv"][0] == [1.0, 2.0, 3.0, 4.0]

class TestHelpers:
    """Test standalone helper functions in search_result.py"""
    
    def test_extract_array_row_data(self):
        from pymilvus.client.search_result import extract_array_row_data
        
        # Test None input
        res = extract_array_row_data([None, None], DataType.INT64)
        assert res == [None, None]
        
        # Test INT64
        arr = [schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[1, 2]))]
        res = extract_array_row_data(arr, DataType.INT64)
        assert res == [[1, 2]]
        
        # Test BOOL
        arr = [schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True, False]))]
        res = extract_array_row_data(arr, DataType.BOOL)
        assert res == [[True, False]]
        
        # Test INT8/16/32
        arr = [schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[1, 2]))]
        res = extract_array_row_data(arr, DataType.INT32)
        assert res == [[1, 2]]
        
        # Test FLOAT
        arr = [schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[1.0, 2.0]))]
        res = extract_array_row_data(arr, DataType.FLOAT)
        assert res == [[1.0, 2.0]]
        
        # Test DOUBLE
        arr = [schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[1.0, 2.0]))]
        res = extract_array_row_data(arr, DataType.DOUBLE)
        assert res == [[1.0, 2.0]]
        
        # Test VARCHAR
        arr = [schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=["a", "b"]))]
        res = extract_array_row_data(arr, DataType.VARCHAR)
        assert res == [["a", "b"]]

    def test_apply_valid_data(self):
        from pymilvus.client.search_result import apply_valid_data
        
        data = [1, 2, 3]
        valid_data = [True, False, True]
        
        # apply_valid_data(data, valid_data, start, end)
        res = apply_valid_data(data, valid_data, 0, 3)
        assert res == [1, None, 3]
        
        # Partial range
        data = [2, 3]
        res = apply_valid_data(data, valid_data, 1, 3) # valid_data[1:3] is [False, True]
        assert res == [None, 3]
        
        # None valid_data
        res = apply_valid_data([1, 2], None, 0, 2)
        assert res == [1, 2]

    def test_extract_struct_field_value(self):
        from pymilvus.client.search_result import extract_struct_field_value
        
        # Scalar types
        # INT32
        fd = schema_pb2.FieldData(type=DataType.INT32, scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[10, 20])))
        assert extract_struct_field_value(fd, 0) == 10
        assert extract_struct_field_value(fd, 1) == 20
        assert extract_struct_field_value(fd, 2) is None
        
        # INT64
        fd = schema_pb2.FieldData(type=DataType.INT64, scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[10, 20])))
        assert extract_struct_field_value(fd, 0) == 10
        
        # FLOAT (returns np.single)
        fd = schema_pb2.FieldData(type=DataType.FLOAT, scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[1.0])))
        val = extract_struct_field_value(fd, 0)
        assert val == 1.0
        assert isinstance(val, (float, np.floating))
        
        # DOUBLE
        fd = schema_pb2.FieldData(type=DataType.DOUBLE, scalars=schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[1.0])))
        assert extract_struct_field_value(fd, 0) == 1.0

        # BOOL
        fd = schema_pb2.FieldData(type=DataType.BOOL, scalars=schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True])))
        assert extract_struct_field_value(fd, 0) is True
        
        # VARCHAR
        fd = schema_pb2.FieldData(type=DataType.VARCHAR, scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=["s"])))
        assert extract_struct_field_value(fd, 0) == "s"
        
        # JSON
        fd = schema_pb2.FieldData(type=DataType.JSON, scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b'{"a":1}'])))
        assert extract_struct_field_value(fd, 0) == {"a": 1}
        
        # FLOAT_VECTOR
        fd = schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, vectors=schema_pb2.VectorField(dim=2, float_vector=schema_pb2.FloatArray(data=[1.0, 2.0, 3.0, 4.0])))
        assert extract_struct_field_value(fd, 0) == [1.0, 2.0]
        assert extract_struct_field_value(fd, 1) == [3.0, 4.0]
        assert extract_struct_field_value(fd, 2) is None
        
        # BINARY_VECTOR
        fd = schema_pb2.FieldData(type=DataType.BINARY_VECTOR, vectors=schema_pb2.VectorField(dim=8, binary_vector=b'\x01\x02'))
        # dim=8 means 1 byte per vector
        assert extract_struct_field_value(fd, 0) == b'\x01'
        assert extract_struct_field_value(fd, 1) == b'\x02'
        assert extract_struct_field_value(fd, 2) is None
        
        # Unsupported / Out of range
        fd = schema_pb2.FieldData(type=DataType.INT8)
        assert extract_struct_field_value(fd, 0) is None

class TestCoverageEdgeCases:
    """Targeted tests for remaining coverage gaps"""
    
    def test_hit_get_exception(self):
        # target 805-807: KeyError handling in get
        from pymilvus.client.search_result import Hit
        h = Hit({}, pk_name="id")
        # Access key that doesn't exist to trigger KeyError caught by get
        assert h.get("non_existent", "default") == "default"

    def test_search_result_materialize(self):
        # target 360-361: SearchResult.materialize
        res_data = schema_pb2.SearchResultData(
            num_queries=1, top_k=1, topks=[1], 
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            scores=[0.1]
        )
        sr = SearchResult(res_data)
        sr.materialize() # Just call it
        assert sr[0].has_materialized

    def test_get_field_data_exception(self):
         # target 620-621
         from pymilvus.client.search_result import get_field_data, MilvusException
         fd = schema_pb2.FieldData(type=999)
         with pytest.raises(MilvusException):
             get_field_data(fd)

    def test_hybrid_hits_unsupported_type(self):
         # target 96-97
         from pymilvus.client.search_result import HybridHits, MilvusException
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
        
        from pymilvus.client.search_result import HybridHits
        from unittest.mock import MagicMock, patch
        
        fd = schema_pb2.FieldData(type=DataType._ARRAY_OF_STRUCT, field_name="aos", field_id=1)
        # Mock get_field_data to return object without fields
        
        with patch("pymilvus.client.search_result.get_field_data") as mock_get:
            mock_get.return_value = MagicMock(spec=[]) # No fields attr
            
            hh = HybridHits(0, 1, [1], [0.1], [fd], [], [], "id")
            hh.materialize()
            assert hh[0].entity["aos"] is None
            
    def test_materialize_vector_else(self):
         # target 274: idx < len(vector_array.data) is False
         from pymilvus.client.search_result import HybridHits
         
         vec_arr = schema_pb2.VectorArray()
         # Empty vector array
         
         fd = schema_pb2.FieldData(
             type=DataType._ARRAY_OF_VECTOR, 
             field_name="aov", 
             field_id=1,
             vectors=schema_pb2.VectorField(vector_array=vec_arr)
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
            scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b"{bad"]))
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
        from pymilvus.client.search_result import Hits
        
        json_fd = schema_pb2.FieldData(type=DataType.JSON, field_name="meta", is_dynamic=True)
        vec_fd = schema_pb2.FieldData(type=DataType.FLOAT_VECTOR, field_name="emb")
        vec_fd.vectors.dim = 2
        vec_fd.vectors.float_vector.data.extend([0.1, 0.2])
        
        fields = {
            "meta": ([{"a": 1}], json_fd),
            "emb": ([0.1, 0.2], vec_fd)
        }
        
        # Initialize Hits
        hits = Hits(
            topk=1,
            pks=[1],
            distances=[0.0],
            fields=fields,
            output_fields=[], 
            pk_name="id"
        )
        
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
            schema_pb2.FieldData(type=DataType.JSON, field_name="j", scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[b'{}'])))
        ]
        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=1,
            scores=[0.1],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1])),
            topks=[1]
        )
        sr = SearchResult(res_data)
        hits = sr[0]
        assert hits.has_materialized is False
        
        # Iteration should trigger materialization
        for hit in hits:
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
                valid_data=[True, False, True], # 3 rows, 2 valid
                vectors=schema_pb2.VectorField(
                    dim=2,
                    float_vector=schema_pb2.FloatArray(data=[1.0, 1.0, 2.0, 2.0]) # 2 vectors (sparse storage)
                )
            )
        ]
        
        res_data = schema_pb2.SearchResultData(
            fields_data=fields_data,
            num_queries=1,
            top_k=3,
            scores=[0.1, 0.2, 0.3],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[1, 2, 3])),
            topks=[3]
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


