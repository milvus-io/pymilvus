import os
from typing import List, Tuple
from pymilvus.client.abstract import Hit, Hits, SearchResult
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2
import random

import pytest
import ujson


class TestHit:
    @pytest.mark.parametrize("pk_dist", [
        (1, 0.1),
        (2,  0.3),
        ("a", 0.4),
    ])
    def test_hit_no_fields(self, pk_dist: List[Tuple]):
        pk, dist = pk_dist
        h = Hit(pk, dist, {})
        assert h.id == pk
        assert h.score == dist
        assert h.distance == dist
        assert h.fields == {}

        assert h.to_dict() == {
            "id": pk,
            "distance": dist,
            "entity": {},
        }

    @pytest.mark.parametrize("pk_dist_fields", [
        (1, 0.1, {"vector": [1., 2., 3., 4.],  "description": "This is a test", 'd_a': "dynamic a"}),
        (2,  0.3, {"vector": [3., 4., 5., 6.], "description": "This is a test too", 'd_b': "dynamic b"}),
        ("a", 0.4, {"vector": [4., 4., 4., 4.], "description": "This is a third test", 'd_a': "dynamic a twice"}),
    ])
    def test_hit_with_fields(self, pk_dist_fields: List[Tuple]):
        h = Hit(*pk_dist_fields)

        # fixed attributes
        assert h.id == pk_dist_fields[0]
        assert h.score == pk_dist_fields[1]
        assert h.distance == h.score
        assert h.fields == pk_dist_fields[2]

        # dynamic attributes
        assert h.description == pk_dist_fields[2].get("description")
        assert h.vector == pk_dist_fields[2].get("vector")

        with pytest.raises(Exception):
            h.field_not_exits

        print(h)


class TestSearchResult:
    @pytest.mark.parametrize("pk", [
        schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[i for i in range(6)])),
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
        assert 2 == len(r)
        for hits in r:
            assert isinstance(hits, Hits)
            assert len(hits.ids) == 3
            assert len(hits.distances) == 3

        # slicable
        assert 1 == len(r[1:])
        first_q, second_q = r[0], r[1]
        assert 3 == len(first_q)
        assert 3 == len(first_q[:])
        assert 2 == len(first_q[1:])
        assert 1 == len(first_q[2:])
        assert 0 == len(first_q[3:])
        print(first_q[:])
        print(first_q[1:])
        print(first_q[2:])

        first_hit = first_q[0]
        print(first_hit)
        assert first_hit.distance == 0.
        assert first_hit.fields == {}

    @pytest.mark.parametrize("pk", [
        schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[i for i in range(6)])),
        schema_pb2.IDs(str_id=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))
    ])
    def test_search_result_with_fields_data(self, pk):
        fields_data = [
            schema_pb2.FieldData(type=DataType.BOOL, field_name="bool_field", field_id=100,
                                 scalars=schema_pb2.ScalarField(bool_data=schema_pb2.BoolArray(data=[True for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.INT8, field_name="int8_field", field_id=101,
                                 scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[i for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.INT16, field_name="int16_field", field_id=102,
                                 scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[i for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.INT32, field_name="int32_field", field_id=103,
                                 scalars=schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[i for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.INT64, field_name="int64_field", field_id=104,
                                 scalars=schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[i for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.FLOAT, field_name="float_field", field_id=105,
                                 scalars=schema_pb2.ScalarField(float_data=schema_pb2.FloatArray(data=[i*1. for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.DOUBLE, field_name="double_field", field_id=106,
                                 scalars=schema_pb2.ScalarField(double_data=schema_pb2.DoubleArray(data=[i*1. for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.VARCHAR, field_name="varchar_field", field_id=107,
                                 scalars=schema_pb2.ScalarField(string_data=schema_pb2.StringArray(data=[str(i*10) for i in range(6)]))),
            schema_pb2.FieldData(type=DataType.ARRAY, field_name="int16_array_field", field_id=108,
                                 scalars=schema_pb2.ScalarField(
                                     array_data=schema_pb2.ArrayArray(
                                         data=[schema_pb2.ScalarField(int_data=schema_pb2.IntArray(data=[j for j in range(10)])) for i in range(6)],
                                         element_type=DataType.INT16,
                                     ),
                                 )),
            schema_pb2.FieldData(type=DataType.ARRAY, field_name="int64_array_field", field_id=109,
                                 scalars=schema_pb2.ScalarField(
                                     array_data=schema_pb2.ArrayArray(
                                         data=[schema_pb2.ScalarField(long_data=schema_pb2.LongArray(data=[j for j in range(10)])) for i in range(6)],
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
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[ujson.dumps({i: i for i in range(3)}).encode() for i in range(6)])),
            ),
            schema_pb2.FieldData(type=DataType.JSON, field_name="$meta", field_id=112,
                is_dynamic=True,
                scalars=schema_pb2.ScalarField(json_data=schema_pb2.JSONArray(data=[ujson.dumps({str(i*100): i}).encode() for i in range(6)])),
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
        print(r[0])
        assert 2 == len(r)
        assert 3 == len(r[0]) == len(r[1])
        assert {'0': 0, '1': 1, '2': 2} == r[0][0].normal_json_field
        # dynamic field
        assert 1 == r[0][1].fields.get('100')

        assert 0 == r[0][0].int32_field
        assert 1 == r[0][1].int8_field
        assert 2 == r[0][2].int16_field
        assert [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] == r[0][1].int64_array_field
        assert 32 == len(r[0][0].entity.bfloat16_vector_field)
        assert 32 == len(r[0][0].entity.float16_vector_field)
