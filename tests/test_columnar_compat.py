from unittest.mock import patch

import orjson
import pytest
from pymilvus.client import entity_helper
from pymilvus.client.columnar_search_result import ColumnarSearchResult
from pymilvus.client.types import DataType
from pymilvus.grpc_gen import schema_pb2
from tests.test_search_result import (
    TestCoverageEdgeCases,
    TestGetFieldsByRange,
    TestSearchResult,
    TestSearchResultExtended,
)


# Helper to bind the patch to the class
def as_columnar(cls):
    class ColumnarCompat(cls):
        def setup_method(self):
            self.patcher = patch("tests.test_search_result.SearchResult", ColumnarSearchResult)
            self.patcher.start()

        def teardown_method(self):
            self.patcher.stop()

    ColumnarCompat.__name__ = f"Columnar{cls.__name__}"
    return ColumnarCompat


# Generate test classes
TestSearchResultColumnar = as_columnar(TestSearchResult)
TestGetFieldsByRangeColumnar = as_columnar(TestGetFieldsByRange)
TestCoverageEdgeCasesColumnar = as_columnar(TestCoverageEdgeCases)


# Override TestSearchResultExtended to fix lazy tests
class TestSearchResultExtendedColumnar(TestSearchResultExtended):
    def setup_method(self):
        self.patcher = patch("tests.test_search_result.SearchResult", ColumnarSearchResult)
        self.patcher.start()

    def teardown_method(self):
        self.patcher.stop()

    def test_json_error_handling(self):
        # Override to force access for lazy evaluation
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

        sr = ColumnarSearchResult(res_data)
        with pytest.raises(orjson.JSONDecodeError):
            # Must access the specific field to trigger lazy decode
            # Accessing .entity works too IF .entity property materializes (but it doesn't in Columnar)
            # Accessing ["entity"] returns materialized dict which triggers decode
            _ = sr[0][0]["entity"]

    def test_sparse_float_vector(self, monkeypatch):
        # Override to ensure start/end logic matches Columnar
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

        # Use helper from parent class (need to ensure it works)
        # But _create_base_result is instance method on parent.
        # super() call or copy? super() works as we inherit.
        res_data = self._create_base_result(fields_data)
        expected_rows = [{0: 1.0}, {1: 0.5}, {2: 0.2}]

        # Mocking logic must handle (data, start, end)
        def mock_sparse_proto_to_rows(x, start, end):
            # Legacy calls with (0, 3) -> returns [r0, r1, r2]
            # Columnar calls with (0, 1), (1, 2), (2, 3) -> returns [r0], [r1], [r2]
            # This slice should work for both.
            return expected_rows[start:end]

        monkeypatch.setattr(entity_helper, "sparse_proto_to_rows", mock_sparse_proto_to_rows)

        sr = ColumnarSearchResult(res_data)
        assert len(sr) == 1
        hits = sr[0]
        assert len(hits) == 3

        # Accessing field triggers lazy load
        # Since expected_rows has 3 elements and we access index 0, 1, 2

        # Debugging: if logic is flawed, we'll see here
        assert hits[0].entity["sparse_vec"] == {0: 1.0}
        assert hits[1].entity["sparse_vec"] == {1: 0.5}
        assert hits[2].entity["sparse_vec"] == {2: 0.2}

        # Test with validity mask
        res_data.fields_data[0].valid_data.extend([True, False, True])
        valid_rows = [{0: 1.0}, {2: 0.2}]

        monkeypatch.setattr(
            entity_helper,
            "sparse_proto_to_rows",
            lambda x, start, end: (
                [valid_rows[1]] if start == 2 else [valid_rows[0]] if start == 0 else []
            ),
        )

        sr = ColumnarSearchResult(res_data)
        hits = sr[0]
        assert hits[0].entity["sparse_vec"] == {0: 1.0}
        assert hits[1].entity["sparse_vec"] is None
        assert hits[2].entity["sparse_vec"] == {2: 0.2}
