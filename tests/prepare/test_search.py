"""Tests for search and query Prepare methods."""

import numpy as np
import pytest
from pymilvus import DataType, Function, FunctionType
from pymilvus.client.abstract import BaseRanker
from pymilvus.client.prepare import Prepare
from pymilvus.exceptions import ParamError
from pymilvus.orm.schema import FunctionScore, LexicalHighlighter


class TestSearchRequestsWithExpr:
    """Tests for search_requests_with_expr."""

    @pytest.fixture
    def basic_search_params(self):
        """Basic search parameters."""
        return {"metric_type": "L2", "params": {"nprobe": 10}}

    def test_search_with_data(self, basic_search_params):
        """Test search with vector data."""
        data = [[1.0, 2.0, 3.0, 4.0]]
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=data,
            anns_field="vector",
            param=basic_search_params,
            limit=10,
        )
        assert req.collection_name == "test"
        assert req.nq == 1

    def test_search_with_ids(self, basic_search_params):
        """Test search with IDs instead of vectors."""
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            ids=[1, 2, 3],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
        )
        assert req.collection_name == "test"
        assert req.nq == 3

    def test_search_neither_data_nor_ids(self, basic_search_params):
        """Test search without data or ids raises error."""
        with pytest.raises(ValueError, match="Either data or ids"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                anns_field="vector",
                param=basic_search_params,
                limit=10,
            )

    def test_search_invalid_params_type(self):
        """Test search with non-dict params."""
        with pytest.raises(ParamError, match="must be a dict"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param={"params": "invalid"},
                limit=10,
            )

    def test_search_page_retain_order_both_kwargs_and_param(self, basic_search_params):
        """Test page_retain_order in both kwargs and param raises error."""
        basic_search_params["page_retain_order"] = True
        with pytest.raises(ParamError, match="page_retain_order both in kwargs and param"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                page_retain_order=True,
            )

    def test_search_page_retain_order_invalid_type(self, basic_search_params):
        """Test page_retain_order with invalid type."""
        with pytest.raises(ParamError, match="expect bool"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                page_retain_order="true",
            )

    def test_search_offset_both_kwargs_and_param(self, basic_search_params):
        """Test offset in both kwargs and param raises error."""
        basic_search_params["offset"] = 10
        with pytest.raises(ParamError, match="offset both in kwargs and param"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                offset=20,
            )

    def test_search_offset_invalid_type(self, basic_search_params):
        """Test offset with invalid type."""
        with pytest.raises(ParamError, match="expect int"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                offset="10",
            )

    @pytest.mark.parametrize(
        "iterator_key,value",
        [
            pytest.param("is_iterator", True, id="iterator"),
            pytest.param("collection_id", 123, id="collection_id"),
            pytest.param("iter_search_v2", True, id="iter_v2"),
            pytest.param("iter_search_batch_size", 100, id="batch_size"),
            pytest.param("iter_search_last_bound", 0.5, id="last_bound"),
            pytest.param("iter_search_id", "test_id", id="iter_id"),
        ],
    )
    def test_search_iterator_params(self, basic_search_params, iterator_key, value):
        """Test search with various iterator parameters."""
        kwargs = {iterator_key: value}
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            **kwargs,
        )
        assert req is not None

    @pytest.mark.parametrize(
        "group_key,value",
        [
            pytest.param("group_by_field", "category", id="group_by"),
            pytest.param("group_size", 5, id="group_size"),
            pytest.param("strict_group_size", True, id="strict_group"),
        ],
    )
    def test_search_group_params(self, basic_search_params, group_key, value):
        """Test search with grouping parameters."""
        kwargs = {group_key: value}
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            **kwargs,
        )
        assert req is not None

    def test_search_order_by_fields_list(self, basic_search_params):
        """Test search with order_by_fields as list of dicts."""
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            order_by_fields=[{"field": "score", "order": "desc"}],
        )
        assert req is not None

    def test_search_order_by_fields_missing_field(self, basic_search_params):
        """Test order_by_fields without 'field' key raises error."""
        with pytest.raises(ParamError, match="'field' key is required"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                order_by_fields=[{"order": "desc"}],
            )

    def test_search_order_by_fields_empty_field(self, basic_search_params):
        """Test order_by_fields with empty 'field' raises error."""
        with pytest.raises(ParamError, match="'field' key is required"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                order_by_fields=[{"field": "", "order": "desc"}],
            )

    def test_search_order_by_fields_invalid_item(self, basic_search_params):
        """Test order_by_fields with non-dict item raises error."""
        with pytest.raises(ParamError, match="expect dict"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                order_by_fields=["invalid"],
            )

    def test_search_order_by_fields_string(self, basic_search_params):
        """Test order_by_fields as string."""
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            order_by_fields="score:desc",
        )
        assert req is not None

    @pytest.mark.parametrize(
        "json_type,expected",
        [
            pytest.param(DataType.INT8, "Int8", id="int8"),
            pytest.param(DataType.INT16, "Int16", id="int16"),
            pytest.param(DataType.INT32, "Int32", id="int32"),
            pytest.param(DataType.INT64, "Int64", id="int64"),
            pytest.param(DataType.BOOL, "Bool", id="bool"),
            pytest.param(DataType.VARCHAR, "VarChar", id="varchar"),
            pytest.param(DataType.STRING, "VarChar", id="string"),
        ],
    )
    def test_search_json_type_casting(self, basic_search_params, json_type, expected):
        """Test search with JSON type casting."""
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            json_path="$.field",
            json_type=json_type,
        )
        assert req is not None

    def test_search_json_type_unsupported(self, basic_search_params):
        """Test search with unsupported JSON type."""
        with pytest.raises(ParamError, match="Unsupported json cast type"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                json_path="$.field",
                json_type=DataType.FLOAT_VECTOR,
            )

    def test_search_with_expr(self, basic_search_params):
        """Test search with filter expression."""
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            expr="id > 100",
        )
        assert req.dsl == "id > 100"

    def test_search_with_function_ranker(self, basic_search_params):
        """Test search with Function ranker."""
        ranker = Function(
            "bm25",
            FunctionType.RERANK,
            input_field_names=["text"],
            output_field_names=[],
            params={"k": 1.2},
        )
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            ranker=ranker,
        )
        assert req.function_score is not None

    def test_search_with_function_score_ranker(self, basic_search_params):
        """Test search with FunctionScore ranker."""
        func = Function(
            "bm25",
            FunctionType.RERANK,
            input_field_names=["text"],
            output_field_names=[],
            params={"k": 1.2},
        )
        ranker = FunctionScore(functions=[func], params={"mode": "multiply"})
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            ranker=ranker,
        )
        assert req.function_score is not None

    def test_search_with_invalid_ranker(self, basic_search_params):
        """Test search with invalid ranker type."""
        with pytest.raises(ParamError, match="must be a Function or FunctionScore"):
            Prepare.search_requests_with_expr(
                collection_name="test",
                data=[[1.0, 2.0]],
                anns_field="vector",
                param=basic_search_params,
                limit=10,
                ranker="invalid",
            )

    def test_search_with_highlighter(self, basic_search_params):
        """Test search with highlighter."""
        highlighter = LexicalHighlighter(pre_tags=["<em>"], post_tags=["</em>"])
        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            highlighter=highlighter,
        )
        assert req.highlighter is not None

    @pytest.mark.parametrize(
        "extra_params",
        [
            pytest.param({"hints": "hint_value"}, id="hints"),
            pytest.param({"analyzer_name": "standard"}, id="analyzer"),
            pytest.param({"timezone": "UTC"}, id="timezone"),
            pytest.param({"time_fields": "timestamp"}, id="time_fields"),
            pytest.param({"strict_cast": True}, id="strict_cast"),
        ],
    )
    def test_search_with_extra_params(self, basic_search_params, extra_params):
        """Test search with various extra parameters."""
        if "hints" in extra_params:
            basic_search_params["hints"] = extra_params["hints"]
            extra_params = {}
        elif "analyzer_name" in extra_params:
            basic_search_params["analyzer_name"] = extra_params["analyzer_name"]
            extra_params = {}

        req = Prepare.search_requests_with_expr(
            collection_name="test",
            data=[[1.0, 2.0]],
            anns_field="vector",
            param=basic_search_params,
            limit=10,
            **extra_params,
        )
        assert req is not None


class TestBuildIdsProto:
    """Tests for _build_ids_proto."""

    def test_empty_ids(self):
        """Test with empty ids list."""
        with pytest.raises(ParamError, match="must not be empty"):
            Prepare._build_ids_proto([])

    def test_boolean_id(self):
        """Test with boolean id raises error."""
        with pytest.raises(ParamError, match="must not contain boolean"):
            Prepare._build_ids_proto([True, False])

    def test_int_ids(self):
        """Test with integer ids."""
        result = Prepare._build_ids_proto([1, 2, 3])
        assert result.int_id.data == [1, 2, 3]

    def test_numpy_int_ids(self):
        """Test with numpy integer ids."""
        result = Prepare._build_ids_proto([np.int64(1), np.int64(2)])
        assert result.int_id.data == [1, 2]

    def test_str_ids(self):
        """Test with string ids."""
        result = Prepare._build_ids_proto(["a", "b", "c"])
        assert list(result.str_id.data) == ["a", "b", "c"]

    def test_unsupported_id_type(self):
        """Test with unsupported id type."""
        with pytest.raises(ParamError, match="Unsupported id type"):
            Prepare._build_ids_proto([1.5, 2.5])


class TestPrepareExpressionTemplate:
    """Tests for prepare_expression_template."""

    def test_empty_values(self):
        """Test with empty values dict."""
        result = Prepare.prepare_expression_template({})
        assert len(result) == 0

    def test_bool_value(self):
        """Test with boolean value."""
        result = Prepare.prepare_expression_template({"flag": True})
        assert result["flag"].bool_val is True

    def test_int_value(self):
        """Test with integer value."""
        result = Prepare.prepare_expression_template({"count": 42})
        assert result["count"].int64_val == 42

    def test_float_value(self):
        """Test with float value."""
        result = Prepare.prepare_expression_template({"score": 3.14})
        assert result["score"].float_val == pytest.approx(3.14)

    def test_string_value(self):
        """Test with string value."""
        result = Prepare.prepare_expression_template({"name": "test"})
        assert result["name"].string_val == "test"

    def test_array_of_ints(self):
        """Test with array of integers."""
        result = Prepare.prepare_expression_template({"ids": [1, 2, 3]})
        assert list(result["ids"].array_val.long_data.data) == [1, 2, 3]

    def test_array_of_floats(self):
        """Test with array of floats."""
        result = Prepare.prepare_expression_template({"scores": [1.0, 2.0, 3.0]})
        assert list(result["scores"].array_val.double_data.data) == [1.0, 2.0, 3.0]

    def test_array_of_strings(self):
        """Test with array of strings."""
        result = Prepare.prepare_expression_template({"names": ["a", "b", "c"]})
        assert list(result["names"].array_val.string_data.data) == ["a", "b", "c"]

    def test_array_of_bools(self):
        """Test with array of booleans."""
        result = Prepare.prepare_expression_template({"flags": [True, False, True]})
        assert list(result["flags"].array_val.bool_data.data) == [True, False, True]

    def test_nested_array(self):
        """Test with nested array."""
        result = Prepare.prepare_expression_template({"matrix": [[1, 2], [3, 4]]})
        assert len(result["matrix"].array_val.array_data.data) == 2

    def test_empty_array(self):
        """Test with empty array."""
        result = Prepare.prepare_expression_template({"empty": []})
        # Empty array should still be valid
        assert "empty" in result


class TestPreparePlaceholderStr:
    """Tests for _prepare_placeholder_str."""

    def test_float_vector(self):
        """Test with float vectors."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None

    def test_numpy_float32(self):
        """Test with numpy float32 arrays."""
        data = [np.array([1.0, 2.0], dtype=np.float32)]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None

    def test_numpy_float16(self):
        """Test with numpy float16 arrays."""
        data = [np.array([1.0, 2.0], dtype=np.float16)]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None

    def test_numpy_int8(self):
        """Test with numpy int8 arrays."""
        data = [np.array([1, 2], dtype=np.int8)]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None

    def test_bytes_binary_vector(self):
        """Test with bytes binary vector."""
        data = [b"\x01\x02\x03\x04"]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None

    def test_string_values(self):
        """Test with string values for text search."""
        data = ["query text 1", "query text 2"]
        result = Prepare._prepare_placeholder_str(data)
        assert result is not None

    def test_unsupported_numpy_dtype(self):
        """Test with unsupported numpy dtype."""
        data = [np.array([1, 2], dtype=np.complex64)]
        with pytest.raises(ParamError, match="unsupported data type"):
            Prepare._prepare_placeholder_str(data)


class TestHybridSearchRequest:
    """Tests for hybrid_search_request_with_ranker."""

    def test_hybrid_search_with_base_ranker(self):
        """Test hybrid search with base ranker."""

        class MockRanker(BaseRanker):
            def dict(self):
                return {"strategy": "rrf", "k": 60}

        req = Prepare.hybrid_search_request_with_ranker(
            collection_name="test",
            reqs=[],
            rerank=MockRanker(),
            limit=10,
        )
        assert req.collection_name == "test"

    def test_hybrid_search_with_function_ranker(self):
        """Test hybrid search with Function ranker."""
        ranker = Function(
            "rrf",
            FunctionType.RERANK,
            input_field_names=[],
            output_field_names=[],
            params={"k": 60},
        )
        req = Prepare.hybrid_search_request_with_ranker(
            collection_name="test",
            reqs=[],
            rerank=ranker,
            limit=10,
        )
        assert req.function_score is not None

    def test_hybrid_search_invalid_ranker(self):
        """Test hybrid search with invalid ranker type."""
        with pytest.raises(ParamError, match="must be a Function or a Ranker"):
            Prepare.hybrid_search_request_with_ranker(
                collection_name="test",
                reqs=[],
                rerank="invalid",
                limit=10,
            )

    @pytest.mark.parametrize(
        "extra_params",
        [
            pytest.param({"rank_group_scorer": "max"}, id="rank_scorer"),
            pytest.param({"group_by_field": "category"}, id="group_by"),
            pytest.param({"group_size": 5}, id="group_size"),
            pytest.param({"strict_group_size": True}, id="strict_group"),
        ],
    )
    def test_hybrid_search_with_extra_params(self, extra_params):
        """Test hybrid search with various extra parameters."""
        req = Prepare.hybrid_search_request_with_ranker(
            collection_name="test",
            reqs=[],
            rerank=None,
            limit=10,
            **extra_params,
        )
        assert req is not None


class TestQueryRequest:
    """Tests for query_request."""

    def test_basic_query(self):
        """Test basic query request."""
        req = Prepare.query_request(
            collection_name="test",
            expr="id > 0",
            output_fields=["id", "vector"],
            partition_names=[],
        )
        assert req.collection_name == "test"
        assert req.expr == "id > 0"

    @pytest.mark.parametrize(
        "query_params",
        [
            pytest.param({"limit": 100}, id="limit"),
            pytest.param({"offset": 10}, id="offset"),
            pytest.param({"limit": 100, "offset": 10}, id="limit_offset"),
            pytest.param({"timezone": "UTC"}, id="timezone"),
            pytest.param({"time_fields": "timestamp"}, id="time_fields"),
            pytest.param({"ignore_growing": True}, id="ignore_growing"),
            pytest.param({"reduce_stop_for_best": True}, id="reduce_stop"),
            pytest.param({"is_iterator": "true"}, id="iterator"),
            pytest.param({"collection_id": 123}, id="collection_id"),
        ],
    )
    def test_query_with_params(self, query_params):
        """Test query with various parameters."""
        req = Prepare.query_request(
            collection_name="test",
            expr="id > 0",
            output_fields=["id"],
            partition_names=[],
            **query_params,
        )
        assert req is not None

    def test_query_group_by_fields(self):
        """Test query with group_by_fields."""
        req = Prepare.query_request(
            collection_name="test",
            expr="id > 0",
            output_fields=["id"],
            partition_names=[],
            group_by_fields=["category"],  # Use the correct kwarg name
        )
        assert req is not None

    def test_query_group_by_fields_invalid_type(self):
        """Test query with non-list group_by_fields."""
        with pytest.raises(TypeError):
            Prepare.query_request(
                collection_name="test",
                expr="id > 0",
                output_fields=["id"],
                partition_names=[],
                group_by_fields="category",  # Use the correct kwarg name
            )


class TestFunctionSchemas:
    """Tests for function score and highlighter schema conversion."""

    def test_highlighter_schema(self):
        """Test highlighter schema conversion."""
        highlighter = LexicalHighlighter(
            pre_tags=["<em>"],
            post_tags=["</em>"],
        )
        result = Prepare.highlighter_schema(highlighter)
        assert result is not None

    def test_function_score_schema(self):
        """Test function score schema conversion."""
        func = Function(
            "bm25",
            FunctionType.RERANK,
            input_field_names=["text"],
            output_field_names=[],
            params={"k": 1.2},
        )
        score = FunctionScore(
            functions=[func],
            params={"mode": "multiply", "config": {"key": "value"}},
        )
        result = Prepare.function_score_schema(score)
        assert len(result.functions) == 1

    def test_ranker_to_function_score(self):
        """Test ranker to function score conversion."""
        ranker = Function(
            "bm25",
            FunctionType.RERANK,
            input_field_names=["text"],
            output_field_names=[],
            params={"k": 1.2, "config": {"nested": "value"}},
        )
        result = Prepare.ranker_to_function_score(ranker)
        assert len(result.functions) == 1

    def test_common_kv_value_dict(self):
        """Test common_kv_value with dict."""
        result = Prepare.common_kv_value({"key": "value"})
        assert result == '{"key": "value"}'

    def test_common_kv_value_list(self):
        """Test common_kv_value with list."""
        result = Prepare.common_kv_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_common_kv_value_string(self):
        """Test common_kv_value with string."""
        result = Prepare.common_kv_value("test")
        assert result == "test"


class TestDummyRequest:
    """Tests for dummy_request."""

    def test_dummy_request(self):
        """Test dummy request."""
        req = Prepare.dummy_request("test_type")
        assert req.request_type == "test_type"
