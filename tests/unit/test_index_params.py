import pytest
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParam, IndexParams, extract_bound_index_param


class TestIndexParamEquality:
    @pytest.mark.parametrize(
        "changes",
        [
            {},
            {"field_name": "other"},
            {"index_type": "FLAT"},
            {"index_name": "other"},
            {"metric_type": "IP"},
            {"params": {"M": 32}},
        ],
    )
    def test_compare_index_params(self, changes):
        kwargs = {
            "field_name": "vector",
            "index_type": "HNSW",
            "index_name": "vector_idx",
            "metric_type": "L2",
            "params": {"M": 16},
        }
        left = IndexParam(**kwargs)
        right = IndexParam(**{**kwargs, **changes})
        assert (left == right) is (not changes)
        assert (left != right) is bool(changes)

    @pytest.mark.parametrize("metric_type", ["L2", "IP"])
    def test_compare_dict(self, metric_type):
        index = IndexParam("vector", "HNSW", "vector_idx", metric_type="L2", M=16)
        config = {
            "field_name": "vector",
            "index_type": "HNSW",
            "index_name": "vector_idx",
            "metric_type": metric_type,
            "M": 16,
        }
        assert (index == config) is (metric_type == "L2")
        assert (config == index) is (metric_type == "L2")

    def test_compare_index_lists(self):
        left, right = IndexParams(), IndexParams()
        left.add_index("vector", "HNSW", "vector_idx", params={"M": 16}, metric_type="L2")
        right.add_index("vector", "HNSW", "vector_idx", M=16, metric_type="L2")
        assert left == right

    @pytest.mark.parametrize(
        "left_nlist, right_nlist, equal",
        [("128", "128", True), (128, 128, True), ("128", 128, False)],
        ids=["string-string", "int-int", "string-int"],
    )
    def test_compare_nlist_preserves_type(self, left_nlist, right_nlist, equal):
        left = IndexParam("vector", "IVF_FLAT", "vector_idx", nlist=left_nlist)
        right = IndexParam("vector", "IVF_FLAT", "vector_idx", nlist=right_nlist)
        assert (left == right) is equal
        assert (right == left) is equal
        assert (left != right) is (not equal)
        assert (left == right.to_dict()) is equal
        assert (right.to_dict() == left) is equal
        assert (left.to_dict() == right) is equal
        assert (right == left.to_dict()) is equal

    @pytest.mark.parametrize("other", [None, "vector", []])
    def test_compare_unrelated_type(self, other):
        assert IndexParam("vector", "FLAT", "") != other


class TestExtractBoundIndexParam:
    """Tests for the shared bound-index validation used by add_function_field."""

    def _valid_params(self, field_name: str = "sparse") -> IndexParams:
        index_params = IndexParams()
        index_params.add_index(
            field_name=field_name,
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_idx",
            metric_type="BM25",
        )
        return index_params

    def test_happy_path(self):
        index_name, configs = extract_bound_index_param("sparse", self._valid_params())
        assert index_name == "sparse_idx"
        assert configs["index_type"] == "SPARSE_INVERTED_INDEX"
        assert configs["metric_type"] == "BM25"

    def test_empty_field_name_matches_output_field(self):
        index_name, configs = extract_bound_index_param("sparse", self._valid_params(field_name=""))
        assert index_name == "sparse_idx"
        assert configs["index_type"] == "SPARSE_INVERTED_INDEX"

    def test_returns_copy_of_index_configs(self):
        index_params = self._valid_params()
        _, configs = extract_bound_index_param("sparse", index_params)
        configs["metric_type"] = "IP"
        _, configs_again = extract_bound_index_param("sparse", index_params)
        assert configs_again["metric_type"] == "BM25"

    def test_wrong_type_rejected(self):
        with pytest.raises(ParamError, match="wrong type of argument index_params"):
            extract_bound_index_param("sparse", {"index_type": "SPARSE_INVERTED_INDEX"})

    def test_empty_index_params_rejected(self):
        with pytest.raises(ParamError, match="exactly one index"):
            extract_bound_index_param("sparse", IndexParams())

    def test_multiple_entries_rejected(self):
        index_params = self._valid_params()
        index_params.add_index(field_name="sparse", index_type="SPARSE_WAND")
        with pytest.raises(ParamError, match="exactly one index"):
            extract_bound_index_param("sparse", index_params)

    def test_field_name_mismatch_rejected(self):
        with pytest.raises(ParamError, match="does not match"):
            extract_bound_index_param("other_field", self._valid_params())

    def test_missing_index_type_rejected(self):
        index_params = IndexParams()
        index_params.add_index(field_name="sparse", metric_type="BM25")
        with pytest.raises(ParamError, match="explicit index_type is required"):
            extract_bound_index_param("sparse", index_params)
