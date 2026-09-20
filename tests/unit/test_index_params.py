import pytest
from pymilvus.exceptions import ParamError
from pymilvus.milvus_client.index import IndexParams, extract_bound_index_param


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
