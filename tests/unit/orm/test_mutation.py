from unittest.mock import MagicMock

import pytest
from pymilvus.orm.mutation import MutationResult


class TestMutationResultWithNone:
    """Test MutationResult when initialized with None (falsy _mr)."""

    def setup_method(self):
        self.mr = MutationResult(None)

    @pytest.mark.parametrize(
        "prop, expected",
        [
            ("primary_keys", []),
            ("insert_count", 0),
            ("delete_count", 0),
            ("upsert_count", 0),
            ("timestamp", 0),
            ("succ_count", 0),
            ("err_count", 0),
            ("succ_index", []),
            ("err_index", []),
            ("cost", 0),
        ],
    )
    def test_properties_return_defaults(self, prop, expected):
        assert getattr(self.mr, prop) == expected

    def test_str_returns_empty(self):
        assert str(self.mr) == ""

    def test_repr_returns_empty(self):
        assert repr(self.mr) == ""


class TestMutationResultWithMock:
    """Test MutationResult when initialized with a truthy mock object."""

    def setup_method(self):
        self.mock = MagicMock()
        self.mock.primary_keys = [1, 2, 3]
        self.mock.insert_count = 3
        self.mock.delete_count = 1
        self.mock.upsert_count = 2
        self.mock.timestamp = 1234567890
        self.mock.succ_count = 3
        self.mock.err_count = 0
        self.mock.succ_index = [0, 1, 2]
        self.mock.err_index = [3]
        self.mock.cost = 42
        self.mock.__str__ = MagicMock(return_value="mock_mutation_result")
        self.mr = MutationResult(self.mock)

    @pytest.mark.parametrize(
        "prop, expected",
        [
            ("primary_keys", [1, 2, 3]),
            ("insert_count", 3),
            ("delete_count", 1),
            ("upsert_count", 2),
            ("timestamp", 1234567890),
            ("succ_count", 3),
            ("err_count", 0),
            ("succ_index", [0, 1, 2]),
            ("err_index", [3]),
            ("cost", 42),
        ],
    )
    def test_properties_return_mock_values(self, prop, expected):
        assert getattr(self.mr, prop) == expected

    def test_str_delegates_to_mr(self):
        assert str(self.mr) == "mock_mutation_result"

    def test_repr_delegates_to_mr(self):
        assert repr(self.mr) == "mock_mutation_result"
