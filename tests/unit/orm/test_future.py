from unittest.mock import MagicMock

from pymilvus.client.search_result import SearchResult
from pymilvus.orm.future import (
    BaseFuture,
    MutationFuture,
    SearchFuture,
    _EmptySearchFuture,
)
from pymilvus.orm.mutation import MutationResult


class TestEmptySearchFuture:
    """Test _EmptySearchFuture returns SearchResult and has no-op cancel/done."""

    def setup_method(self):
        self.future = _EmptySearchFuture()

    def test_result_returns_search_result(self):
        result = self.future.result()
        assert isinstance(result, SearchResult)

    def test_cancel_is_noop(self):
        assert self.future.cancel() is None

    def test_done_is_noop(self):
        assert self.future.done() is None


class TestBaseFutureWithMock:
    """Test BaseFuture delegates to the wrapped future."""

    def setup_method(self):
        self.mock_future = MagicMock()
        self.mock_future.result.return_value = "some_result"
        self.mock_future.cancel.return_value = True
        self.mock_future.done.return_value = True
        self.bf = BaseFuture(self.mock_future)

    def test_result_calls_inner_result(self):
        res = self.bf.result()
        self.mock_future.result.assert_called_once()
        assert res == "some_result"

    def test_on_response_is_identity(self):
        assert self.bf.on_response("hello") == "hello"

    def test_cancel_delegates(self):
        assert self.bf.cancel() is True
        self.mock_future.cancel.assert_called_once()

    def test_done_delegates(self):
        assert self.bf.done() is True
        self.mock_future.done.assert_called_once()


class TestBaseFutureWithNone:
    """Test BaseFuture with None falls back to _EmptySearchFuture."""

    def setup_method(self):
        self.bf = BaseFuture(None)

    def test_internal_future_is_empty_search_future(self):
        assert isinstance(self.bf._f, _EmptySearchFuture)

    def test_result_returns_search_result(self):
        result = self.bf.result()
        assert isinstance(result, SearchResult)

    def test_cancel_is_noop(self):
        assert self.bf.cancel() is None

    def test_done_is_noop(self):
        assert self.bf.done() is None


class TestMutationFuture:
    """Test MutationFuture.result wraps the response in MutationResult."""

    def test_result_returns_mutation_result(self):
        mock_future = MagicMock()
        mock_inner = MagicMock()
        mock_inner.primary_keys = [1]
        mock_future.result.return_value = mock_inner

        mf = MutationFuture(mock_future)
        result = mf.result()
        assert isinstance(result, MutationResult)
        assert result.primary_keys == [1]

    def test_on_response_wraps_in_mutation_result(self):
        mf = MutationFuture(MagicMock())
        res = mf.on_response("raw")
        assert isinstance(res, MutationResult)

    def test_with_none_future(self):
        mf = MutationFuture(None)
        assert isinstance(mf._f, _EmptySearchFuture)
        # result() will call on_response with a SearchResult, wrapping it in MutationResult
        result = mf.result()
        assert isinstance(result, MutationResult)


class TestSearchFuture:
    """Test SearchFuture inherits BaseFuture behavior without overriding."""

    def test_result_delegates(self):
        mock_future = MagicMock()
        mock_future.result.return_value = "search_data"

        sf = SearchFuture(mock_future)
        assert sf.result() == "search_data"

    def test_with_none_future(self):
        sf = SearchFuture(None)
        assert isinstance(sf._f, _EmptySearchFuture)
        result = sf.result()
        assert isinstance(result, SearchResult)
