import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from pymilvus.client.constants import (
    COLLECTION_ID,
    ITERATOR_SESSION_CP_FILE,
    ITERATOR_SESSION_TS_FIELD,
)
from pymilvus.client.iterator import QueryIterator
from pymilvus.client.types import DataType


class MockQueryResult(list):
    """Stand-in for a real query response, with the `.extra` mapping the
    iterator reads the session timestamp from."""

    def __init__(self, rows, ts=100):
        super().__init__(rows)
        self.extra = {ITERATOR_SESSION_TS_FIELD: ts}


class TestQueryIteratorCheckpoint:
    @pytest.fixture
    def schema(self):
        return {"fields": [{"name": "pk", "type": DataType.INT64, "is_primary": True}]}

    @pytest.fixture
    def mock_handler(self):
        handler = Mock()
        handler.describe_collection.return_value = {COLLECTION_ID: 999}
        return handler

    def make_iterator(self, handler, cp_path, schema, pages):
        # The constructor issues one internal query() call to establish
        # the session timestamp when there's no valid checkpoint to read
        # it from. Prepend a dummy response to absorb that call, so
        # `pages` maps 1:1 onto the explicit `.next()` calls a test makes.
        queue = [MockQueryResult([])] + [MockQueryResult(p) for p in pages]

        def query_side_effect(*args, **kwargs):
            return queue.pop(0) if queue else MockQueryResult([])

        handler.query.side_effect = query_side_effect
        return QueryIterator(
            handler=handler,
            context=None,
            collection_name="test_collection",
            batch_size=10,
            expr="pk > 0",
            output_fields=["pk"],
            schema=schema,
            rpc_options={ITERATOR_SESSION_CP_FILE: str(cp_path)},
        )

    def test_empty_cp_file_starts_fresh_instead_of_raising(self, mock_handler, schema):
        """An empty checkpoint file, e.g. left behind by a process that
        crashed before writing anything, is treated like a missing file
        and does not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            cp_path.touch()  # exists, but zero bytes

            iterator = self.make_iterator(mock_handler, cp_path, schema, pages=[[]])

            assert iterator._session_ts > 0
            assert iterator._next_id is None

    def test_one_line_cp_file_restores_ts_without_cursor(self, mock_handler, schema):
        """A one-line checkpoint file (session_ts saved, no batch
        completed yet) restores the timestamp and starts with no cursor,
        instead of raising."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            cp_path.write_text("12345\n")

            iterator = self.make_iterator(mock_handler, cp_path, schema, pages=[[]])

            assert iterator._session_ts == 12345
            assert iterator._next_id is None

    def test_failed_checkpoint_save_does_not_advance_cursor(self, mock_handler, schema):
        """If the checkpoint write fails, the cursor must not advance,
        and a retry must re-deliver the batch that failed to save
        instead of skipping past it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            iterator = self.make_iterator(
                mock_handler, cp_path, schema, pages=[[{"pk": 1}, {"pk": 2}]]
            )
            # Use a fixed return_value instead of the queue: since the
            # cursor rolls back and the query filter is unchanged, a
            # real server would return the same page again on retry.
            mock_handler.query.side_effect = None
            mock_handler.query.return_value = MockQueryResult([{"pk": 1}, {"pk": 2}])

            prev_next_id = iterator._next_id
            prev_next_element_offset = iterator._next_element_offset

            broken_handle = Mock()
            broken_handle.writelines.side_effect = OSError("simulated disk failure")
            iterator._cp_file_handler = broken_handle

            with pytest.raises(Exception, match="failed to save pk cursor"):
                iterator.next()

            assert (
                iterator._next_id == prev_next_id
            ), "cursor must not advance when the checkpoint save fails"
            assert (
                iterator._next_element_offset == prev_next_element_offset
            ), "element offset must not advance when the checkpoint save fails"

            iterator._cp_file_handler = Mock()
            retried = iterator.next()
            assert [r["pk"] for r in retried] == [
                1,
                2,
            ], "retry after a failed save must re-deliver the same batch, not skip ahead"

    def test_successful_checkpoint_save_advances_cursor(self, mock_handler, schema):
        """Sanity check: the cursor still advances normally when the save
        succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            iterator = self.make_iterator(
                mock_handler, cp_path, schema, pages=[[{"pk": 1}, {"pk": 2}]]
            )

            ret = iterator.next()

            assert len(ret) == 2
            assert iterator._next_id == 2

    def test_failed_checkpoint_save_does_not_skip_cached_batch(self, mock_handler, schema):
        """A failed save must roll back the internal result cache too,
        not just the cursor fields -- otherwise a batch served from
        cache (rather than fetched fresh) is silently skipped on retry.

        With BATCH_SIZE=10 and a single 40-row query result:
          next() #1 -> caches rows 10-39, delivers rows 0-9 (fresh fetch)
          next() #2 -> serves rows 10-19 from cache, advances the cache
                       to rows 20-39. If the save fails here without a
                       cache rollback, a retry delivers rows 20-29 and
                       silently drops rows 10-19.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            batch_size = 10
            rows = [{"pk": i} for i in range(40)]
            iterator = self.make_iterator(mock_handler, cp_path, schema, pages=[rows])

            first = iterator.next()
            assert [r["pk"] for r in first] == list(range(batch_size))

            broken_handle = Mock()
            broken_handle.writelines.side_effect = OSError("simulated disk failure")
            iterator._cp_file_handler = broken_handle

            with pytest.raises(Exception, match="failed to save pk cursor"):
                iterator.next()

            iterator._cp_file_handler = Mock()
            retried = iterator.next()
            assert [r["pk"] for r in retried] == list(
                range(10, 20)
            ), "retry after a failed save must re-serve the same batch, not skip ahead"
