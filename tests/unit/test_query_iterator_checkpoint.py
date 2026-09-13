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
    """Minimal stand-in for the real query response, with the `.extra`
    mapping the iterator reads the session timestamp from."""

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
        # When there's no valid checkpoint to read a session_ts from yet
        # (no cp file, or an empty one), the constructor itself issues one
        # internal query() call just to establish the timestamp -- before
        # any explicit `.next()` call happens. We prepend a dummy leading
        # response to absorb that call, so `pages` maps 1:1 onto the
        # actual `.next()` calls the test makes.
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
        """An empty checkpoint file (e.g. left behind by a process that
        crashed before writing anything) should be treated the same as
        no checkpoint file at all -- not raise ParamError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            cp_path.touch()  # exists, but zero bytes

            iterator = self.make_iterator(mock_handler, cp_path, schema, pages=[[]])

            assert iterator._session_ts > 0
            assert iterator._next_id is None

    def test_one_line_cp_file_restores_ts_without_cursor(self, mock_handler, schema):
        """A one-line checkpoint file (session_ts saved, but no batch
        completed yet -- e.g. crash right after the first save) should
        restore the timestamp and start with no cursor, not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            cp_path.write_text("12345\n")

            iterator = self.make_iterator(mock_handler, cp_path, schema, pages=[[]])

            assert iterator._session_ts == 12345
            assert iterator._next_id is None

    def test_failed_checkpoint_save_does_not_advance_cursor(self, mock_handler, schema):
        """If persisting the checkpoint fails, the in-memory cursor must
        not advance -- otherwise the batch that was already fetched from
        Milvus is silently dropped and skipped on retry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            iterator = self.make_iterator(
                mock_handler, cp_path, schema, pages=[[{"pk": 1}, {"pk": 2}]]
            )

            prev_next_id = iterator._next_id

            # Simulate a disk failure on the checkpoint write specifically,
            # independent of the (already-succeeded) Milvus query.
            broken_handle = Mock()
            broken_handle.writelines.side_effect = OSError("simulated disk failure")
            iterator._cp_file_handler = broken_handle

            with pytest.raises(Exception, match="failed to save pk cursor"):
                iterator.next()

            assert (
                iterator._next_id == prev_next_id
            ), "cursor must not advance when the checkpoint save fails"

    def test_successful_checkpoint_save_advances_cursor(self, mock_handler, schema):
        """Sanity check: the happy path still works after the fix --
        cursor advances normally when the save succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "cursor.cp"
            iterator = self.make_iterator(
                mock_handler, cp_path, schema, pages=[[{"pk": 1}, {"pk": 2}]]
            )

            ret = iterator.next()

            assert len(ret) == 2
            assert iterator._next_id == 2
