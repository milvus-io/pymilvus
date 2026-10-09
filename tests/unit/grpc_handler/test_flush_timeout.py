"""Tests that synchronous flush APIs enforce ``timeout`` as one end-to-end budget.

Regression tests for the bug where ``GrpcHandler.flush``/``flush_all`` restarted a
fresh timer for the state-poll phase, so a single ``timeout`` could be overshot by
the time spent in the initial RPC (plus per-collection waits).
"""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus.exceptions import MilvusException


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it instead of blocking."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _make_flush_response():
    resp = MagicMock()
    resp.status.code = 0
    resp.status.error_code = 0
    resp.status.reason = ""
    resp.coll_segIDs = {"coll": MagicMock(data=[1, 2])}
    resp.coll_flush_ts = {"coll": 123}
    resp.flush_all_ts = 123
    return resp


class TestFlushTimeoutBudget:
    def test_flush_enforces_end_to_end_timeout_budget(self, handler):
        clock = FakeClock()
        resp = _make_flush_response()

        # The initial Flush RPC "spends" 8s of the 10s budget.
        calls = {"n": 0}

        def result_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                clock.now += 8
            return resp

        handler._stub.Flush.future.return_value.result.side_effect = result_side_effect

        # Segments never report flushed, so the wait loop must hit the deadline.
        state = MagicMock()
        state.status.code = 0
        state.status.error_code = 0
        state.status.reason = ""
        state.flushed = False
        handler._stub.GetFlushState.return_value = state

        with patch("pymilvus.client.grpc_handler.time", clock):
            start = clock.now
            with pytest.raises(MilvusException, match="wait for flush timeout"):
                handler.flush(["coll"], timeout=10)
            elapsed = clock.now - start

        # The whole operation (initial RPC + polling) must stay within the 10s
        # budget, not restart a fresh 10s timer after the 8s RPC (which used to
        # push the raise out to ~18s).
        assert elapsed <= 10.5, f"flush overshot its timeout budget: {elapsed}s"

    def test_flush_all_enforces_end_to_end_timeout_budget(self, handler):
        clock = FakeClock()
        resp = _make_flush_response()

        calls = {"n": 0}

        def result_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                clock.now += 8
            return resp

        handler._stub.FlushAll.future.return_value.result.side_effect = result_side_effect

        state = MagicMock()
        state.status.code = 0
        state.status.error_code = 0
        state.status.reason = ""
        state.flushed = False
        handler._stub.GetFlushAllState.return_value = state

        with patch("pymilvus.client.grpc_handler.time", clock):
            start = clock.now
            with pytest.raises(MilvusException, match="wait for flush all timeout"):
                handler.flush_all(timeout=10)
            elapsed = clock.now - start

        assert elapsed <= 10.5, f"flush_all overshot its timeout budget: {elapsed}s"

    def test_flush_without_timeout_still_waits_until_flushed(self, handler):
        clock = FakeClock()
        resp = _make_flush_response()
        handler._stub.Flush.future.return_value.result.return_value = resp

        # Not flushed on the first poll, flushed on the second.
        seq = iter([False, True])

        def get_flush_state(*args, **kwargs):
            state = MagicMock()
            state.status.code = 0
            state.status.error_code = 0
            state.status.reason = ""
            state.flushed = next(seq)
            return state

        handler._stub.GetFlushState.side_effect = get_flush_state

        with patch("pymilvus.client.grpc_handler.time", clock):
            handler.flush(["coll"])  # timeout=None: must not raise

        assert handler._stub.GetFlushState.call_count == 2
