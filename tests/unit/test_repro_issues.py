"""Regression tests for pymilvus issues."""

import asyncio
import gc
import logging
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pymilvus import AnnSearchRequest, WeightedRanker
from pymilvus.client.abstract import CollectionSchema
from pymilvus.client.async_grpc_handler import AsyncGrpcHandler
from pymilvus.client.connection_manager import ConnectionManager
from pymilvus.client.grpc_handler import GrpcHandler
from pymilvus.client.types import ConsistencyLevel
from pymilvus.exceptions import ParamError
from pymilvus.grpc_gen import common_pb2
from pymilvus.grpc_gen import milvus_pb2 as milvus_types
from pymilvus.milvus_client.milvus_client import MilvusClient

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def reset_connection_manager():
    """Reset ConnectionManager singleton before and after each test."""
    ConnectionManager._reset_instance()
    yield
    ConnectionManager._reset_instance()


def make_client():
    mock_handler = MagicMock()
    mock_handler.get_server_type.return_value = "milvus"
    mock_handler._wait_for_channel_ready = MagicMock()
    with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=mock_handler):
        return MilvusClient()


def _open_fd_count():
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        return None
    return len(tuple(fd_dir.iterdir()))


def _collect_until_released(ref):
    for _ in range(3):
        gc.collect()
        if ref() is None:
            return
    assert ref() is None


def _pending_asyncio_tasks():
    current = asyncio.current_task()
    return {task for task in asyncio.all_tasks() if task is not current and not task.done()}


class _Issue3541FakeChannel:
    def __init__(self, name):
        self.name = name
        self.close_count = 0
        self.callbacks = []

    def close(self):
        self.close_count += 1

    def subscribe(self, callback, try_to_connect=False):
        self.callbacks.append((callback, try_to_connect))

    def unsubscribe(self, callback):
        self.callbacks = [item for item in self.callbacks if item[0] is not callback]

    def unary_unary(self, *args, **kwargs):
        return MagicMock()

    def unary_stream(self, *args, **kwargs):
        return MagicMock()

    def stream_unary(self, *args, **kwargs):
        return MagicMock()

    def stream_stream(self, *args, **kwargs):
        return MagicMock()


class _Issue3541VersionStub:
    def __init__(self, version, channel=None, started=None, release=None):
        self.version = version
        self.channel = channel
        self.started = started
        self.release = release

    def Connect(self, request, timeout=None):
        return SimpleNamespace(
            status=SimpleNamespace(code=0, error_code=0, reason=""),
            identifier=1,
        )

    def GetVersion(self, request, timeout=None, metadata=None):
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            assert self.release.wait(timeout=5)
        return SimpleNamespace(
            status=SimpleNamespace(code=0, error_code=0, reason=""),
            version=self.version,
        )


def test_issue_3541_sync_reconnect_swaps_without_waiting_for_paused_rpc():
    old_channel = _Issue3541FakeChannel("old")
    new_channel = _Issue3541FakeChannel("new")
    old_rpc_started = threading.Event()
    release_old_rpc = threading.Event()
    old_stub = _Issue3541VersionStub("old", old_channel, old_rpc_started, release_old_rpc)
    new_stub = _Issue3541VersionStub("new", new_channel)
    handler = GrpcHandler(
        uri="http://localhost:19530", address="localhost:19530", channel=old_channel
    )
    handler._stub = old_stub
    handler._final_channel = old_channel

    old_results = []
    old_rpc = threading.Thread(target=lambda: old_results.append(handler.get_server_version()))
    old_rpc.start()
    assert old_rpc_started.wait(timeout=1)
    reconnect_done = threading.Event()
    reconnect_errors = []

    def wait_for_new_channel(timeout=10, **kwargs):
        assert timeout == 30
        return kwargs["final_channel"], new_stub

    def run_reconnect():
        try:
            handler.reconnect(timeout=30)
        except Exception as exc:
            reconnect_errors.append(exc)
        finally:
            reconnect_done.set()

    reconnect = None
    try:
        with patch("pymilvus.client.grpc_handler.grpc.insecure_channel", return_value=new_channel):
            with patch.object(handler, "_wait_for_channel_ready", side_effect=wait_for_new_channel):
                reconnect = threading.Thread(target=run_reconnect)
                reconnect.start()
                assert reconnect_done.wait(timeout=5)

        assert reconnect_errors == []

        assert handler.get_server_version() == "new"
        assert old_channel.close_count == 0
    finally:
        release_old_rpc.set()
        old_rpc.join(timeout=1)
        if reconnect is not None:
            reconnect.join(timeout=1)

    assert old_results == ["old"]
    old_ref = weakref.ref(old_channel)
    old_stub.channel = None
    del old_stub
    del old_channel
    gc.collect()
    assert old_ref() is None


def test_issue_3541_sync_retired_channel_released_after_in_flight_rpc_drains():
    old_channel = _Issue3541FakeChannel("old")
    new_channel = _Issue3541FakeChannel("new")
    old_rpc_started = threading.Event()
    release_old_rpc = threading.Event()
    old_stub = _Issue3541VersionStub("old", old_channel, old_rpc_started, release_old_rpc)
    new_stub = _Issue3541VersionStub("new", new_channel)
    handler = GrpcHandler(
        uri="http://localhost:19530", address="localhost:19530", channel=old_channel
    )
    handler._stub = old_stub
    handler._final_channel = old_channel

    old_results = []
    old_rpc = threading.Thread(target=lambda: old_results.append(handler.get_server_version()))
    old_rpc.start()
    assert old_rpc_started.wait(timeout=1)

    def wait_for_new_channel(**kwargs):
        return kwargs["final_channel"], new_stub

    try:
        with patch("pymilvus.client.grpc_handler.grpc.insecure_channel", return_value=new_channel):
            with patch.object(handler, "_wait_for_channel_ready", side_effect=wait_for_new_channel):
                handler.reconnect()

        old_ref = weakref.ref(old_channel)
        assert handler.get_server_version() == "new"
        assert old_channel.close_count == 0
    finally:
        release_old_rpc.set()
        old_rpc.join(timeout=1)

    assert old_results == ["old"]
    old_stub.channel = None
    del old_stub
    del old_channel
    _collect_until_released(old_ref)


def test_issue_3541_sync_repeated_reconnects_do_not_leak_retired_channels_or_resources():
    fd_count_before = _open_fd_count()
    thread_count_before = len(threading.enumerate())
    handler = GrpcHandler(
        uri="http://localhost:19530",
        address="localhost:19530",
        channel=_Issue3541FakeChannel("initial"),
    )
    retired_refs = []

    def create_channel(address, options=None):
        return _Issue3541FakeChannel(f"replacement-{len(retired_refs)}")

    def wait_for_new_channel(**kwargs):
        return kwargs["final_channel"], kwargs["stub"]

    with patch(
        "pymilvus.client.grpc_handler.grpc.insecure_channel", side_effect=create_channel
    ), patch.object(handler, "_wait_for_channel_ready", side_effect=wait_for_new_channel):
        for _ in range(100):
            retired_channel = handler._channel
            retired_refs.append(weakref.ref(retired_channel))
            handler.reconnect()
            assert retired_channel.close_count == 0
            del retired_channel

    for retired_ref in retired_refs:
        _collect_until_released(retired_ref)

    fd_count_after = _open_fd_count()
    if fd_count_before is not None and fd_count_after is not None:
        assert fd_count_after <= fd_count_before
    assert len(threading.enumerate()) <= thread_count_before


def test_issue_3541_sync_failed_reconnect_keeps_current_connection():
    old_channel = _Issue3541FakeChannel("old")
    new_channel = _Issue3541FakeChannel("new")
    old_stub = _Issue3541VersionStub("old", old_channel)
    handler = GrpcHandler(
        uri="http://localhost:19530", address="localhost:19530", channel=old_channel
    )
    handler._stub = old_stub
    handler._final_channel = old_channel

    with patch("pymilvus.client.grpc_handler.grpc.insecure_channel", return_value=new_channel):
        with patch.object(
            handler,
            "_wait_for_channel_ready",
            side_effect=RuntimeError("replacement validation failed"),
        ):
            with pytest.raises(RuntimeError, match="replacement validation failed"):
                handler.reconnect()

    assert handler._channel is old_channel
    assert handler._stub is old_stub
    assert handler.get_server_version() == "old"
    assert old_channel.close_count == 0
    assert new_channel.close_count == 1


class TestIssue2587:
    """Regression test for #2587

    The error message for search/hybrid_search/flush with non-string collection
    name is not user friendly. Passing an integer (e.g. 1) should raise a clear
    ParamError rather than a cryptic gRPC internal error.
    """

    def test_issue_2587_search_non_string_collection_name(self):
        # Regression test for #2587
        client = make_client()
        with pytest.raises(ParamError, match="collection_name"):
            client.search(
                collection_name=1,
                data=[[0.1, 0.2]],
                limit=10,
            )

    def test_issue_2587_hybrid_search_non_string_collection_name(self):
        # Regression test for #2587
        client = make_client()
        reqs = [AnnSearchRequest([[0.1, 0.2]], "vec", {}, 10)]
        ranker = WeightedRanker(1.0)
        with pytest.raises(ParamError, match="collection_name"):
            client.hybrid_search(
                collection_name=1,
                reqs=reqs,
                ranker=ranker,
                limit=10,
            )

    def test_issue_2587_flush_non_string_collection_name(self):
        # Regression test for #2587
        client = make_client()
        with pytest.raises(ParamError, match="collection_name"):
            client.flush(collection_name=1)


class TestIssue2985:
    """Regression test for #2985

    PR #3409 (commit 07a29c8f) changed CollectionSchema.dict() to emit
    consistency_level as a string (ConsistencyLevel.Name(...)) instead of the
    historical int. This broke public API back-compat: downstream callers
    (including milvus e2e test_milvus_client_search_query_default) that
    indexed dict()['consistency_level'] expecting an int began failing.

    Dual-key fix: keep consistency_level as int (back-compat),
    add consistency_level_name as the human-readable string for callers who
    want the name. Users can migrate at leisure.
    """

    def _make_raw(self, consistency_level_int: int):
        """Build a minimal raw describe-collection response."""
        raw = MagicMock()
        raw.collection_name = "c"
        raw.auto_id = False
        raw.num_shards = 1
        raw.description = ""
        raw.schema = MagicMock()
        raw.schema.fields = []
        raw.schema.struct_array_fields = []
        raw.schema.functions = []
        raw.schema.enable_dynamic_field = False
        raw.schema.enable_namespace = False
        raw.schema.properties = []
        raw.aliases = []
        raw.collection_id = 0
        raw.consistency_level = consistency_level_int
        raw.properties = []
        raw.num_partitions = 1
        raw.created_timestamp = 0
        raw.update_timestamp = 0
        raw.external_source = None
        raw.external_spec = None
        raw.db_name = ""
        return raw

    def test_issue_2985_consistency_level_is_int_back_compat(self):
        # Regression test for #2985
        for level in (
            ConsistencyLevel.Strong,
            ConsistencyLevel.Session,
            ConsistencyLevel.Bounded,
            ConsistencyLevel.Eventually,
            ConsistencyLevel.Customized,
        ):
            raw = self._make_raw(int(level))
            d = CollectionSchema(raw).dict()
            # Back-compat: consistency_level MUST remain an int (the pre-#3409 shape).
            assert d["consistency_level"] == int(
                level
            ), f"Expected int {int(level)} for level {level}, got {d['consistency_level']!r}"

    def test_issue_2985_consistency_level_name_is_string(self):
        # Regression test for #2985
        expected = {
            ConsistencyLevel.Strong: "Strong",
            ConsistencyLevel.Session: "Session",
            ConsistencyLevel.Bounded: "Bounded",
            ConsistencyLevel.Eventually: "Eventually",
            ConsistencyLevel.Customized: "Customized",
        }
        for level, name in expected.items():
            raw = self._make_raw(int(level))
            d = CollectionSchema(raw).dict()
            # New key: consistency_level_name carries the human-readable string.
            assert d["consistency_level_name"] == name, (
                f"Expected name '{name}' for level {level}, got "
                f"{d.get('consistency_level_name')!r}"
            )


class _Issue3541FakeAsyncChannel:
    def __init__(self, name, wait_for_release=True):
        self.name = name
        self.closed = False
        self.close_count = 0
        self.close_grace = None
        self.close_started = asyncio.Event()
        self.closed_event = asyncio.Event()
        self.release_close = asyncio.Event()
        self.wait_for_release = wait_for_release
        self._unary_unary_interceptors = []

    async def close(self, grace=None):
        self.close_count += 1
        self.close_grace = grace
        self.close_started.set()
        if grace is None or not self.wait_for_release:
            self.closed = True
            self.closed_event.set()
            return
        await self.release_close.wait()
        self.closed = True
        self.closed_event.set()

    async def channel_ready(self):
        return None


class _Issue3541AsyncStub:
    def __init__(self, channel, rpc_entered=None, release_rpc=None):
        self.channel = channel
        self.rpc_entered = rpc_entered
        self.release_rpc = release_rpc

    async def DescribeCollection(self, request, timeout=None, metadata=None):
        if self.rpc_entered is not None:
            self.rpc_entered.set()
            await asyncio.wait_for(self.release_rpc.wait(), timeout=1)
        if self.channel.closed:
            raise ValueError("Cannot invoke RPC on closed channel!")
        return milvus_types.DescribeCollectionResponse(
            status=common_pb2.Status(error_code=common_pb2.Success)
        )

    async def Connect(self, request, timeout=None):
        return milvus_types.ConnectResponse(
            status=common_pb2.Status(error_code=common_pb2.Success),
            identifier=1,
        )


class _Issue3541RetiringAsyncChannel(_Issue3541FakeAsyncChannel):
    def __init__(self, name):
        super().__init__(name, wait_for_release=False)
        self.rpc_started = {}
        self.fast_rpc_can_finish = asyncio.Event()
        self.fast_rpc_finished = asyncio.Event()
        self.cancelled_rpcs = []
        self._rpc_tasks = {}

    def rpc_started_event(self, collection_name):
        return self.rpc_started.setdefault(collection_name, asyncio.Event())

    async def describe_collection(self, collection_name):
        task = asyncio.current_task()
        self._rpc_tasks[collection_name] = task
        self.rpc_started_event(collection_name).set()
        try:
            if collection_name == "fast":
                await self.fast_rpc_can_finish.wait()
                self.fast_rpc_finished.set()
            else:
                await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled_rpcs.append(collection_name)
            raise
        finally:
            self._rpc_tasks.pop(collection_name, None)

        if self.closed:
            raise ValueError("Cannot invoke RPC on closed channel!")
        return milvus_types.DescribeCollectionResponse(
            status=common_pb2.Status(error_code=common_pb2.Success)
        )

    async def close(self, grace=None):
        self.close_count += 1
        self.close_grace = grace
        self.close_started.set()
        self.fast_rpc_can_finish.set()
        await asyncio.wait_for(self.fast_rpc_finished.wait(), timeout=1)
        for name, task in list(self._rpc_tasks.items()):
            if name != "fast":
                task.cancel()
        await asyncio.gather(*list(self._rpc_tasks.values()), return_exceptions=True)
        self.closed = True
        self.closed_event.set()


class _Issue3541RetiringAsyncStub(_Issue3541AsyncStub):
    async def DescribeCollection(self, request, timeout=None, metadata=None):
        return await self.channel.describe_collection(request.collection_name)


class TestIssue3541:
    @pytest.mark.asyncio
    async def test_async_reconnect_gracefully_retires_in_flight_channel(self):
        old_channel = _Issue3541FakeAsyncChannel("old")
        new_channel = _Issue3541FakeAsyncChannel("new")
        rpc_entered = asyncio.Event()
        release_rpc = asyncio.Event()

        def stub_factory(channel):
            if channel is old_channel:
                return _Issue3541AsyncStub(channel, rpc_entered, release_rpc)
            return _Issue3541AsyncStub(channel)

        with patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
            side_effect=stub_factory,
        ), patch(
            "pymilvus.client.async_grpc_handler.grpc.aio.insecure_channel",
            return_value=new_channel,
        ):
            handler = AsyncGrpcHandler(uri="http://localhost:19530", channel=old_channel)

            rpc_task = asyncio.create_task(handler.has_collection("c"))
            await asyncio.wait_for(rpc_entered.wait(), timeout=1)

            await handler.reconnect(timeout=1)

            try:
                await asyncio.wait_for(old_channel.close_started.wait(), timeout=1)
                assert old_channel.close_grace == 1
                assert old_channel.closed is False
            finally:
                release_rpc.set()
                assert await asyncio.wait_for(rpc_task, timeout=1) is True
                old_channel.release_close.set()
                await asyncio.wait_for(old_channel.closed_event.wait(), timeout=1)
                assert old_channel.closed is True

    @pytest.mark.asyncio
    async def test_async_reconnect_close_grace_completes_and_cancels_in_flight_rpcs(self):
        old_channel = _Issue3541RetiringAsyncChannel("old")
        old_channel_holder = {"channel": old_channel}
        new_channel = _Issue3541FakeAsyncChannel("new", wait_for_release=False)

        def stub_factory(channel):
            if channel is old_channel_holder["channel"]:
                return _Issue3541RetiringAsyncStub(channel)
            return _Issue3541AsyncStub(channel)

        with patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
            side_effect=stub_factory,
        ), patch(
            "pymilvus.client.async_grpc_handler.grpc.aio.insecure_channel",
            return_value=new_channel,
        ):
            handler = AsyncGrpcHandler(uri="http://localhost:19530", channel=old_channel)
            fast_started = old_channel.rpc_started_event("fast")
            straggler_started = old_channel.rpc_started_event("straggler")
            fast_rpc = asyncio.create_task(handler.has_collection("fast"))
            straggler_rpc = asyncio.create_task(handler.has_collection("straggler"))
            await asyncio.wait_for(fast_started.wait(), timeout=1)
            await asyncio.wait_for(straggler_started.wait(), timeout=1)

            await handler.reconnect(timeout=0.25)
            await asyncio.wait_for(old_channel.closed_event.wait(), timeout=1)

            assert old_channel.close_grace == 0.25
            assert await asyncio.wait_for(fast_rpc, timeout=1) is True
            with pytest.raises(asyncio.CancelledError):
                await straggler_rpc
            assert old_channel.cancelled_rpcs == ["straggler"]

            retire_tasks = list(handler._retired_channel_close_tasks)
            if retire_tasks:
                await asyncio.gather(*retire_tasks)

        old_ref = weakref.ref(old_channel)
        old_channel_holder.clear()
        del old_channel
        _collect_until_released(old_ref)

    @pytest.mark.asyncio
    async def test_async_repeated_reconnects_do_not_leak_retired_channels_or_tasks(self):
        fd_count_before = _open_fd_count()
        pending_tasks_before = _pending_asyncio_tasks()
        retired_refs = []

        def stub_factory(channel):
            return _Issue3541AsyncStub(channel)

        def create_channel(address, options=None):
            return _Issue3541FakeAsyncChannel(
                f"replacement-{len(retired_refs)}", wait_for_release=False
            )

        with patch(
            "pymilvus.client.async_grpc_handler.milvus_pb2_grpc.MilvusServiceStub",
            side_effect=stub_factory,
        ), patch(
            "pymilvus.client.async_grpc_handler.grpc.aio.insecure_channel",
            side_effect=create_channel,
        ):
            handler = AsyncGrpcHandler(
                uri="http://localhost:19530",
                channel=_Issue3541FakeAsyncChannel("initial", wait_for_release=False),
            )
            for _ in range(100):
                retired_channel = handler._async_channel
                retired_refs.append(weakref.ref(retired_channel))
                await handler.reconnect(timeout=0.25)
                del retired_channel

            retire_tasks = list(handler._retired_channel_close_tasks)
            if retire_tasks:
                await asyncio.gather(*retire_tasks)

        for retired_ref in retired_refs:
            _collect_until_released(retired_ref)

        fd_count_after = _open_fd_count()
        if fd_count_before is not None and fd_count_after is not None:
            assert fd_count_after <= fd_count_before
        assert _pending_asyncio_tasks() <= pending_tasks_before
