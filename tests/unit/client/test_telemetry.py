import asyncio
import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import grpc
import pymilvus.client.telemetry as telemetry_module
import pytest
from pymilvus.client.asynch import Future as PyMilvusFuture
from pymilvus.client.call_context import CallContext
from pymilvus.client.telemetry import (
    AsyncClientTelemetryManager,
    AsyncTelemetryUnaryUnaryInterceptor,
    ClientCommand,
    ClientTelemetryManager,
    CommandReply,
    Metrics,
    MetricsSnapshot,
    OperationMetrics,
    TelemetryConfig,
    TelemetryUnaryUnaryInterceptor,
    _request_id_from_metadata,
    _response_error,
    is_valid_client_request_id,
    new_client_request_id,
    telemetry_operation,
)
from pymilvus.decorators import retry_on_rpc_failure
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2, milvus_pb2


def test_config_hash_matches_server_algorithm():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=False))
    commands = [
        common_pb2.ClientCommand(
            command_id="cfg-b",
            command_type="push_config",
            payload=b'{"sampling_rate":0.5}',
            persistent=True,
        ),
        common_pb2.ClientCommand(
            command_id="cfg-a",
            command_type="push_config",
            payload=b'{"heartbeat_interval_ms":5000}',
            persistent=True,
        ),
    ]

    assert manager.calculate_config_hash(commands) == "a271ff0bb1941777"


def test_process_commands_is_idempotent_and_queues_replies():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    calls = []
    manager.register_command_handler(
        "custom",
        lambda command: (
            calls.append(command.command_id)
            or CommandReply(command.command_id, True, payload=b"ok")
        ),
    )
    command = common_pb2.ClientCommand(command_id="cmd-1", command_type="custom", create_time=1000)

    manager.process_commands([command])
    manager.process_commands([command])

    assert calls == ["cmd-1"]
    assert [reply.command_id for reply in manager._pending_replies] == ["cmd-1", "cmd-1"]


def test_custom_command_replies_use_server_id_and_none_becomes_failure():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager.register_command_handler("missing", lambda _command: None)

    manager.process_commands(
        [common_pb2.ClientCommand(command_id="missing-id", command_type="missing", create_time=1)]
    )

    missing = manager._pending_replies.pop()
    assert missing.command_id == "missing-id"
    assert missing.success is False
    assert missing.error_message == "command handler returned no reply"

    manager.register_command_handler(
        "wrong-id", lambda _command: CommandReply("other-id", False, "failed", b"payload")
    )
    manager.process_commands(
        [common_pb2.ClientCommand(command_id="server-id", command_type="wrong-id", create_time=2)]
    )

    canonical = manager._pending_replies.pop()
    assert canonical.command_id == "server-id"
    assert canonical.success is False
    assert canonical.error_message == "failed"
    assert canonical.payload == b"payload"


def test_builtin_config_and_error_commands():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager.record_operation("Search", "books", time.perf_counter(), RuntimeError("boom"))
    commands = [
        common_pb2.ClientCommand(
            command_id="cfg",
            command_type="push_config",
            payload=b'{"sampling_rate":0.5,"heartbeat_interval_ms":5000}',
            create_time=1,
            persistent=True,
        ),
        common_pb2.ClientCommand(
            command_id="errors",
            command_type="show_errors",
            payload=b'{"max_count":1}',
            create_time=2,
        ),
    ]

    manager.process_commands(commands)

    with manager._config_lock:
        assert manager._config.sampling_rate == 0.5
        assert manager._config.heartbeat_interval == 5.0
    errors_reply = next(reply for reply in manager._pending_replies if reply.command_id == "errors")
    assert json.loads(errors_reply.payload)[0]["error_msg"] == "boom"


def test_push_config_is_atomic_and_reports_applied_and_ignored_keys():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    with pytest.raises(ValueError, match="heartbeat_interval_ms must be positive"):
        manager._handle_push_config(
            ClientCommand(
                command_id="bad",
                command_type="push_config",
                payload=b'{"enabled":false,"heartbeat_interval_ms":0}',
            )
        )

    with manager._config_lock:
        assert manager._config.enabled is True
        assert manager._config.heartbeat_interval == 10.0

    reply = manager._handle_push_config(
        ClientCommand(
            command_id="good",
            command_type="push_config",
            payload=b'{"sampling_rate":2,"ttl_seconds":30,"future":"value"}',
        )
    )
    assert json.loads(reply.payload) == {
        "applied": ["sampling_rate"],
        "ignored": ["future", "ttl_seconds"],
    }
    with manager._config_lock:
        assert manager._config.sampling_rate == 1.0


@pytest.mark.parametrize(
    "payload,error",
    [
        (b'{"enabled":"false"}', "enabled must be a boolean"),
        (b'{"heartbeat_interval_ms":1.5}', "heartbeat_interval_ms must be an integer"),
        (b'{"sampling_rate":true}', "sampling_rate must be a number"),
        (b'{"sampling_rate":NaN}', "invalid JSON constant"),
    ],
)
def test_push_config_rejects_wrong_json_types(payload, error):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    with pytest.raises((TypeError, ValueError), match=error):
        manager._handle_push_config(
            ClientCommand(command_id="bad", command_type="push_config", payload=payload)
        )


def test_process_command_batches_are_serialized():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    calls = 0
    calls_lock = threading.Lock()
    second_lock_attempted = threading.Event()

    class ObservableLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._attempts = 0
            self._attempts_lock = threading.Lock()

        def __enter__(self):
            with self._attempts_lock:
                self._attempts += 1
                if self._attempts == 2:
                    second_lock_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._lock.release()

    # Signal the second acquisition attempt before it blocks on the real lock. The first
    # handler waits for that signal, so the test proves the two batches actually contend.
    manager._command_batch_lock = ObservableLock()

    def handler(command):
        nonlocal calls
        with calls_lock:
            calls += 1
        assert second_lock_attempted.wait(timeout=5)
        return CommandReply(command.command_id, True)

    manager.register_command_handler("custom", handler)
    command = common_pb2.ClientCommand(
        command_id="same-command", command_type="custom", create_time=1000
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(manager.process_commands, [command]) for _ in range(2)]
        for future in futures:
            future.result()

    assert second_lock_attempted.is_set()
    assert calls == 1


def test_equal_timestamp_command_stays_deduplicated_after_repeated_delivery():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    calls = 0

    def handler(command):
        nonlocal calls
        calls += 1
        return CommandReply(command.command_id, True)

    manager.register_command_handler("custom", handler)
    command = common_pb2.ClientCommand(
        command_id="same-command", command_type="custom", create_time=1000
    )

    manager.process_commands([command])
    manager.process_commands([command])
    manager.process_commands([command])

    assert calls == 1


def test_persistent_configs_bypass_one_time_command_state():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager.register_command_handler(
        "custom", lambda command: CommandReply(command.command_id, True)
    )
    manager.process_commands(
        [common_pb2.ClientCommand(command_id="cursor", command_type="custom", create_time=20)]
    )

    older_config = common_pb2.ClientCommand(
        command_id="database-config",
        command_type="push_config",
        payload=b'{"sampling_rate":0.25}',
        create_time=10,
        persistent=True,
    )
    manager.process_commands([older_config])

    assert manager.last_command_timestamp == 20
    assert manager.config_hash == manager.calculate_config_hash([older_config])
    with manager._config_lock:
        assert manager._config.sampling_rate == 0.25

    newer_config = common_pb2.ClientCommand(
        command_id="newer-database-config",
        command_type="push_config",
        payload=b'{"sampling_rate":0.75}',
        create_time=30,
        persistent=True,
    )
    manager.process_commands([newer_config])
    assert manager.last_command_timestamp == 20
    with manager._config_lock:
        assert manager._config.sampling_rate == 0.75

    # Returning to a previously delivered persistent set must reapply it even
    # though its command ID has already been observed.
    manager.process_commands([older_config])
    assert manager.last_command_timestamp == 20
    with manager._config_lock:
        assert manager._config.sampling_rate == 0.25


def test_empty_command_batch_preserves_hash_without_authoritative_config_snapshot():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    config = common_pb2.ClientCommand(
        command_id="database-config",
        command_type="push_config",
        payload=b'{"sampling_rate":0.25}',
        create_time=10,
        persistent=True,
    )

    manager.process_commands([config])
    assert manager.config_hash == manager.calculate_config_hash([config])

    manager.process_commands([])
    assert manager.config_hash == manager.calculate_config_hash([config])


class _RetryableRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class _UnimplementedRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNIMPLEMENTED


def test_logical_operation_counts_retries_once():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    class Handler:
        def __init__(self):
            self._telemetry = manager
            self.attempts = 0

        @telemetry_operation("Search")
        @retry_on_rpc_failure(retry_times=2, initial_back_off=0, max_back_off=0)
        def search(self, collection_name, context=None):
            self.attempts += 1
            if self.attempts == 1:
                raise _RetryableRpcError
            return "parsed-result"

    handler = Handler()

    assert handler.search("books") == "parsed-result"
    assert handler.attempts == 2
    collector = manager._collectors["Search"]
    with collector.lock:
        assert collector.global_bucket.request_count == 1
        assert collector.global_bucket.success_count == 1
        assert collector.global_bucket.error_count == 0


@pytest.mark.asyncio
async def test_async_logical_operation_counts_retries_once_and_records_final_error():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    request_id = new_client_request_id()

    class Handler:
        def __init__(self):
            self._telemetry = manager
            self.attempts = 0

        @telemetry_operation("Query")
        @retry_on_rpc_failure(retry_times=1, initial_back_off=0, max_back_off=0)
        async def query(self, collection_name, context=None):
            self.attempts += 1
            raise _RetryableRpcError

    handler = Handler()

    with pytest.raises(MilvusException):
        await handler.query("books", context=CallContext(client_request_id=request_id))

    collector = manager._collectors["Query"]
    with collector.lock:
        assert collector.global_bucket.request_count == 1
        assert collector.global_bucket.success_count == 0
        assert collector.global_bucket.error_count == 1
    assert manager.get_recent_errors(1)[0].request_id == request_id


def test_logical_operation_suppresses_per_rpc_interceptor_recording():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    interceptor = TelemetryUnaryUnaryInterceptor(manager)

    class ImmediateCall:
        @staticmethod
        def exception():
            return None

        @staticmethod
        def result():
            return milvus_pb2.QueryResults(status=common_pb2.Status())

        def add_done_callback(self, callback):
            callback(self)

    class Handler:
        _telemetry = manager

        @telemetry_operation("Query")
        def query(self, collection_name, context=None):
            details = SimpleNamespace(
                method="/milvus.proto.milvus.MilvusService/Query", metadata=()
            )
            request = SimpleNamespace(collection_name=collection_name)
            return interceptor.intercept_unary_unary(
                lambda *_args: ImmediateCall(), details, request
            )

    Handler().query("books")

    collector = manager._collectors["Query"]
    with collector.lock:
        assert collector.global_bucket.request_count == 1


def test_public_client_local_validation_is_included_in_logical_metric():
    from pymilvus import MilvusClient  # noqa: PLC0415

    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    client = MilvusClient.__new__(MilvusClient)
    client._handler = SimpleNamespace(_telemetry=manager)

    with pytest.raises(TypeError, match="wrong type of argument 'data'"):
        client.insert("books", "not-rows")

    collector = manager._collectors["Insert"]
    with collector.lock:
        assert collector.global_bucket.request_count == 1
        assert collector.global_bucket.error_count == 1


def test_sync_get_and_session_get_are_counted_once_as_query():
    from pymilvus import MilvusClient  # noqa: PLC0415

    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    class Handler:
        def __init__(self):
            self._telemetry = manager
            self.query_calls = 0

        def _get_schema(self, *_args, **_kwargs):
            return ({"fields": []}, None)

        @telemetry_operation("Query")
        def query(self, *_args, **_kwargs):
            self.query_calls += 1
            return []

    handler = Handler()
    client = MilvusClient.__new__(MilvusClient)
    client._handler = handler
    client._config = SimpleNamespace(db_name="default")
    client._pack_pks_expr = lambda _schema, _ids: "id in [1]"

    assert client.get("books", ids=[1]) == []
    assert client.session("cluster-1").get("books", ids=[2]) == []
    assert client.get("books", ids=[]) == []

    collector = manager._collectors["Query"]
    with collector.lock:
        assert collector.global_bucket.request_count == 3
        assert collector.global_bucket.success_count == 3
    assert handler.query_calls == 2


@pytest.mark.asyncio
async def test_async_get_and_session_get_are_counted_once_as_query():
    from pymilvus import AsyncMilvusClient  # noqa: PLC0415

    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    class Handler:
        def __init__(self):
            self._telemetry = manager
            self.query_calls = 0

        async def _get_schema(self, *_args, **_kwargs):
            return ({"fields": []}, None)

        @telemetry_operation("Query")
        async def query(self, *_args, **_kwargs):
            self.query_calls += 1
            return []

    handler = Handler()
    client = AsyncMilvusClient.__new__(AsyncMilvusClient)
    client._handler = handler
    client._config = SimpleNamespace(db_name="default")
    client._closed = False
    client._pack_pks_expr = lambda _schema, _ids: "id in [1]"

    assert await client.get("books", ids=[1]) == []
    assert await client.session("cluster-1").get("books", ids=[2]) == []
    assert await client.get("books", ids=[]) == []

    collector = manager._collectors["Query"]
    with collector.lock:
        assert collector.global_bucket.request_count == 3
        assert collector.global_bucket.success_count == 3
    assert handler.query_calls == 2


def test_sync_session_keyword_collection_name_is_attributed():
    from pymilvus import MilvusClient  # noqa: PLC0415

    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager._handle_collection_metrics(
        ClientCommand(
            command_id="enable-books",
            command_type="collection_metrics",
            payload=b'{"enabled":true,"collections":["books"]}',
        )
    )

    class Handler:
        _telemetry = manager

        @staticmethod
        def query(*_args, **_kwargs):
            return []

    client = MilvusClient.__new__(MilvusClient)
    client._handler = Handler()
    client._config = SimpleNamespace(db_name="")

    assert client.session("cluster-1").query(collection_name="books") == []

    collector = manager._collectors["Query"]
    with collector.lock:
        assert collector.global_bucket.request_count == 1
        assert collector.collections["books"].request_count == 1


@pytest.mark.asyncio
async def test_async_session_keyword_collection_name_is_attributed():
    from pymilvus import AsyncMilvusClient  # noqa: PLC0415

    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager._handle_collection_metrics(
        ClientCommand(
            command_id="enable-books",
            command_type="collection_metrics",
            payload=b'{"enabled":true,"collections":["books"]}',
        )
    )

    class Handler:
        _telemetry = manager

        @staticmethod
        async def query(*_args, **_kwargs):
            return []

    client = AsyncMilvusClient.__new__(AsyncMilvusClient)
    client._closed = False
    client._handler = Handler()
    client._config = SimpleNamespace(db_name="")

    assert await client.session("cluster-1").query(collection_name="books") == []

    collector = manager._collectors["Query"]
    with collector.lock:
        assert collector.global_bucket.request_count == 1
        assert collector.collections["books"].request_count == 1


def test_collection_metrics_are_filtered_again_at_wire_time():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager._handle_collection_metrics(
        ClientCommand(
            command_id="enable",
            command_type="collection_metrics",
            payload=b'{"enabled":true,"collections":["books"]}',
        )
    )
    snapshot_metrics = [
        OperationMetrics(
            "Search",
            Metrics(request_count=1, success_count=1),
            {"books": Metrics(request_count=1, success_count=1)},
        )
    ]
    manager._handle_collection_metrics(
        ClientCommand(
            command_id="disable",
            command_type="collection_metrics",
            payload=b'{"enabled":false,"collections":["books"]}',
        )
    )

    assert dict(manager._to_proto_metrics(snapshot_metrics)[0].collection_metrics) == {}


def test_collection_metrics_rejects_coerced_json_types():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    with pytest.raises(TypeError, match="enabled must be a boolean"):
        manager._handle_collection_metrics(
            ClientCommand(
                command_id="bad",
                command_type="collection_metrics",
                payload=b'{"enabled":"true","collections":["books"]}',
            )
        )


@pytest.mark.parametrize(
    "start_time",
    [
        "2026-08-23T12:34Z",
        "2026-08-23T12:34:00",
        "2026-08-23",
        "2026-02-30T12:34:00Z",
    ],
)
def test_latency_history_requires_strict_rfc3339(start_time):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    with pytest.raises(ValueError):
        manager._handle_latency_history(
            ClientCommand(
                command_id="bad-history",
                command_type="show_latency_history",
                payload=json.dumps(
                    {
                        "start_time": start_time,
                        "end_time": "2026-08-23T12:35:00Z",
                    }
                ).encode(),
            )
        )


def test_latency_history_accepts_rfc3339_offset_and_long_fraction():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    reply = manager._handle_latency_history(
        ClientCommand(
            command_id="valid-history",
            command_type="show_latency_history",
            payload=(
                b'{"start_time":"2026-08-23T12:34:00.123456789123+08:00",'
                b'"end_time":"2026-08-23T12:35:00.987654321987+08:00"}'
            ),
        )
    )

    assert reply.success is True


def test_real_heartbeat_response_clears_unsupported_backoff_on_status_error():
    class Stub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(error_code=common_pb2.UnexpectedError, reason="failed")
            )

    manager = ClientTelemetryManager(Stub, TelemetryConfig(enabled=True))
    manager._unsupported_streak = 3

    manager._send_heartbeat()

    assert manager.is_supported() is True
    assert str(manager.last_heartbeat_error()) == "failed"


@pytest.mark.parametrize("outcome", ["response", "error"])
def test_rebind_fences_stale_sync_heartbeat_state(outcome):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    original_error = RuntimeError("original")
    manager._unsupported_streak = 3
    manager._last_heartbeat_error = original_error
    manager._queue_reply(CommandReply("pending", True))

    class OldStub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            manager.rebind_stub(object())
            if outcome == "error":
                raise _UnimplementedRpcError
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(),
                commands=[
                    common_pb2.ClientCommand(
                        command_id="stale-config",
                        command_type="push_config",
                        payload=b'{"enabled":false}',
                        create_time=10,
                        persistent=True,
                    )
                ],
            )

    manager.rebind_stub(OldStub())
    manager._send_heartbeat()

    assert manager._unsupported_streak == 3
    assert manager.last_heartbeat_error() is original_error
    assert [reply.command_id for reply in manager._pending_replies] == ["pending"]
    assert manager.last_command_timestamp == 0
    with manager._config_lock:
        assert manager._config.enabled is True


def test_reentrant_sync_rebind_stops_old_command_batch_until_redelivery():
    commands = [
        common_pb2.ClientCommand(
            command_id="switch", command_type="switch_endpoint", create_time=10
        ),
        common_pb2.ClientCommand(
            command_id="config",
            command_type="push_config",
            payload=b'{"sampling_rate":0.5}',
            create_time=20,
            persistent=True,
        ),
    ]
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    class NewStub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status(), commands=commands)

    def switch_endpoint(_command):
        manager.rebind_stub(NewStub())
        return CommandReply("wrong-id", True)

    manager.register_command_handler("switch_endpoint", switch_endpoint)

    class OldStub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status(), commands=commands)

    manager.rebind_stub(OldStub())
    manager._send_heartbeat()

    assert [reply.command_id for reply in manager._pending_replies] == ["switch"]
    assert manager.last_command_timestamp == 0
    with manager._config_lock:
        assert manager._config.sampling_rate == 1.0

    manager._send_heartbeat()

    assert [reply.command_id for reply in manager._pending_replies] == ["switch", "config"]
    assert manager.last_command_timestamp == 10
    with manager._config_lock:
        assert manager._config.sampling_rate == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["response", "error"])
async def test_rebind_fences_stale_async_heartbeat_state(outcome):
    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    original_error = RuntimeError("original")
    manager._unsupported_streak = 3
    manager._last_heartbeat_error = original_error
    manager._queue_reply(CommandReply("pending", True))

    class OldStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            manager.rebind_stub(object())
            if outcome == "error":
                raise _UnimplementedRpcError
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(),
                commands=[
                    common_pb2.ClientCommand(
                        command_id="stale-config",
                        command_type="push_config",
                        payload=b'{"enabled":false}',
                        create_time=10,
                        persistent=True,
                    )
                ],
            )

    manager.rebind_stub(OldStub())
    await manager._send_heartbeat_async()

    assert manager._unsupported_streak == 3
    assert manager.last_heartbeat_error() is original_error
    assert [reply.command_id for reply in manager._pending_replies] == ["pending"]
    assert manager.last_command_timestamp == 0
    with manager._config_lock:
        assert manager._config.enabled is True


@pytest.mark.asyncio
async def test_reentrant_async_rebind_stops_old_command_batch_until_redelivery():
    commands = [
        common_pb2.ClientCommand(
            command_id="switch", command_type="switch_endpoint", create_time=10
        ),
        common_pb2.ClientCommand(
            command_id="config",
            command_type="push_config",
            payload=b'{"sampling_rate":0.25}',
            create_time=20,
            persistent=True,
        ),
    ]
    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    class NewStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status(), commands=commands)

    def switch_endpoint(_command):
        manager.rebind_stub(NewStub())
        return CommandReply("", True)

    manager.register_command_handler("switch_endpoint", switch_endpoint)

    class OldStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status(), commands=commands)

    manager.rebind_stub(OldStub())
    await manager._send_heartbeat_async()

    assert [reply.command_id for reply in manager._pending_replies] == ["switch"]
    assert manager.last_command_timestamp == 0
    with manager._config_lock:
        assert manager._config.sampling_rate == 1.0

    await manager._send_heartbeat_async()

    assert [reply.command_id for reply in manager._pending_replies] == ["switch", "config"]
    assert manager.last_command_timestamp == 10
    with manager._config_lock:
        assert manager._config.sampling_rate == 0.25


def test_show_errors_uses_default_for_non_positive_count_and_empty_payload_for_no_errors():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    reply = manager._handle_show_errors(
        ClientCommand(command_id="errors", command_type="show_errors", payload=b'{"max_count":0}')
    )

    assert reply.success is True
    assert reply.payload == b""


def test_new_client_request_id_is_valid_trace_id():
    request_id = new_client_request_id()
    assert len(request_id) == 32
    assert request_id != "0" * 32
    int(request_id, 16)
    assert is_valid_client_request_id(request_id)


@pytest.mark.parametrize(
    "request_id",
    ["", "0" * 32, "A" * 32, "g" * 32, "0123456789abcdef"],
)
def test_invalid_client_request_id_is_not_recorded(request_id):
    assert not is_valid_client_request_id(request_id)
    assert _request_id_from_metadata((("client-request-id", request_id),)) == ""
    wire_metadata = dict(CallContext(client_request_id=request_id).to_grpc_metadata())
    if request_id:
        assert wire_metadata["client-request-id"] == request_id
    else:
        assert "client-request-id" not in wire_metadata


def test_valid_client_request_id_is_forwarded_in_grpc_metadata():
    request_id = new_client_request_id()

    assert (
        dict(CallContext(client_request_id=request_id).to_grpc_metadata())["client-request-id"]
        == request_id
    )


def test_response_error_detects_milvus_status_failure():
    response = milvus_pb2.QueryResults(
        status=common_pb2.Status(error_code=common_pb2.CollectionNotExists, reason="missing")
    )

    error = _response_error(response)

    assert str(error) == "missing"


def test_response_error_accepts_success_status():
    response = milvus_pb2.QueryResults(status=common_pb2.Status())

    assert _response_error(response) is None


def test_runtime_client_id_is_reused_without_becoming_stable():
    manager = ClientTelemetryManager(
        lambda: None,
        TelemetryConfig(enabled=False),
        runtime_client_id="runtime-client-id",
    )

    assert manager.client_id == "runtime-client-id"
    assert manager._client_id_stable is False


def test_show_errors_truncates_a_single_large_error():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager.record_operation(
        "Query",
        "books",
        time.perf_counter(),
        RuntimeError("x" * (2 * 1024 * 1024)),
    )

    reply = manager._handle_show_errors(
        ClientCommand(command_id="errors", command_type="show_errors")
    )

    assert reply.success is True
    assert len(reply.payload) <= 1024 * 1024


def test_show_errors_truncates_oversized_non_message_fields_without_mutating_history():
    collection = "x" * (1024 * 1024 + 100)
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager.record_operation(
        "Query", collection, time.perf_counter(), RuntimeError("short message")
    )

    reply = manager._handle_show_errors(
        ClientCommand(command_id="errors", command_type="show_errors")
    )

    assert reply.success is True
    assert len(reply.payload) <= 1024 * 1024
    assert json.loads(reply.payload)[0]["collection"].endswith("...(truncated)")
    assert manager.get_recent_errors(1)[0].collection == collection


def test_metrics_snapshot_sampling_and_collection_scope():
    manager = ClientTelemetryManager(
        lambda: None,
        TelemetryConfig(enabled=True),
        database_provider=lambda: "analytics",
    )
    manager._handle_collection_metrics(
        ClientCommand(
            command_id="enable-books",
            command_type="collection_metrics",
            payload=b'{"enabled":true,"collections":["books"]}',
        )
    )

    assert manager._should_sample(0.0) is False
    assert [manager._should_sample(0.25) for _ in range(4)] == [False, False, False, True]

    started_at = time.perf_counter() - 0.01
    manager.record_operation("Search", "books", started_at)
    manager.record_operation("Search", "private", started_at, RuntimeError("failed"))
    manager._create_snapshot()

    snapshot = manager.get_metrics_snapshots()[-1]
    search = snapshot.metrics[0]
    assert search.operation == "Search"
    assert search.global_metrics.request_count == 2
    assert search.global_metrics.success_count == 1
    assert search.global_metrics.error_count == 1
    assert search.global_metrics.avg_latency_ms > 0
    assert search.collection_metrics["books"].request_count == 1
    assert "private" not in search.collection_metrics

    # A second window resets the collector and must not repeat the previous metrics.
    manager._create_snapshot()
    assert manager.get_metrics_snapshots()[-1].metrics == []
    assert manager._build_client_info().reserved["db_name"] == "analytics"
    assert manager.config_hash == ""
    assert manager.ready is False

    manager._unsupported_streak = 3
    assert manager._next_heartbeat_delay() == 80.0
    manager._unsupported_streak = 4096
    assert manager._next_heartbeat_delay() == 1800.0

    disabled = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=False))
    disabled.record_operation("Query", "books", time.perf_counter())
    disabled._create_snapshot()
    assert disabled.get_metrics_snapshots() == []


def test_snapshot_history_uses_one_hour_ttl_with_independent_hard_cap(monkeypatch):
    now_ms = 2_000_000_000_000
    monkeypatch.setattr(telemetry_module.time, "time", lambda: now_ms / 1000)
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    hour_ms = 60 * 60 * 1000
    manager._snapshots.extend(
        [
            MetricsSnapshot(now_ms - hour_ms - 2, now_ms - hour_ms - 1, []),
            MetricsSnapshot(now_ms - hour_ms, now_ms - hour_ms, []),
            MetricsSnapshot(now_ms - hour_ms // 2, now_ms - hour_ms // 2, []),
        ]
    )

    manager._handle_push_config(
        ClientCommand(
            command_id="interval",
            command_type="push_config",
            payload=b'{"heartbeat_interval_ms":600000}',
        )
    )
    retained = manager.get_metrics_snapshots()
    assert [snapshot.end_time for snapshot in retained] == [
        now_ms - hour_ms,
        now_ms - hour_ms // 2,
    ]

    manager._snapshots.clear()
    manager._snapshots.extend(
        MetricsSnapshot(now_ms + index, now_ms + index, []) for index in range(4097)
    )
    retained = manager.get_metrics_snapshots()
    assert len(retained) == 4096
    assert retained[0].timestamp == now_ms + 1


def test_command_queries_redaction_collection_modes_and_failure_replies():
    manager = ClientTelemetryManager(
        lambda: None,
        TelemetryConfig(enabled=True),
        config_provider=lambda: {
            "password": "secret",
            "token": "secret",
            "api_key": "secret",
            "address": "localhost:19530",
        },
    )

    initial = manager._handle_collection_metrics(
        ClientCommand(command_id="state", command_type="collection_metrics")
    )
    assert json.loads(initial.payload) == {
        "enabled_collections": [],
        "all_collections_enabled": False,
    }

    with pytest.raises(ValueError, match="collections list cannot be empty"):
        manager._handle_collection_metrics(
            ClientCommand(
                command_id="bad-enable",
                command_type="collection_metrics",
                payload=b'{"enabled":true,"collections":[]}',
            )
        )

    manager._handle_collection_metrics(
        ClientCommand(
            command_id="all",
            command_type="collection_metrics",
            payload=b'{"enabled":true,"collections":["*"]}',
        )
    )
    config = manager._handle_get_config(
        ClientCommand(command_id="config", command_type="get_config")
    )
    user_config = json.loads(config.payload)["user_config"]
    assert user_config["address"] == "localhost:19530"
    assert user_config["enabled_collections"] == ["*"]
    assert user_config["all_collections_enabled"] is True
    assert not {"password", "token", "api_key"} & user_config.keys()

    manager._handle_collection_metrics(
        ClientCommand(
            command_id="none",
            command_type="collection_metrics",
            payload=b'{"enabled":false,"collections":[]}',
        )
    )
    assert manager._all_collections_enabled is False
    assert manager._enabled_collections == set()

    push = manager._handle_push_config(
        ClientCommand(
            command_id="disable",
            command_type="push_config",
            payload=b'{"enabled":false}',
        )
    )
    assert json.loads(push.payload) == {"applied": ["enabled"]}
    assert manager._config.enabled is False

    unknown = manager._handle_command(ClientCommand("unknown", "not-registered"))
    assert unknown.success is False
    assert "unknown command type" in unknown.error_message

    manager.register_command_handler("raises", lambda _command: 1 / 0)
    failed = manager._handle_command(ClientCommand("failed", "raises"))
    assert failed.success is False
    assert "division by zero" in failed.error_message

    manager._last_command_timestamp = 10
    manager.process_commands(
        [common_pb2.ClientCommand(command_id="stale", command_type="push_config", create_time=9)]
    )
    assert manager._pending_replies[-1].command_id == "stale"
    assert manager.calculate_config_hash([]) == ""


def test_unprintable_command_handler_error_does_not_stop_sync_heartbeat_loop():
    requests = []

    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("formatting failed")

    class Stub:
        @staticmethod
        def ClientHeartbeat(request, **_kwargs):
            requests.append(request)
            if len(requests) == 1:
                return milvus_pb2.ClientHeartbeatResponse(
                    status=common_pb2.Status(),
                    commands=[
                        common_pb2.ClientCommand(
                            command_id="unprintable",
                            command_type="custom",
                            create_time=1,
                        )
                    ],
                )
            manager._stop_event.set()
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status())

    manager = ClientTelemetryManager(Stub, TelemetryConfig(enabled=True))

    def raise_unprintable(_command):
        raise UnprintableError

    manager.register_command_handler("custom", raise_unprintable)
    manager._create_snapshot = lambda: None
    manager._next_heartbeat_delay = lambda: 0

    manager._heartbeat_loop()

    assert len(requests) == 2
    reply = requests[1].command_replies[0]
    assert reply.command_id == "unprintable"
    assert reply.success is False
    assert reply.error_message == "UnprintableError (failed to format exception)"


@pytest.mark.parametrize(
    "command,error",
    [
        (
            ClientCommand("array", "push_config", payload=b"[]"),
            "command payload must be a JSON object",
        ),
        (
            ClientCommand("finite", "push_config", payload=b'{"sampling_rate":1e309}'),
            "sampling_rate must be finite",
        ),
        (
            ClientCommand(
                "strings",
                "collection_metrics",
                payload=b'{"enabled":true,"collections":[1]}',
            ),
            "collections must be an array of strings",
        ),
    ],
)
def test_command_payload_rejects_strict_json_edge_cases(command, error):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    reply = manager._handle_command(command)
    assert reply.success is False
    assert error in reply.error_message


def test_config_validation_covers_invalid_type_interval_and_error_limit_default():
    with pytest.raises(TypeError, match="telemetry_config must be"):
        TelemetryConfig.from_value("enabled")
    with pytest.raises(ValueError, match="heartbeat_interval must be positive"):
        TelemetryConfig(heartbeat_interval=0)
    assert TelemetryConfig(error_max_count=0).error_max_count == 100


def test_telemetry_config_is_copied_for_each_manager():
    supplied = TelemetryConfig(enabled=True, sampling_rate=0.75)
    first = ClientTelemetryManager(lambda: None, supplied)
    second = ClientTelemetryManager(lambda: None, supplied)

    first._handle_push_config(
        ClientCommand(
            command_id="disable",
            command_type="push_config",
            payload=b'{"enabled":false,"sampling_rate":0.25}',
        )
    )

    assert supplied.enabled is True
    assert supplied.sampling_rate == 0.75
    assert second._config.enabled is True
    assert second._config.sampling_rate == 0.75


def test_latency_history_detail_aggregate_redaction_and_range_validation(monkeypatch):
    monkeypatch.setattr(telemetry_module.time, "time", lambda: 3)
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager._snapshots.append(
        MetricsSnapshot(
            timestamp=1000,
            end_time=2000,
            metrics=[
                OperationMetrics(
                    "Search",
                    Metrics(
                        request_count=2,
                        success_count=1,
                        error_count=1,
                        avg_latency_ms=4.0,
                        p99_latency_ms=7.0,
                        max_latency_ms=9.0,
                    ),
                )
            ],
        )
    )

    def history(payload):
        return manager._handle_latency_history(
            ClientCommand(
                command_id="history",
                command_type="show_latency_history",
                payload=json.dumps(payload).encode(),
            )
        )

    window = {
        "start_time": "1970-01-01T00:00:00Z",
        "end_time": "1970-01-01T00:00:03Z",
    }
    aggregate = json.loads(history(window).payload)
    assert aggregate["snapshot_count"] == 1
    assert aggregate["aggregated"]["metrics"]["Search"] == {
        "request_count": 2,
        "success_count": 1,
        "error_count": 1,
        "avg_latency_ms": 4.0,
        "p99_latency_ms": 7.0,
        "max_latency_ms": 9.0,
    }

    detail = json.loads(history({**window, "detail": True}).payload)
    assert detail["total_snapshots"] == 1
    assert detail["snapshots"][0]["metrics"]["Search"]["request_count"] == 2

    with pytest.raises(TypeError, match="start_time must be a string"):
        history({"start_time": 1, "end_time": window["end_time"]})
    with pytest.raises(TypeError, match="end_time must be a string"):
        history({"start_time": window["start_time"], "end_time": 3})
    with pytest.raises(ValueError, match="end_time must be after start_time"):
        history(
            {
                "start_time": "1970-01-01T00:00:03Z",
                "end_time": "1970-01-01T00:00:02Z",
            }
        )
    with pytest.raises(ValueError, match="time range cannot exceed 1 hour"):
        history(
            {
                "start_time": "1970-01-01T00:00:00Z",
                "end_time": "1970-01-01T01:00:01Z",
            }
        )
    with pytest.raises(ValueError, match="payload is required"):
        history({})

    monkeypatch.setattr(telemetry_module, "_MAX_REPLY_PAYLOAD_SIZE", 1)
    with pytest.raises(ValueError, match="response too large"):
        history(window)


def test_latency_history_aggregates_weighted_samples_without_flattening_sort(monkeypatch):
    bucket = telemetry_module._MetricsBucket()
    for latency_us in range(1000):
        bucket.record(latency_us, True)
    _, retained = bucket.snapshot_and_reset(retain_history_samples=True)
    assert len(retained) == 128
    assert retained[0] == 0
    assert retained[-1] == 999

    snapshots = [
        MetricsSnapshot(
            timestamp=0,
            end_time=1000,
            metrics=[
                OperationMetrics(
                    "Search",
                    Metrics(
                        request_count=100,
                        success_count=100,
                        avg_latency_ms=1.0,
                        p99_latency_ms=1.0,
                        max_latency_ms=1.0,
                    ),
                    _global_latency_samples_us=telemetry_module.array("q", [1000]),
                )
            ],
        ),
        MetricsSnapshot(
            timestamp=1000,
            end_time=2000,
            metrics=[
                OperationMetrics(
                    "Search",
                    Metrics(
                        request_count=100,
                        success_count=100,
                        avg_latency_ms=100.0,
                        p99_latency_ms=100.0,
                        max_latency_ms=100.0,
                    ),
                    _global_latency_samples_us=telemetry_module.array("q", [100_000]),
                )
            ],
        ),
    ]

    def reject_flattening_sort(*_args, **_kwargs):
        pytest.fail("history aggregation must merge retained sorted samples without sorting a copy")

    monkeypatch.setattr(telemetry_module, "sorted", reject_flattening_sort, raising=False)
    aggregate = telemetry_module._aggregate_snapshots(snapshots, 0, 2000)
    search = aggregate["aggregated"]["metrics"]["Search"]
    assert search["avg_latency_ms"] == 50.5
    assert search["p99_latency_ms"] == 100.0
    assert set(telemetry_module._snapshot_dict(snapshots[0])["metrics"]["Search"]) == {
        "request_count",
        "success_count",
        "error_count",
        "avg_latency_ms",
        "p99_latency_ms",
        "max_latency_ms",
    }


@pytest.mark.parametrize("heartbeat_interval", [math.nan, math.inf, -math.inf])
def test_config_rejects_non_finite_heartbeat_intervals(heartbeat_interval):
    with pytest.raises(ValueError, match="fit in signed 64-bit milliseconds"):
        TelemetryConfig(heartbeat_interval=heartbeat_interval)


def test_config_rejects_heartbeat_interval_beyond_signed_int64_milliseconds():
    with pytest.raises(ValueError, match="fit in signed 64-bit milliseconds"):
        TelemetryConfig(heartbeat_interval=telemetry_module._MAX_HEARTBEAT_INTERVAL * 2)


def test_sync_long_heartbeat_interval_uses_cancellable_wait_chunks(monkeypatch):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    waits = []

    class RecordingStopEvent:
        def wait(self, delay):
            waits.append(delay)
            return len(waits) == 3

    monkeypatch.setattr(telemetry_module, "_MAX_WAIT_CHUNK_SECONDS", 2.0)
    manager._stop_event = RecordingStopEvent()

    assert manager._wait_for_stop(10.0) is True
    assert waits == [2.0, 2.0, 2.0]


def test_max_int64_push_interval_remains_representable():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager._handle_push_config(
        ClientCommand(
            command_id="max-interval",
            command_type="push_config",
            payload=f'{{"heartbeat_interval_ms":{2**63 - 1}}}'.encode(),
        )
    )

    assert manager._heartbeat_interval_ms() == 2**63 - 1
    reply = manager._handle_get_config(ClientCommand("get", "get_config"))
    assert json.loads(reply.payload)["user_config"]["telemetry_heartbeat_interval_ms"] == (
        2**63 - 1
    )


def test_push_interval_beyond_int64_is_rejected_without_mutating_config():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    reply = manager._handle_command(
        ClientCommand(
            command_id="oversized-interval",
            command_type="push_config",
            payload=f'{{"heartbeat_interval_ms":{2**63}}}'.encode(),
        )
    )

    assert reply.success is False
    assert "fit in a signed 64-bit integer" in reply.error_message
    assert manager._heartbeat_interval() == telemetry_module._DEFAULT_HEARTBEAT_INTERVAL


def test_sync_heartbeat_loop_exits_before_work_when_owner_is_gone():
    manager = ClientTelemetryManager(
        lambda: None,
        TelemetryConfig(enabled=True),
        owner_alive_provider=lambda: False,
    )
    manager._stop_event.set()

    manager._heartbeat_loop()

    assert manager.get_metrics_snapshots() == []


def test_sync_stop_waits_for_worker_exit_before_clearing_thread_handle():
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()
    stop_started = threading.Event()
    stop_returned = threading.Event()

    class BlockingStub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            heartbeat_entered.set()
            assert release_heartbeat.wait(timeout=5)
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status())

    manager = ClientTelemetryManager(BlockingStub, TelemetryConfig(enabled=True))
    manager.start()
    assert heartbeat_entered.wait(timeout=1)
    worker = manager._thread
    assert worker is not None

    def stop_manager():
        stop_started.set()
        manager.stop()
        stop_returned.set()

    stopper = threading.Thread(target=stop_manager)
    stopper.start()

    assert stop_started.wait(timeout=1)
    assert not stop_returned.wait(timeout=0.05)
    assert manager._thread is worker
    assert worker.is_alive()

    release_heartbeat.set()
    assert stop_returned.wait(timeout=1)
    stopper.join(timeout=1)
    assert not stopper.is_alive()
    assert not worker.is_alive()
    assert manager._thread is None


def test_sync_stop_retains_live_worker_handle_when_join_times_out(monkeypatch):
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()

    class BlockingStub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            heartbeat_entered.set()
            assert release_heartbeat.wait(timeout=5)
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status())

    monkeypatch.setattr(telemetry_module, "_HEARTBEAT_STOP_JOIN_TIMEOUT", 0.01)
    manager = ClientTelemetryManager(BlockingStub, TelemetryConfig(enabled=True))
    manager.start()
    assert heartbeat_entered.wait(timeout=1)
    worker = manager._thread
    assert worker is not None

    manager.stop()

    assert worker.is_alive()
    assert manager._thread is worker

    release_heartbeat.set()
    worker.join(timeout=1)
    assert not worker.is_alive()
    manager.stop()
    assert manager._thread is None


def test_sync_heartbeat_loop_isolates_unexpected_iteration_failure():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    calls = 0

    def flaky_heartbeat():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient telemetry failure")
        manager._stop_event.set()

    manager._create_snapshot = lambda: None
    manager._next_heartbeat_delay = lambda: 0
    manager._send_heartbeat = flaky_heartbeat

    manager._heartbeat_loop()

    assert calls == 2
    assert str(manager.last_heartbeat_error()) == "transient telemetry failure"


def test_sync_owner_release_wakes_long_wait():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    manager.owner_released()

    assert manager._stop_event.is_set()


@pytest.mark.asyncio
async def test_async_long_heartbeat_interval_uses_cancellable_sleep_chunks(monkeypatch):
    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    sleeps = []

    async def record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(telemetry_module, "_MAX_WAIT_CHUNK_SECONDS", 2.0)
    monkeypatch.setattr(telemetry_module.asyncio, "sleep", record_sleep)

    await manager._sleep_until_next_heartbeat(5.5)

    assert sleeps == [2.0, 2.0, 1.5]


@pytest.mark.asyncio
async def test_async_heartbeat_loop_exits_before_work_when_owner_is_gone():
    manager = AsyncClientTelemetryManager(
        lambda: None,
        TelemetryConfig(enabled=True),
        owner_alive_provider=lambda: False,
    )
    manager._stop_event.set()

    await manager._async_heartbeat_loop()

    assert manager.get_metrics_snapshots() == []


@pytest.mark.asyncio
async def test_async_owner_release_cancels_long_sleep():
    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    manager._task = asyncio.create_task(asyncio.Event().wait())

    manager.owner_released()
    with pytest.raises(asyncio.CancelledError):
        await manager._task

    assert manager._stop_event.is_set()
    assert manager._task.cancelled()


def test_sync_heartbeat_success_unimplemented_and_lifecycle_short_circuits():
    captured = []

    class SuccessStub:
        @staticmethod
        def ClientHeartbeat(request, **_kwargs):
            captured.append(request)
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(),
                commands=[
                    common_pb2.ClientCommand(
                        command_id="server-config",
                        command_type="push_config",
                        payload=b'{"sampling_rate":0.5}',
                        create_time=7,
                        persistent=True,
                    )
                ],
            )

    manager = ClientTelemetryManager(SuccessStub, TelemetryConfig(enabled=True))
    manager._unsupported_streak = 2
    manager._queue_reply(CommandReply("pending", True))
    manager._send_heartbeat()

    assert captured[0].command_replies[0].command_id == "pending"
    assert manager._pending_replies[0].command_id == "server-config"
    assert manager._unsupported_streak == 0
    assert manager.last_heartbeat_error() is None
    assert manager.last_command_timestamp == 0

    class UnsupportedStub:
        @staticmethod
        def ClientHeartbeat(*_args, **_kwargs):
            raise _UnimplementedRpcError

    unsupported = ClientTelemetryManager(
        UnsupportedStub, TelemetryConfig(enabled=True, heartbeat_interval=0.5)
    )
    unsupported._send_heartbeat()
    assert unsupported._unsupported_streak == 1
    assert isinstance(unsupported.last_heartbeat_error(), _UnimplementedRpcError)
    assert unsupported._next_heartbeat_delay() == 1.0

    ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=False))._send_heartbeat()
    ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))._send_heartbeat()

    disabled = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=False))
    disabled.start()
    disabled.start()
    assert disabled.ready is True
    assert disabled._thread is None

    joined = []

    def record_join(timeout):
        joined.append(timeout)

    disabled._thread = SimpleNamespace(join=record_join, is_alive=lambda: False)
    disabled.stop()
    assert joined == [telemetry_module._HEARTBEAT_STOP_JOIN_TIMEOUT]


def test_sync_disabled_metrics_keeps_control_plane_and_can_be_reenabled():
    captured = []

    class Stub:
        @staticmethod
        def ClientHeartbeat(request, **_kwargs):
            captured.append(request)
            if len(captured) == 1:
                command = common_pb2.ClientCommand(
                    command_id="disable",
                    command_type="push_config",
                    payload=b'{"enabled":false}',
                    create_time=1,
                    persistent=True,
                )
            elif len(captured) == 2:
                command = common_pb2.ClientCommand(
                    command_id="enable",
                    command_type="push_config",
                    payload=b'{"enabled":true}',
                    create_time=2,
                    persistent=True,
                )
            else:
                return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status())
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(), commands=[command]
            )

    manager = ClientTelemetryManager(Stub, TelemetryConfig(enabled=True))
    manager.record_operation("Query", "books", time.perf_counter() - 0.001)
    manager._create_snapshot()

    manager._send_heartbeat()
    disable_hash = manager.config_hash
    with manager._config_lock:
        assert manager._config.enabled is False
    assert captured[0].metrics

    manager.record_operation("Query", "books", time.perf_counter() - 0.001)
    manager._create_snapshot()
    assert len(manager.get_metrics_snapshots()) == 1
    manager._send_heartbeat()

    assert not captured[1].metrics
    assert [reply.command_id for reply in captured[1].command_replies] == ["disable"]
    assert captured[1].config_hash == disable_hash
    assert [reply.command_id for reply in manager._pending_replies] == ["enable"]
    with manager._config_lock:
        assert manager._config.enabled is True

    manager._create_snapshot()
    enable_hash = manager.config_hash
    manager._send_heartbeat()
    assert [reply.command_id for reply in captured[2].command_replies] == ["enable"]
    assert captured[2].config_hash == enable_hash
    assert manager._pending_replies == []


@pytest.mark.asyncio
async def test_async_heartbeat_success_errors_and_lifecycle_short_circuits():
    class SuccessStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status())

    manager = AsyncClientTelemetryManager(SuccessStub, TelemetryConfig(enabled=True))
    manager._unsupported_streak = 2
    manager._queue_reply(CommandReply("pending", True))
    await manager._send_heartbeat_async()
    assert manager._unsupported_streak == 0
    assert manager.last_heartbeat_error() is None
    assert manager._pending_replies == []

    class StatusErrorStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(error_code=common_pb2.UnexpectedError, reason="bad")
            )

    status_error = AsyncClientTelemetryManager(StatusErrorStub, TelemetryConfig(enabled=True))
    status_error._unsupported_streak = 2
    await status_error._send_heartbeat_async()
    assert status_error._unsupported_streak == 0
    assert str(status_error.last_heartbeat_error()) == "bad"

    class UnsupportedStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            raise _UnimplementedRpcError

    unsupported = AsyncClientTelemetryManager(UnsupportedStub, TelemetryConfig(enabled=True))
    await unsupported._send_heartbeat_async()
    assert unsupported._unsupported_streak == 1
    assert isinstance(unsupported.last_heartbeat_error(), _UnimplementedRpcError)

    class StaleFailureStub:
        def __init__(self, owner):
            self.owner = owner

        async def ClientHeartbeat(self, *_args, **_kwargs):
            self.owner.rebind_stub(object())
            raise RuntimeError("stale")

    stale = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    stale.rebind_stub(StaleFailureStub(stale))
    await stale._send_heartbeat_async()
    assert stale.last_heartbeat_error() is None

    disabled = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=False))
    await disabled._send_heartbeat_async()
    enabled_without_stub = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    await enabled_without_stub._send_heartbeat_async()

    disabled.start()
    disabled.start()
    assert disabled.ready is True
    assert disabled._task is None

    blocker = asyncio.Event()
    task_manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    task_manager._task = asyncio.create_task(blocker.wait())
    await task_manager.stop_async()
    assert task_manager._task is None

    task_manager._task = asyncio.create_task(blocker.wait())
    task_manager.stop()
    await asyncio.sleep(0)
    assert task_manager._task.cancelled()


@pytest.mark.asyncio
async def test_async_heartbeat_loop_isolates_failures_but_propagates_cancellation():
    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    calls = 0

    async def flaky_heartbeat():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient telemetry failure")
        manager._stop_event.set()

    manager._create_snapshot = lambda: None
    manager._next_heartbeat_delay = lambda: 0
    manager._send_heartbeat_async = flaky_heartbeat
    await manager._async_heartbeat_loop()
    assert calls == 2
    assert str(manager.last_heartbeat_error()) == "transient telemetry failure"

    cancelled = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    entered = asyncio.Event()
    blocker = asyncio.Event()

    async def blocking_heartbeat():
        entered.set()
        await blocker.wait()

    cancelled._create_snapshot = lambda: None
    cancelled._send_heartbeat_async = blocking_heartbeat
    task = asyncio.create_task(cancelled._async_heartbeat_loop())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    send_entered = asyncio.Event()

    class BlockingStub:
        @staticmethod
        async def ClientHeartbeat(*_args, **_kwargs):
            send_entered.set()
            await asyncio.Event().wait()

    send_manager = AsyncClientTelemetryManager(BlockingStub, TelemetryConfig(enabled=True))
    send_task = asyncio.create_task(send_manager._send_heartbeat_async())
    await send_entered.wait()
    send_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await send_task


@pytest.mark.asyncio
async def test_async_stop_swallows_completed_telemetry_task_failure():
    manager = AsyncClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    async def fail():
        raise RuntimeError("background telemetry failed")

    manager._task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    await manager.stop_async()

    assert manager._task is None
    assert str(manager.last_heartbeat_error()) == "background telemetry failed"


@pytest.mark.asyncio
async def test_async_disabled_metrics_keeps_control_plane_and_can_be_reenabled():
    captured = []

    class Stub:
        @staticmethod
        async def ClientHeartbeat(request, **_kwargs):
            captured.append(request)
            if len(captured) == 1:
                command = common_pb2.ClientCommand(
                    command_id="disable",
                    command_type="push_config",
                    payload=b'{"enabled":false}',
                    create_time=1,
                    persistent=True,
                )
            elif len(captured) == 2:
                command = common_pb2.ClientCommand(
                    command_id="enable",
                    command_type="push_config",
                    payload=b'{"enabled":true}',
                    create_time=2,
                    persistent=True,
                )
            else:
                return milvus_pb2.ClientHeartbeatResponse(status=common_pb2.Status())
            return milvus_pb2.ClientHeartbeatResponse(
                status=common_pb2.Status(), commands=[command]
            )

    manager = AsyncClientTelemetryManager(Stub, TelemetryConfig(enabled=True))
    manager.record_operation("Query", "books", time.perf_counter() - 0.001)
    manager._create_snapshot()

    await manager._send_heartbeat_async()
    disable_hash = manager.config_hash
    with manager._config_lock:
        assert manager._config.enabled is False
    assert captured[0].metrics

    manager.record_operation("Query", "books", time.perf_counter() - 0.001)
    manager._create_snapshot()
    assert len(manager.get_metrics_snapshots()) == 1
    await manager._send_heartbeat_async()

    assert not captured[1].metrics
    assert [reply.command_id for reply in captured[1].command_replies] == ["disable"]
    assert captured[1].config_hash == disable_hash
    assert [reply.command_id for reply in manager._pending_replies] == ["enable"]
    with manager._config_lock:
        assert manager._config.enabled is True

    manager._create_snapshot()
    enable_hash = manager.config_hash
    await manager._send_heartbeat_async()
    assert [reply.command_id for reply in captured[2].command_replies] == ["enable"]
    assert captured[2].config_hash == enable_hash
    assert manager._pending_replies == []


def test_sync_interceptor_and_deferred_future_record_final_status():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    interceptor = TelemetryUnaryUnaryInterceptor(manager)
    request_id = new_client_request_id()

    class ImmediateCall:
        @staticmethod
        def exception():
            return None

        @staticmethod
        def result():
            return milvus_pb2.QueryResults(
                status=common_pb2.Status(
                    error_code=common_pb2.CollectionNotExists, reason="missing"
                )
            )

        def add_done_callback(self, callback):
            callback(self)

    details = SimpleNamespace(
        method="/milvus.proto.milvus.MilvusService/Query",
        metadata=(("client-request-id", request_id),),
    )
    result = interceptor.intercept_unary_unary(
        lambda *_args: ImmediateCall(), details, SimpleNamespace(collectionName="books")
    )
    assert isinstance(result, ImmediateCall)
    assert manager.get_recent_errors(1)[0].request_id == request_id

    future = Future()

    class Handler:
        _telemetry = manager

        @telemetry_operation("Search")
        def search(self, collection_name):
            return SimpleNamespace(_exception=None, _future=future)

    Handler().search("books")
    assert "Search" not in manager._collectors
    future.set_result(milvus_pb2.SearchResults(status=common_pb2.Status()))
    assert manager._collectors["Search"].global_bucket.success_count == 1


def test_sync_pymilvus_future_records_after_result_processing(monkeypatch):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    clock = {"now": 10.0}
    monkeypatch.setattr(telemetry_module.time, "perf_counter", lambda: clock["now"])

    class ParsingFuture(PyMilvusFuture):
        def on_response(self, _response):
            clock["now"] = 10.04
            raise ValueError("response parser failed")

    raw_future = Future()

    class Handler:
        _telemetry = manager

        @telemetry_operation("Search")
        def search(self, collection_name):
            return ParsingFuture(raw_future)

    result = Handler().search("books")
    raw_future.set_result("wire response")

    assert "Search" not in manager._collectors
    with pytest.raises(ValueError, match="response parser failed"):
        result.result()

    bucket = manager._collectors["Search"].global_bucket
    assert bucket.request_count == 1
    assert bucket.success_count == 0
    assert bucket.error_count == 1
    assert bucket.total_latency_us == pytest.approx(40_000, abs=1)
    assert manager.get_recent_errors(1)[0].error_msg == "response parser failed"


def test_sync_pymilvus_future_done_records_response_processing_error():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))

    class ParsingFuture(PyMilvusFuture):
        def on_response(self, _response):
            raise ValueError("done parser failed")

    raw_future = Future()

    class Handler:
        _telemetry = manager

        @telemetry_operation("Search")
        def search(self, collection_name):
            return ParsingFuture(raw_future)

    result = Handler().search("books")
    raw_future.set_result("wire response")

    assert "Search" not in manager._collectors
    result.done()

    bucket = manager._collectors["Search"].global_bucket
    assert bucket.request_count == 1
    assert bucket.error_count == 1
    assert str(result._exception) == "done parser failed"


@pytest.mark.parametrize("failure_stage", ["callback", "on_response"])
def test_sync_pymilvus_future_concurrent_consumers_record_first_failure_once(failure_stage):
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    processing_entered = threading.Event()
    release_processing = threading.Event()
    done_returned = threading.Event()
    parse_calls = 0
    callback_calls = 0

    class ParsingFuture(PyMilvusFuture):
        def on_response(self, _response):
            nonlocal parse_calls
            parse_calls += 1
            if failure_stage == "on_response":
                processing_entered.set()
                assert release_processing.wait(timeout=2)
                raise ValueError("on_response failed")
            return "parsed"

    def user_callback(_result):
        nonlocal callback_calls
        callback_calls += 1
        if failure_stage == "callback":
            processing_entered.set()
            assert release_processing.wait(timeout=2)
            raise ValueError("callback failed")

    raw_future = Future()
    raw_future.set_result("wire response")

    class Handler:
        _telemetry = manager

        @telemetry_operation("Search")
        def search(self, collection_name):
            return ParsingFuture(raw_future, user_callback)

    result = Handler().search("books")
    result_errors = []
    done_errors = []

    def consume_result():
        try:
            result.result()
        except BaseException as exc:
            result_errors.append(exc)

    def consume_done():
        try:
            result.done()
        except BaseException as exc:
            done_errors.append(exc)
        finally:
            done_returned.set()

    result_thread = threading.Thread(target=consume_result)
    result_thread.start()
    assert processing_entered.wait(timeout=2)

    done_thread = threading.Thread(target=consume_done)
    done_thread.start()
    assert not done_returned.wait(timeout=0.05)

    release_processing.set()
    result_thread.join(timeout=2)
    done_thread.join(timeout=2)

    assert not result_thread.is_alive()
    assert not done_thread.is_alive()
    assert done_errors == []
    assert len(result_errors) == 1
    assert str(result_errors[0]) == f"{failure_stage} failed"
    assert parse_calls == 1
    assert callback_calls == (1 if failure_stage == "callback" else 0)

    bucket = manager._collectors["Search"].global_bucket
    assert bucket.request_count == 1
    assert bucket.success_count == 0
    assert bucket.error_count == 1
    assert len(manager.get_recent_errors()) == 1
    assert manager.get_recent_errors()[0].error_msg == f"{failure_stage} failed"


@pytest.mark.asyncio
async def test_async_interceptor_callback_records_response_and_transport_error():
    manager = ClientTelemetryManager(lambda: None, TelemetryConfig(enabled=True))
    interceptor = AsyncTelemetryUnaryUnaryInterceptor(manager)

    class AsyncCall:
        def __init__(self, response=None, error=None):
            self.response = response
            self.error = error

        def __await__(self):
            async def complete():
                if self.error is not None:
                    raise self.error
                return self.response

            return complete().__await__()

        def add_done_callback(self, callback):
            callback(self)

    details = SimpleNamespace(method=b"/milvus.proto.milvus.MilvusService/Search", metadata=())

    async def continuation(_details, _request):
        return AsyncCall(milvus_pb2.SearchResults(status=common_pb2.Status()))

    call = await interceptor.intercept_unary_unary(
        continuation, details, SimpleNamespace(collection_name="books")
    )
    assert isinstance(call, AsyncCall)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    async def failing_continuation(_details, _request):
        return AsyncCall(error=RuntimeError("transport"))

    await interceptor.intercept_unary_unary(
        failing_continuation, details, SimpleNamespace(collection_name="books")
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    collector = manager._collectors["Search"]
    assert collector.global_bucket.request_count == 2
    assert collector.global_bucket.success_count == 1
    assert collector.global_bucket.error_count == 1
    assert _request_id_from_metadata(()) == ""
    assert _response_error(SimpleNamespace()) is None
