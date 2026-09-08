"""Client-side telemetry, heartbeat, and server command support."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import heapq
import inspect
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from array import array
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import grpc

from pymilvus.client import __version__
from pymilvus.client.call_context import is_valid_client_request_id
from pymilvus.grpc_gen import common_pb2, milvus_pb2

# Seconds between heartbeats, and therefore the metrics window: each heartbeat carries the
# operations since the last one. The coordinator answers a telemetry query from the window
# before the newest, so what a caller reads is between one and two intervals old.
_DEFAULT_HEARTBEAT_INTERVAL = 10.0
_HEARTBEAT_RPC_TIMEOUT = 10.0
# Give the synchronous RPC its full deadline plus a small scheduling margin. If
# custom handler code outlives that bound, retain the live thread handle so a later
# stop can finish joining it instead of losing ownership of the worker.
_HEARTBEAT_STOP_JOIN_TIMEOUT = _HEARTBEAT_RPC_TIMEOUT + 1.0
_MAX_UNSUPPORTED_BACKOFF = 30 * 60.0
_MAX_REPLY_PAYLOAD_SIZE = 1024 * 1024
_MAX_INT64 = 2**63 - 1
_MAX_HEARTBEAT_INTERVAL = _MAX_INT64 / 1000.0
# Keep timer conversions within a conservative signed 32-bit millisecond range.
# Longer protocol-valid intervals are represented by cancellable chunks.
_MAX_WAIT_CHUNK_SECONDS = (2**31 - 1) / 1000.0
# Fixed-point unit for accumulating a fractional sampling rate. A rate becomes an integer
# step of this many units, so the smallest rate that still samples is 1e-9 -- far below
# anything an operator would set, which is the point: a configured rate must never round
# down to "off".
_SAMPLING_SCALE = 1_000_000_000
_LATENCY_SAMPLE_SIZE = 1000
_HISTORY_LATENCY_SAMPLE_SIZE = 128
_SNAPSHOT_HISTORY_TTL_MS = 60 * 60 * 1000
_SNAPSHOT_HARD_LIMIT = 4096
_RFC3339_PATTERN = re.compile(
    r"^(?P<datetime>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d+)?(?P<timezone>Z|[+-]\d{2}:\d{2})$"
)

# A logical-operation wrapper stays active across validation, schema/cache work, retries,
# result parsing, and (for asyncio) awaits. The gRPC interceptor consults this depth and
# leaves recording to the outer wrapper, preventing one logical call from being counted
# once per transport attempt.
_LOGICAL_TELEMETRY_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "pymilvus_logical_telemetry_depth", default=0
)

_OPERATION_NAMES = {
    "Insert": "Insert",
    "Delete": "Delete",
    "Upsert": "Upsert",
    "Search": "Search",
    "HybridSearch": "HybridSearch",
    "Query": "Query",
    "RunAnalyzer": "RunAnalyzer",
}

_PUSH_CONFIG_KEYS = {
    "enabled",
    "heartbeat_interval_ms",
    "sampling_rate",
}
_UNSET = object()


@contextmanager
def suppress_telemetry():
    """Exclude an internal operation and its transport attempts from public metrics."""

    depth = _LOGICAL_TELEMETRY_DEPTH.get()
    token = _LOGICAL_TELEMETRY_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        _LOGICAL_TELEMETRY_DEPTH.reset(token)


def _reject_json_constant(value: str) -> None:
    msg = f"invalid JSON constant: {value}"
    raise ValueError(msg)


def _json_object(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {}
    value = json.loads(payload.decode(), parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        msg = "command payload must be a JSON object"
        raise TypeError(msg)
    return value


def _optional_bool(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise TypeError(msg)
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise TypeError(msg)
    if value < -(2**63) or value > 2**63 - 1:
        msg = f"{key} must fit in a signed 64-bit integer"
        raise ValueError(msg)
    return value


def _optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{key} must be a number"
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result):
        msg = f"{key} must be finite"
        raise ValueError(msg)
    return result


def _optional_string_list(payload: Mapping[str, Any], key: str) -> list[str] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        msg = f"{key} must be an array of strings"
        raise TypeError(msg)
    return list(value)


def _safe_exception_message(exc: BaseException) -> str:
    try:
        return str(exc)
    except BaseException:
        try:
            return f"{type(exc).__name__} (failed to format exception)"
        except BaseException:
            return "command handler raised an unprintable exception"


def _logical_call_info(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[str, str]:
    try:
        arguments = signature.bind_partial(*args, **kwargs).arguments
    except TypeError:
        arguments = {}
    extra = arguments.get("kwargs")
    if not isinstance(extra, Mapping):
        extra = {}
    collection = arguments.get("collection_name") or extra.get("collection_name") or ""
    if not collection:
        positional = arguments.get("args")
        if isinstance(positional, tuple) and positional:
            collection = positional[0]
    context = arguments.get("context")
    request_id = getattr(context, "_client_request_id", "") if context is not None else ""
    if not request_id:
        request_id = extra.get("client_request_id") or extra.get("client-request-id", "")
    return str(collection), request_id if is_valid_client_request_id(request_id) else ""


def _record_logical_operation(
    owner: Any,
    operation: str,
    collection: str,
    started_at: float,
    error: BaseException | None,
    request_id: str,
) -> None:
    manager = getattr(owner, "_telemetry", None)
    if manager is None:
        manager = getattr(getattr(owner, "_handler", None), "_telemetry", None)
    if manager is None:
        parent = getattr(owner, "_parent", None)
        manager = getattr(parent, "_telemetry", None)
        if manager is None:
            manager = getattr(getattr(parent, "_handler", None), "_telemetry", None)
    if manager is None:
        return
    try:
        manager.record_operation(operation, collection, started_at, error, request_id)
    except BaseException:
        # Telemetry is best-effort and must never replace the operation's own result.
        return


def _defer_sync_future_recording(
    result: Any,
    owner: Any,
    operation: str,
    collection: str,
    started_at: float,
    request_id: str,
) -> bool:
    pre_exception = getattr(result, "_exception", None)
    if pre_exception is not None:
        _record_logical_operation(
            owner, operation, collection, started_at, pre_exception, request_id
        )
        return True

    add_processed_callback = getattr(result, "_add_processed_callback", None)
    if callable(add_processed_callback):

        def processed(error: BaseException | None) -> None:
            _record_logical_operation(owner, operation, collection, started_at, error, request_id)

        registered = False
        with suppress(BaseException):
            add_processed_callback(processed)
            registered = True
        if registered:
            return True

    future = getattr(result, "_future", None)
    add_done_callback = getattr(future, "add_done_callback", None)
    if not callable(add_done_callback):
        return False

    def done(completed: Any) -> None:
        error: BaseException | None = None
        try:
            error = completed.exception()
            if error is None:
                error = _response_error(completed.result())
        except BaseException as exc:
            error = exc
        _record_logical_operation(owner, operation, collection, started_at, error, request_id)

    try:
        add_done_callback(done)
    except BaseException:
        return False
    return True


def telemetry_operation(operation: str) -> Callable[[Callable], Callable]:
    """Record one completed public operation around retries and result processing."""

    def decorate(func: Callable) -> Callable:
        signature = inspect.signature(func)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                started_at = time.perf_counter()
                collection, request_id = _logical_call_info(signature, args, kwargs)
                depth = _LOGICAL_TELEMETRY_DEPTH.get()
                token = _LOGICAL_TELEMETRY_DEPTH.set(depth + 1)
                error: BaseException | None = None
                try:
                    return await func(*args, **kwargs)
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    _LOGICAL_TELEMETRY_DEPTH.reset(token)
                    if depth == 0 and args:
                        _record_logical_operation(
                            args[0], operation, collection, started_at, error, request_id
                        )

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            collection, request_id = _logical_call_info(signature, args, kwargs)
            depth = _LOGICAL_TELEMETRY_DEPTH.get()
            token = _LOGICAL_TELEMETRY_DEPTH.set(depth + 1)
            error: BaseException | None = None
            deferred = False
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            else:
                if depth == 0 and args:
                    deferred = _defer_sync_future_recording(
                        result, args[0], operation, collection, started_at, request_id
                    )
                return result
            finally:
                _LOGICAL_TELEMETRY_DEPTH.reset(token)
                if depth == 0 and args and not deferred:
                    _record_logical_operation(
                        args[0], operation, collection, started_at, error, request_id
                    )

        return wrapper

    return decorate


@dataclass
class TelemetryConfig:
    """Configuration for client telemetry.

    ``client_id`` can be pinned to preserve identity across process restarts. When omitted,
    a random UUID is generated for this process.
    """

    enabled: bool = True
    heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL
    sampling_rate: float = 1.0
    error_max_count: int = 100
    client_id: str = ""

    @classmethod
    def from_value(cls, value: Any) -> TelemetryConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            # Managers apply server-pushed configuration in place. Give every
            # manager its own snapshot so a push cannot mutate the caller's
            # object or another connection that was initialized from it.
            return cls(
                enabled=value.enabled,
                heartbeat_interval=value.heartbeat_interval,
                sampling_rate=value.sampling_rate,
                error_max_count=value.error_max_count,
                client_id=value.client_id,
            )
        if isinstance(value, Mapping):
            aliases = {
                "Enabled": "enabled",
                "HeartbeatInterval": "heartbeat_interval",
                "SamplingRate": "sampling_rate",
                "ErrorMaxCount": "error_max_count",
                "ClientID": "client_id",
            }
            normalized = {aliases.get(key, key): item for key, item in value.items()}
            return cls(**normalized)
        msg = "telemetry_config must be a TelemetryConfig or mapping"
        raise TypeError(msg)

    def __post_init__(self) -> None:
        try:
            heartbeat_interval = float(self.heartbeat_interval)
        except (TypeError, ValueError, OverflowError) as exc:
            msg = "heartbeat_interval must be positive and fit in signed 64-bit milliseconds"
            raise ValueError(msg) from exc
        if (
            not math.isfinite(heartbeat_interval)
            or heartbeat_interval <= 0
            or heartbeat_interval > _MAX_HEARTBEAT_INTERVAL
        ):
            msg = "heartbeat_interval must be positive and fit in signed 64-bit milliseconds"
            raise ValueError(msg)
        self.heartbeat_interval = heartbeat_interval
        self.sampling_rate = min(1.0, max(0.0, float(self.sampling_rate)))
        if self.error_max_count <= 0:
            self.error_max_count = 100


@dataclass
class Metrics:
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0


@dataclass
class OperationMetrics:
    operation: str
    global_metrics: Metrics
    collection_metrics: dict[str, Metrics] = field(default_factory=dict)
    # History aggregation needs samples from the complete window, not an average of
    # per-window percentiles. Keep a compact, private approximation of the global
    # distribution; collection samples and this field never enter heartbeat/history JSON.
    _global_latency_samples_us: array = field(
        default_factory=lambda: array("q"), repr=False, compare=False
    )


@dataclass
class MetricsSnapshot:
    timestamp: int
    end_time: int
    metrics: list[OperationMetrics]


@dataclass
class ClientCommand:
    command_id: str
    command_type: str
    payload: bytes = b""
    create_time: int = 0
    persistent: bool = False
    target_scope: str = ""


@dataclass
class CommandReply:
    command_id: str
    success: bool
    error_message: str = ""
    payload: bytes = b""


@dataclass
class ErrorInfo:
    timestamp: int
    operation: str
    error_msg: str
    collection: str = ""
    request_id: str = ""


class _MetricsBucket:
    def __init__(self) -> None:
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency_us = 0
        self.max_latency_us = 0
        self.samples: deque[int] = deque(maxlen=_LATENCY_SAMPLE_SIZE)

    def record(self, latency_us: int, success: bool) -> None:
        self.request_count += 1
        self.success_count += int(success)
        self.error_count += int(not success)
        self.total_latency_us += latency_us
        self.max_latency_us = max(self.max_latency_us, latency_us)
        self.samples.append(latency_us)

    def snapshot_and_reset(
        self, *, retain_history_samples: bool = False
    ) -> tuple[Metrics | None, array]:
        if self.request_count == 0:
            return None, array("q")
        samples = sorted(self.samples)
        p99 = samples[min(len(samples) - 1, int(len(samples) * 0.99))] if samples else 0
        history_samples = array("q")
        if retain_history_samples:
            if len(samples) <= _HISTORY_LATENCY_SAMPLE_SIZE:
                history_samples = array("q", samples)
            else:
                # Evenly retain order statistics, including both endpoints. This keeps
                # history memory bounded while preserving the whole observed range.
                denominator = _HISTORY_LATENCY_SAMPLE_SIZE - 1
                last_index = len(samples) - 1
                history_samples = array(
                    "q",
                    (
                        samples[(index * last_index + denominator // 2) // denominator]
                        for index in range(_HISTORY_LATENCY_SAMPLE_SIZE)
                    ),
                )
        result = Metrics(
            request_count=self.request_count,
            success_count=self.success_count,
            error_count=self.error_count,
            avg_latency_ms=self.total_latency_us / self.request_count / 1000.0,
            p99_latency_ms=p99 / 1000.0,
            max_latency_ms=self.max_latency_us / 1000.0,
        )
        self.__init__()
        return result, history_samples


class _OperationCollector:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.global_bucket = _MetricsBucket()
        self.collections: dict[str, _MetricsBucket] = {}

    def record(self, collection: str, latency_us: int, success: bool) -> None:
        with self.lock:
            self.global_bucket.record(latency_us, success)
            if collection:
                self.collections.setdefault(collection, _MetricsBucket()).record(
                    latency_us, success
                )

    def snapshot_and_reset(self, enabled_collections: set[str] | None) -> OperationMetrics | None:
        with self.lock:
            global_metrics, global_samples = self.global_bucket.snapshot_and_reset(
                retain_history_samples=True
            )
            if global_metrics is None:
                return None
            collection_metrics: dict[str, Metrics] = {}
            for name, bucket in self.collections.items():
                metrics, _ = bucket.snapshot_and_reset()
                if metrics is not None and (
                    enabled_collections is None or name in enabled_collections
                ):
                    collection_metrics[name] = metrics
            self.collections = {}
            return OperationMetrics("", global_metrics, collection_metrics, global_samples)


class ClientTelemetryManager:
    """Collects metrics and exchanges commands over ``ClientHeartbeat``."""

    def __init__(
        self,
        stub_provider: Callable[[], Any],
        config: Any = None,
        *,
        user: str = "",
        database_provider: Callable[[], str] | None = None,
        config_provider: Callable[[], Mapping[str, Any]] | None = None,
        owner_alive_provider: Callable[[], bool] | None = None,
        runtime_client_id: str = "",
    ) -> None:
        self._stub_provider = stub_provider
        # Rebinding a handler to another endpoint must fence the old endpoint's in-flight
        # heartbeat.  Keep the stub and its generation under one lock so an old response
        # cannot clear replies/backoff or execute commands after the new endpoint commits.
        self._endpoint_lock = threading.RLock()
        self._bound_stub: Any = None
        self._endpoint_bound = False
        self._endpoint_generation = 0
        self._bound_database = ""
        self._database_bound = False
        self._transport_binding_token: Any = None
        self._config = TelemetryConfig.from_value(config)
        self._config_lock = threading.RLock()
        self._user = user or ""
        self._database_provider = database_provider or (lambda: "")
        self._config_provider = config_provider or (dict)
        self._owner_alive_provider = owner_alive_provider or (lambda: True)
        self._client_id = self._config.client_id or runtime_client_id or str(uuid.uuid4())
        self._client_id_stable = bool(self._config.client_id)

        self._collectors: dict[str, _OperationCollector] = {}
        self._collectors_lock = threading.Lock()
        self._enabled_collections: set[str] = set()
        self._all_collections_enabled = False
        self._collections_lock = threading.RLock()
        self._errors: deque[ErrorInfo] = deque(maxlen=self._config.error_max_count)
        self._errors_lock = threading.Lock()
        self._snapshots: deque[MetricsSnapshot] = deque()
        self._snapshots_lock = threading.Lock()

        self._handlers: dict[str, Callable[[ClientCommand], CommandReply]] = {}
        self._handlers_lock = threading.RLock()
        self._command_batch_lock = threading.RLock()
        self._pending_replies: list[common_pb2.CommandReply] = []
        self._pending_lock = threading.Lock()
        self._executed_commands: dict[str, int] = {}
        self._executed_lock = threading.Lock()
        self._last_command_timestamp = 0
        self._config_hash = ""
        # Carries the fractional sampling rate between calls, in _SAMPLING_SCALE units:
        # each operation adds the rate and the one that pushes it past a whole unit is the
        # one sampled. See _should_sample.
        self._sampling_accum = 0
        self._state_lock = threading.RLock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = False
        self._unsupported_streak = 0
        self._last_heartbeat_error: BaseException | None = None
        self._last_snapshot_end = 0
        self._register_default_handlers()

    @property
    def client_id(self) -> str:
        return self._client_id

    @property
    def config_hash(self) -> str:
        with self._state_lock:
            return self._config_hash

    @property
    def last_command_timestamp(self) -> int:
        with self._state_lock:
            return self._last_command_timestamp

    @property
    def ready(self) -> bool:
        return self._ready

    def is_supported(self) -> bool:
        with self._state_lock:
            return self._unsupported_streak == 0

    def last_heartbeat_error(self) -> BaseException | None:
        with self._state_lock:
            return self._last_heartbeat_error

    def start(self) -> None:
        if self._thread is not None or self._ready:
            return
        self._ready = True
        if not self._enabled():
            return
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"pymilvus-telemetry-{self._client_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        # Wait through the heartbeat RPC deadline. A custom handler can run longer;
        # in that case retain the handle so a later stop can finish joining it.
        thread.join(timeout=_HEARTBEAT_STOP_JOIN_TIMEOUT)
        if not thread.is_alive() and self._thread is thread:
            self._thread = None

    def register_command_handler(
        self, command_type: str, handler: Callable[[ClientCommand], CommandReply]
    ) -> None:
        with self._handlers_lock:
            self._handlers[command_type] = handler

    def rebind_stub(self, stub: Any, *, database: Any = _UNSET) -> None:
        """Atomically bind heartbeat transport/identity and fence the old endpoint."""

        with self._endpoint_lock:
            self._transport_binding_token = None
            self._bound_stub = stub
            self._endpoint_bound = True
            if database is not _UNSET:
                self._bound_database = str(database or "")
                self._database_bound = True
            self._endpoint_generation += 1

    def bind_transport(self, stub: Any, database: str, binding_token: Any) -> None:
        """Bind this logical manager to a pooled handler using an opaque lease token."""

        with self._endpoint_lock:
            self._transport_binding_token = binding_token
            self._bound_stub = stub
            self._endpoint_bound = True
            self._bound_database = database or ""
            self._database_bound = True
            self._endpoint_generation += 1

    def rebind_transport(self, stub: Any, binding_token: Any) -> bool:
        """Rebind a pooled transport only while its lease is still current."""

        with self._endpoint_lock:
            if binding_token is not self._transport_binding_token:
                return False
            self._bound_stub = stub
            self._endpoint_bound = True
            self._endpoint_generation += 1
            return True

    def unbind_transport(self, binding_token: Any) -> bool:
        """Fence a detached transport without clearing a newer handler binding."""

        with self._endpoint_lock:
            if binding_token is not self._transport_binding_token:
                return False
            self._transport_binding_token = None
            self._bound_stub = None
            self._endpoint_bound = True
            self._endpoint_generation += 1
            return True

    def _heartbeat_endpoint(self) -> tuple[Any, int, str]:
        with self._endpoint_lock:
            stub = self._bound_stub if self._endpoint_bound else self._stub_provider()
            database = (
                self._bound_database if self._database_bound else self._database_provider() or ""
            )
            return stub, self._endpoint_generation, database

    def record_operation(
        self,
        operation: str,
        collection: str,
        started_at: float,
        error: BaseException | None = None,
        request_id: str = "",
    ) -> None:
        with self._config_lock:
            enabled = self._config.enabled
            sampling_rate = self._config.sampling_rate
        if not enabled or not self._should_sample(sampling_rate):
            return

        latency_us = max(0, int((time.perf_counter() - started_at) * 1_000_000))
        with self._collections_lock:
            collection_enabled = (
                self._all_collections_enabled or collection in self._enabled_collections
            )
        collection_key = collection if collection_enabled else ""

        with self._collectors_lock:
            collector = self._collectors.setdefault(operation, _OperationCollector())
        collector.record(collection_key, latency_us, error is None)

        if error is not None:
            with self._errors_lock:
                self._errors.append(
                    ErrorInfo(
                        timestamp=int(time.time() * 1000),
                        operation=operation,
                        error_msg=str(error),
                        collection=collection,
                        request_id=request_id,
                    )
                )

    def get_recent_errors(self, max_count: int = 100) -> list[ErrorInfo]:
        with self._errors_lock:
            return list(reversed(self._errors))[:max_count]

    def get_metrics_snapshots(self) -> list[MetricsSnapshot]:
        with self._snapshots_lock:
            self._prune_snapshots_locked(int(time.time() * 1000))
            return list(self._snapshots)

    def process_commands(
        self, commands: Iterable[Any], *, expected_generation: int | None = None
    ) -> None:
        commands = list(commands)
        with self._command_batch_lock:
            self._process_commands_locked(commands, expected_generation)

    def _generation_matches(self, expected_generation: int | None) -> bool:
        if expected_generation is None:
            return True
        with self._endpoint_lock:
            return expected_generation == self._endpoint_generation

    def _process_commands_locked(
        self, commands: list[Any], expected_generation: int | None
    ) -> None:
        if not self._generation_matches(expected_generation):
            return
        with self._state_lock:
            last_timestamp = self._last_command_timestamp
        max_timestamp = last_timestamp
        has_persistent = False

        for command in commands:
            if not self._generation_matches(expected_generation):
                return
            local = ClientCommand(
                command_id=command.command_id,
                command_type=command.command_type,
                payload=bytes(command.payload),
                create_time=command.create_time,
                persistent=command.persistent,
                target_scope=command.target_scope,
            )
            has_persistent = has_persistent or local.persistent
            if not local.persistent:
                max_timestamp = max(max_timestamp, local.create_time)
                if local.create_time < last_timestamp:
                    self._queue_reply(CommandReply(local.command_id, True))
                    if not self._generation_matches(expected_generation):
                        return
                    continue
                with self._executed_lock:
                    already_executed = local.command_id in self._executed_commands
                if already_executed:
                    self._queue_reply(CommandReply(local.command_id, True))
                    if not self._generation_matches(expected_generation):
                        return
                    continue

            reply = self._handle_command(local)
            if not local.persistent:
                with self._executed_lock:
                    self._executed_commands[local.command_id] = local.create_time
            if reply is not None:
                self._queue_reply(reply)

            # Preserve the completed command's dedup state and ACK, but stop the old
            # endpoint batch before applying any remaining commands or its cursor/hash.
            if not self._generation_matches(expected_generation):
                return

        def commit_batch() -> None:
            with self._executed_lock:
                # Timestamp filtering only rejects commands older than the cursor.
                # Keep IDs at the new cursor timestamp so equal-timestamp
                # redeliveries remain idempotent on every retry.
                self._executed_commands = {
                    command_id: timestamp
                    for command_id, timestamp in self._executed_commands.items()
                    if timestamp >= max_timestamp
                }
            with self._state_lock:
                # An empty persistent subset is not an authoritative empty snapshot:
                # the current server also omits configs when our non-empty hash already
                # matches. Keep the last accepted hash until the response protocol can
                # explicitly distinguish those states, otherwise it oscillates every
                # heartbeat and repeatedly reapplies the same persistent configs.
                if has_persistent:
                    self._config_hash = self.calculate_config_hash(commands)
                self._last_command_timestamp = max(self._last_command_timestamp, max_timestamp)

        if expected_generation is None:
            commit_batch()
            return
        # Commit only while the response still belongs to the current endpoint. No user
        # code runs in this short critical section.
        with self._endpoint_lock:
            if expected_generation != self._endpoint_generation:
                return
            commit_batch()

    @staticmethod
    def calculate_config_hash(commands: Iterable[Any]) -> str:
        persistent = sorted(
            (command for command in commands if command.persistent),
            key=lambda command: command.command_id,
        )
        if not persistent:
            return ""
        digest = hashlib.sha256()
        for command in persistent:
            digest.update(command.command_id.encode())
            digest.update(command.command_type.encode())
            digest.update(bytes(command.payload))
        return digest.hexdigest()[:16]

    def _enabled(self) -> bool:
        with self._config_lock:
            return self._config.enabled

    def _heartbeat_interval(self) -> float:
        with self._config_lock:
            interval = self._config.heartbeat_interval
        if not math.isfinite(interval) or interval <= 0 or interval > _MAX_HEARTBEAT_INTERVAL:
            return _DEFAULT_HEARTBEAT_INTERVAL
        return interval

    def _heartbeat_interval_ms(self) -> int:
        # The max int64 millisecond value rounds up when represented as float seconds.
        return min(_MAX_INT64, int(self._heartbeat_interval() * 1000))

    def _wait_for_stop(self, delay: float) -> bool:
        while delay > _MAX_WAIT_CHUNK_SECONDS:
            if self._stop_event.wait(_MAX_WAIT_CHUNK_SECONDS):
                return True
            delay -= _MAX_WAIT_CHUNK_SECONDS
        return self._stop_event.wait(max(0.0, delay))

    def _next_heartbeat_delay(self) -> float:
        interval = self._heartbeat_interval()
        with self._state_lock:
            unsupported_streak = self._unsupported_streak
        if unsupported_streak <= 0:
            return interval
        if interval >= _MAX_UNSUPPORTED_BACKOFF:
            return interval
        delay = interval
        for _ in range(unsupported_streak):
            if delay >= _MAX_UNSUPPORTED_BACKOFF / 2:
                return _MAX_UNSUPPORTED_BACKOFF
            delay *= 2
        return delay

    def _owner_alive(self) -> bool:
        try:
            return bool(self._owner_alive_provider())
        except BaseException:
            return False

    def owner_released(self) -> None:
        """Wake the worker promptly when the logical client is garbage-collected."""

        self._stop_event.set()

    def _heartbeat_loop(self) -> None:
        while self._owner_alive() and not self._stop_event.is_set():
            try:
                self._create_snapshot()
                self._send_heartbeat()
            except BaseException as exc:
                with self._state_lock:
                    self._last_heartbeat_error = exc
            if self._wait_for_stop(self._next_heartbeat_delay()):
                return

    def _send_heartbeat(self) -> None:
        stub, endpoint_generation, database = self._heartbeat_endpoint()
        if stub is None:
            return

        metrics_enabled = self._enabled()
        with self._snapshots_lock:
            latest = self._snapshots[-1] if metrics_enabled and self._snapshots else None
        metrics = self._to_proto_metrics(latest.metrics if latest else [])
        with self._pending_lock:
            replies = list(self._pending_replies)
        with self._state_lock:
            config_hash = self._config_hash
            last_timestamp = self._last_command_timestamp

        request = milvus_pb2.ClientHeartbeatRequest(
            client_info=self._build_client_info(database),
            report_timestamp=int(time.time() * 1000),
            metrics=metrics,
            command_replies=replies,
            config_hash=config_hash,
            last_command_timestamp=last_timestamp,
        )
        try:
            response = stub.ClientHeartbeat(
                request, timeout=_HEARTBEAT_RPC_TIMEOUT, wait_for_ready=False
            )
        except grpc.RpcError as exc:
            with self._endpoint_lock:
                if endpoint_generation != self._endpoint_generation:
                    return
                with self._state_lock:
                    self._last_heartbeat_error = exc
                    if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                        self._unsupported_streak += 1
            return
        except BaseException as exc:  # best-effort background channel
            with self._endpoint_lock:
                if endpoint_generation != self._endpoint_generation:
                    return
                with self._state_lock:
                    self._last_heartbeat_error = exc
            return

        with self._endpoint_lock:
            if endpoint_generation != self._endpoint_generation:
                return
            # Any server response proves the RPC exists, even when the application status
            # is an error. Unsupported backoff is only for transport-level UNIMPLEMENTED.
            with self._state_lock:
                self._unsupported_streak = 0
                if response.status.code != 0 or response.status.error_code != 0:
                    self._last_heartbeat_error = RuntimeError(
                        response.status.reason or "client telemetry heartbeat failed"
                    )
                    return
                self._last_heartbeat_error = None
            with self._pending_lock:
                del self._pending_replies[: len(replies)]
        # Custom handlers may reconnect. Run them without endpoint_lock and fence every
        # command plus the final cursor/hash commit against this response's generation.
        self.process_commands(response.commands, expected_generation=endpoint_generation)

    def _build_client_info(self, database: Any = _UNSET) -> common_pb2.ClientInfo:
        reserved = {
            "client_id": self._client_id,
            "client_id_stable": str(self._client_id_stable).lower(),
        }
        if database is _UNSET:
            with self._endpoint_lock:
                database = (
                    self._bound_database
                    if self._database_bound
                    else self._database_provider() or ""
                )
        if database:
            reserved["db_name"] = str(database)
        return common_pb2.ClientInfo(
            sdk_type="Python",
            sdk_version=__version__,
            local_time=time.ctime(),
            user=self._user,
            host=socket.gethostname(),
            reserved=reserved,
        )

    def _should_sample(self, rate: float) -> bool:
        """Decide whether this operation is recorded, spreading the sampled ones evenly.

        Each call adds the rate to an accumulator and samples on the call that carries it
        across a whole unit: at 0.25 that is every fourth operation. What matters is that
        the ratio holds over any stretch of calls, not only over a long one -- metrics are
        reported per heartbeat window, and a window is tens or hundreds of operations. A
        scheme that sampled a contiguous run and then dropped one would give the right
        long-run ratio while making every individual window either complete or empty.
        """
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        # A rate too small to represent still means "sample rarely", never "sample never":
        # silently disabling telemetry for a positive rate is the one outcome nobody could
        # have intended.
        step = max(1, int(rate * _SAMPLING_SCALE))
        with self._state_lock:
            before = self._sampling_accum
            self._sampling_accum = before + step
            return self._sampling_accum // _SAMPLING_SCALE != before // _SAMPLING_SCALE

    def _create_snapshot(self) -> None:
        if not self._enabled():
            return
        with self._collections_lock:
            enabled_collections = (
                None if self._all_collections_enabled else set(self._enabled_collections)
            )
        metrics: list[OperationMetrics] = []
        with self._collectors_lock:
            collectors = list(self._collectors.items())
        for operation, collector in collectors:
            item = collector.snapshot_and_reset(enabled_collections)
            if item is not None:
                item.operation = operation
                metrics.append(item)

        now = int(time.time() * 1000)
        start = self._last_snapshot_end
        if start == 0 or start > now:
            start = now - self._heartbeat_interval_ms()
        self._last_snapshot_end = now
        with self._snapshots_lock:
            self._snapshots.append(MetricsSnapshot(start, now, metrics))
            self._prune_snapshots_locked(now)

    def _prune_snapshots_locked(self, now_ms: int) -> None:
        cutoff = now_ms - _SNAPSHOT_HISTORY_TTL_MS
        while self._snapshots and self._snapshots[0].end_time < cutoff:
            self._snapshots.popleft()
        while len(self._snapshots) > _SNAPSHOT_HARD_LIMIT:
            self._snapshots.popleft()

    def _to_proto_metrics(
        self, items: Iterable[OperationMetrics]
    ) -> list[common_pb2.OperationMetrics]:
        with self._collections_lock:
            all_collections_enabled = self._all_collections_enabled
            enabled_collections = set(self._enabled_collections)
        result = []
        for item in items:
            result.append(
                common_pb2.OperationMetrics(
                    **{
                        "operation": item.operation,
                        "global": ClientTelemetryManager._metrics_proto(item.global_metrics),
                        "collection_metrics": {
                            name: ClientTelemetryManager._metrics_proto(metrics)
                            for name, metrics in item.collection_metrics.items()
                            if all_collections_enabled or name in enabled_collections
                        },
                    }
                )
            )
        return result

    @staticmethod
    def _metrics_proto(metrics: Metrics) -> common_pb2.Metrics:
        return common_pb2.Metrics(
            request_count=metrics.request_count,
            success_count=metrics.success_count,
            error_count=metrics.error_count,
            avg_latency_ms=metrics.avg_latency_ms,
            p99_latency_ms=metrics.p99_latency_ms,
            max_latency_ms=metrics.max_latency_ms,
        )

    def _queue_reply(self, reply: CommandReply) -> None:
        with self._pending_lock:
            self._pending_replies.append(
                common_pb2.CommandReply(
                    command_id=reply.command_id,
                    success=reply.success,
                    error_message=reply.error_message,
                    payload=reply.payload,
                )
            )

    def _handle_command(self, command: ClientCommand) -> CommandReply:
        with self._handlers_lock:
            handler = self._handlers.get(command.command_type)
        if handler is None:
            return CommandReply(
                command.command_id,
                False,
                error_message=f"unknown command type: {command.command_type}",
            )
        try:
            reply = handler(command)
            if reply is None:
                return CommandReply(
                    command.command_id,
                    False,
                    error_message="command handler returned no reply",
                )
            return CommandReply(
                command.command_id,
                reply.success,
                error_message=reply.error_message,
                payload=reply.payload,
            )
        except BaseException as exc:
            return CommandReply(
                command.command_id,
                False,
                error_message=_safe_exception_message(exc),
            )

    def _register_default_handlers(self) -> None:
        self.register_command_handler("push_config", self._handle_push_config)
        self.register_command_handler("collection_metrics", self._handle_collection_metrics)
        self.register_command_handler("show_errors", self._handle_show_errors)
        self.register_command_handler("show_latency_history", self._handle_latency_history)
        self.register_command_handler("get_config", self._handle_get_config)

    @staticmethod
    def _payload(command: ClientCommand) -> dict[str, Any]:
        return _json_object(command.payload)

    def _handle_push_config(self, command: ClientCommand) -> CommandReply:
        payload = self._payload(command)
        # Validate every applied field before mutating the current config. This makes the
        # command atomic when, for example, a valid enabled flag accompanies an invalid
        # interval. Unknown fields (such as the server-side push API's ttl_seconds, which
        # never reaches clients by design) are reported as ignored, never validated.
        enabled = _optional_bool(payload, "enabled")
        interval_ms = _optional_int(payload, "heartbeat_interval_ms")
        sampling_rate = _optional_number(payload, "sampling_rate")
        if interval_ms is not None and interval_ms <= 0:
            msg = "heartbeat_interval_ms must be positive"
            raise ValueError(msg)

        applied: list[str] = []
        if enabled is not None:
            applied.append("enabled")
        if interval_ms is not None:
            applied.append("heartbeat_interval_ms")
        if sampling_rate is not None:
            applied.append("sampling_rate")
        ignored = sorted(key for key in payload if key not in _PUSH_CONFIG_KEYS)

        with self._config_lock:
            if enabled is not None:
                self._config.enabled = enabled
            if interval_ms is not None:
                self._config.heartbeat_interval = interval_ms / 1000.0
            if sampling_rate is not None:
                self._config.sampling_rate = min(1.0, max(0.0, sampling_rate))
        reply_payload: dict[str, Any] = {"applied": applied}
        if ignored:
            reply_payload["ignored"] = ignored
        return CommandReply(
            command.command_id,
            True,
            payload=json.dumps(reply_payload, separators=(",", ":")).encode(),
        )

    def _handle_collection_metrics(self, command: ClientCommand) -> CommandReply:
        if not command.payload:
            with self._collections_lock:
                payload = {
                    "enabled_collections": sorted(self._enabled_collections),
                    "all_collections_enabled": self._all_collections_enabled,
                }
            return CommandReply(command.command_id, True, payload=json.dumps(payload).encode())

        payload = self._payload(command)
        collections = _optional_string_list(payload, "collections") or []
        enabled = _optional_bool(payload, "enabled") or False
        _optional_string_list(payload, "metrics_types")
        wildcard = "*" in collections
        with self._collections_lock:
            if enabled:
                if not collections:
                    msg = "collections list cannot be empty when enabled=true"
                    raise ValueError(msg)
                if wildcard:
                    self._all_collections_enabled = True
                else:
                    self._enabled_collections.update(collections)
            elif wildcard or not collections:
                self._all_collections_enabled = False
                self._enabled_collections.clear()
            else:
                self._enabled_collections.difference_update(collections)
        return CommandReply(command.command_id, True)

    def _handle_show_errors(self, command: ClientCommand) -> CommandReply:
        payload = self._payload(command)
        max_count = _optional_int(payload, "max_count")
        if max_count is None or max_count <= 0:
            max_count = 100
        errors = [dict(vars(item)) for item in self.get_recent_errors(max_count)]
        if not errors:
            return CommandReply(command.command_id, True)
        encoded = json.dumps(errors, separators=(",", ":")).encode()
        while len(encoded) > _MAX_REPLY_PAYLOAD_SIZE and len(errors) > 1:
            errors = errors[: max(1, len(errors) // 2)]
            encoded = json.dumps(errors, separators=(",", ":")).encode()
        if len(encoded) > _MAX_REPLY_PAYLOAD_SIZE and errors:
            original_strings = {
                key: value for key, value in errors[0].items() if isinstance(value, str)
            }
            limits = {key: len(value) for key, value in original_strings.items()}
            while len(encoded) > _MAX_REPLY_PAYLOAD_SIZE and any(limits.values()):
                key = max(limits, key=limits.get)
                limit = limits[key]
                limits[key] = limit // 2 if limit > 1 else 0
                errors[0][key] = original_strings[key][: limits[key]] + "...(truncated)"
                encoded = json.dumps(errors, separators=(",", ":")).encode()
        if len(encoded) > _MAX_REPLY_PAYLOAD_SIZE:
            msg = "show_errors response exceeds the 1MB payload limit"
            raise ValueError(msg)
        return CommandReply(command.command_id, True, payload=encoded)

    def _handle_get_config(self, command: ClientCommand) -> CommandReply:
        user_config = dict(self._config_provider())
        for key in ("password", "token", "api_key"):
            user_config.pop(key, None)
        with self._config_lock:
            user_config.update(
                telemetry_enabled=self._config.enabled,
                telemetry_heartbeat_interval_ms=self._heartbeat_interval_ms(),
                telemetry_sampling_rate=self._config.sampling_rate,
            )
        with self._collections_lock:
            user_config["enabled_collections"] = (
                ["*"] if self._all_collections_enabled else sorted(self._enabled_collections)
            )
            user_config["all_collections_enabled"] = self._all_collections_enabled
        return CommandReply(
            command.command_id,
            True,
            payload=json.dumps({"user_config": user_config}, default=str).encode(),
        )

    def _handle_latency_history(self, command: ClientCommand) -> CommandReply:
        payload = self._payload(command)
        start_time = payload.get("start_time")
        end_time = payload.get("end_time")
        if start_time is not None and not isinstance(start_time, str):
            msg = "start_time must be a string"
            raise TypeError(msg)
        if end_time is not None and not isinstance(end_time, str):
            msg = "end_time must be a string"
            raise TypeError(msg)
        detail = _optional_bool(payload, "detail") or False
        start_ms = _parse_rfc3339_ms(start_time)
        end_ms = _parse_rfc3339_ms(end_time)
        if end_ms < start_ms:
            msg = "end_time must be after start_time"
            raise ValueError(msg)
        if end_ms - start_ms > 60 * 60 * 1000:
            msg = "time range cannot exceed 1 hour"
            raise ValueError(msg)
        snapshots = [
            item
            for item in self.get_metrics_snapshots()
            if item.end_time >= start_ms and item.timestamp <= end_ms
        ]
        if detail:
            body = {
                "snapshots": [_snapshot_dict(item) for item in snapshots],
                "total_snapshots": len(snapshots),
            }
        else:
            body = _aggregate_snapshots(snapshots, start_ms, end_ms)
        encoded = json.dumps(body, separators=(",", ":")).encode()
        if len(encoded) > _MAX_REPLY_PAYLOAD_SIZE:
            msg = "response too large, try a smaller time range"
            raise ValueError(msg)
        return CommandReply(command.command_id, True, payload=encoded)


class AsyncClientTelemetryManager(ClientTelemetryManager):
    """Asyncio heartbeat variant used by ``AsyncGrpcHandler``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is not None or self._ready:
            return
        self._ready = True
        if not self._enabled():
            return
        self._task = asyncio.get_running_loop().create_task(self._async_heartbeat_loop())

    async def stop_async(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                with self._state_lock:
                    self._last_heartbeat_error = exc
        self._task = None

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    def owner_released(self) -> None:
        super().owner_released()
        task = self._task
        if task is None:
            return
        try:
            loop = task.get_loop()
            if not loop.is_closed():
                loop.call_soon_threadsafe(task.cancel)
        except BaseException:
            # Owner cleanup can run during event-loop or interpreter teardown.
            return

    async def _async_heartbeat_loop(self) -> None:
        while self._owner_alive():
            try:
                self._create_snapshot()
                await self._send_heartbeat_async()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                with self._state_lock:
                    self._last_heartbeat_error = exc
            if self._stop_event.is_set():
                return
            await self._sleep_until_next_heartbeat(self._next_heartbeat_delay())

    async def _sleep_until_next_heartbeat(self, delay: float) -> None:
        while delay > 0:
            chunk = min(delay, _MAX_WAIT_CHUNK_SECONDS)
            await asyncio.sleep(chunk)
            delay -= chunk

    async def _send_heartbeat_async(self) -> None:
        stub, endpoint_generation, database = self._heartbeat_endpoint()
        if stub is None:
            return
        metrics_enabled = self._enabled()
        with self._snapshots_lock:
            latest = self._snapshots[-1] if metrics_enabled and self._snapshots else None
        with self._pending_lock:
            replies = list(self._pending_replies)
        with self._state_lock:
            config_hash = self._config_hash
            last_timestamp = self._last_command_timestamp
        request = milvus_pb2.ClientHeartbeatRequest(
            client_info=self._build_client_info(database),
            report_timestamp=int(time.time() * 1000),
            metrics=self._to_proto_metrics(latest.metrics if latest else []),
            command_replies=replies,
            config_hash=config_hash,
            last_command_timestamp=last_timestamp,
        )
        try:
            response = await stub.ClientHeartbeat(
                request, timeout=_HEARTBEAT_RPC_TIMEOUT, wait_for_ready=False
            )
        except asyncio.CancelledError:
            raise
        except grpc.RpcError as exc:
            with self._endpoint_lock:
                if endpoint_generation != self._endpoint_generation:
                    return
                with self._state_lock:
                    self._last_heartbeat_error = exc
                    if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                        self._unsupported_streak += 1
            return
        except BaseException as exc:
            with self._endpoint_lock:
                if endpoint_generation != self._endpoint_generation:
                    return
                with self._state_lock:
                    self._last_heartbeat_error = exc
            return
        with self._endpoint_lock:
            if endpoint_generation != self._endpoint_generation:
                return
            with self._state_lock:
                self._unsupported_streak = 0
                if response.status.code != 0 or response.status.error_code != 0:
                    self._last_heartbeat_error = RuntimeError(
                        response.status.reason or "client telemetry heartbeat failed"
                    )
                    return
                self._last_heartbeat_error = None
            with self._pending_lock:
                del self._pending_replies[: len(replies)]
        self.process_commands(response.commands, expected_generation=endpoint_generation)


class TelemetryUnaryUnaryInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, manager: ClientTelemetryManager) -> None:
        self._manager = manager

    def intercept_unary_unary(self, continuation: Callable, call_details: Any, request: Any):
        operation = _operation_from_method(call_details.method)
        if operation is None or _LOGICAL_TELEMETRY_DEPTH.get() > 0:
            return continuation(call_details, request)
        started_at = time.perf_counter()
        call = continuation(call_details, request)

        def done(completed: Any) -> None:
            error = completed.exception()
            if error is None:
                try:
                    error = _response_error(completed.result())
                except BaseException as exc:
                    error = exc
            self._manager.record_operation(
                operation,
                _collection_from_request(request),
                started_at,
                error,
                _request_id_from_metadata(call_details.metadata),
            )

        call.add_done_callback(done)
        return call


class AsyncTelemetryUnaryUnaryInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
    def __init__(self, manager: ClientTelemetryManager) -> None:
        self._manager = manager

    async def intercept_unary_unary(self, continuation: Callable, call_details: Any, request: Any):
        operation = _operation_from_method(call_details.method)
        if operation is None or _LOGICAL_TELEMETRY_DEPTH.get() > 0:
            return await continuation(call_details, request)
        started_at = time.perf_counter()
        call = await continuation(call_details, request)

        def done(completed: Any) -> None:
            async def finalize() -> None:
                error: BaseException | None = None
                try:
                    response = await completed
                    error = _response_error(response)
                except BaseException as exc:
                    error = exc
                self._manager.record_operation(
                    operation,
                    _collection_from_request(request),
                    started_at,
                    error,
                    _request_id_from_metadata(call_details.metadata),
                )

            asyncio.get_running_loop().create_task(finalize())

        call.add_done_callback(done)
        return call


def _operation_from_method(method: Any) -> str | None:
    name = method.decode() if isinstance(method, bytes) else str(method)
    return _OPERATION_NAMES.get(name.rsplit("/", 1)[-1])


def _collection_from_request(request: Any) -> str:
    for name in ("collection_name", "collectionName"):
        value = getattr(request, name, "")
        if value:
            return str(value)
    return ""


def _request_id_from_metadata(metadata: Any) -> str:
    for key, value in metadata or ():
        if key in ("client_request_id", "client-request-id"):
            request_id = value.decode() if isinstance(value, bytes) else str(value)
            return request_id if is_valid_client_request_id(request_id) else ""
    return ""


def _response_error(response: Any) -> BaseException | None:
    """Return an error for Milvus failures carried in an otherwise-OK gRPC response."""

    status = (
        response if isinstance(response, common_pb2.Status) else getattr(response, "status", None)
    )
    if status is None:
        return None
    if int(getattr(status, "error_code", 0)) == 0 and int(getattr(status, "code", 0)) == 0:
        return None
    return RuntimeError(getattr(status, "reason", "") or "Milvus request failed")


def _parse_rfc3339_ms(value: Any) -> int:
    if not value:
        msg = "payload is required with start_time and end_time"
        raise ValueError(msg)
    from datetime import datetime  # noqa: PLC0415

    match = _RFC3339_PATTERN.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        msg = "timestamp must use RFC3339 with seconds and timezone"
        raise ValueError(msg)
    # Python 3.9 rejects fractional seconds longer than microsecond precision,
    # although RFC3339 permits any number of fractional digits. Telemetry stores
    # milliseconds, so safely truncate only after the complete syntax is validated.
    fraction = (match.group("fraction") or "")[:7]
    timezone = "+00:00" if match.group("timezone") == "Z" else match.group("timezone")
    normalized = f"{match.group('datetime')}{fraction}{timezone}"
    return int(datetime.fromisoformat(normalized).timestamp() * 1000)


def _metrics_dict(metrics: Metrics) -> dict[str, Any]:
    return {
        "request_count": metrics.request_count,
        "success_count": metrics.success_count,
        "error_count": metrics.error_count,
        "avg_latency_ms": metrics.avg_latency_ms,
        "p99_latency_ms": metrics.p99_latency_ms,
        "max_latency_ms": metrics.max_latency_ms,
    }


def _snapshot_dict(snapshot: MetricsSnapshot) -> dict[str, Any]:
    return {
        "timestamp": snapshot.timestamp,
        "end_time": snapshot.end_time,
        "metrics": {
            item.operation: _metrics_dict(item.global_metrics) for item in snapshot.metrics
        },
    }


def _aggregate_snapshots(
    snapshots: Iterable[MetricsSnapshot], start_ms: int, end_ms: int
) -> dict[str, Any]:
    snapshots = list(snapshots)
    totals: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        for item in snapshot.metrics:
            metrics = item.global_metrics
            total = totals.setdefault(
                item.operation,
                {
                    "request_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "weighted_avg": 0.0,
                    "max_latency_ms": 0.0,
                    "latency_groups": [],
                },
            )
            total["request_count"] += metrics.request_count
            total["success_count"] += metrics.success_count
            total["error_count"] += metrics.error_count
            total["weighted_avg"] += metrics.avg_latency_ms * metrics.request_count
            total["max_latency_ms"] = max(total["max_latency_ms"], metrics.max_latency_ms)
            samples = item._global_latency_samples_us
            if samples:
                weight = metrics.request_count / len(samples)
                total["latency_groups"].append((samples, weight))
            elif metrics.request_count:
                # Runtime snapshots created before samples were retained remain usable.
                total["latency_groups"].append(
                    (
                        (metrics.p99_latency_ms * 1000.0,),
                        float(metrics.request_count),
                    )
                )
    result = {}
    for operation, total in totals.items():
        count = int(total["request_count"])
        p99_latency_ms = 0.0
        groups = total["latency_groups"]
        if count and groups:
            threshold = count * 0.99
            cumulative = 0.0
            heap = [
                (samples[0], group_index, 0)
                for group_index, (samples, _weight) in enumerate(groups)
                if samples
            ]
            heapq.heapify(heap)
            while heap:
                latency_us, group_index, sample_index = heapq.heappop(heap)
                samples, weight = groups[group_index]
                cumulative += weight
                p99_latency_ms = latency_us / 1000.0
                if cumulative > threshold:
                    break
                next_index = sample_index + 1
                if next_index < len(samples):
                    heapq.heappush(heap, (samples[next_index], group_index, next_index))
        result[operation] = {
            "request_count": count,
            "success_count": int(total["success_count"]),
            "error_count": int(total["error_count"]),
            "avg_latency_ms": total["weighted_avg"] / count if count else 0.0,
            "p99_latency_ms": p99_latency_ms,
            "max_latency_ms": total["max_latency_ms"],
        }
    return {
        "aggregated": {"start_time": start_ms, "end_time": end_ms, "metrics": result},
        "snapshot_count": len(snapshots),
    }


def new_client_request_id() -> str:
    """Return a lowercase 32-character OpenTelemetry TraceID."""

    while True:
        value = os.urandom(16)
        if any(value):
            return value.hex()
