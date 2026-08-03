"""Client-side telemetry, heartbeat, and server command support."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import grpc

from pymilvus.client import __version__
from pymilvus.grpc_gen import common_pb2, milvus_pb2

_DEFAULT_HEARTBEAT_INTERVAL = 30.0
_MAX_UNSUPPORTED_BACKOFF = 30 * 60.0
_MAX_REPLY_PAYLOAD_SIZE = 1024 * 1024
_SAMPLING_DENOMINATOR = 10_000
_LATENCY_SAMPLE_SIZE = 1000
_SNAPSHOT_LIMIT = 120

_OPERATION_NAMES = {
    "Insert": "Insert",
    "Delete": "Delete",
    "Upsert": "Upsert",
    "Search": "Search",
    "HybridSearch": "HybridSearch",
    "Query": "Query",
    "RunAnalyzer": "RunAnalyzer",
}


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
            return value
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
        if self.heartbeat_interval <= 0:
            msg = "heartbeat_interval must be positive"
            raise ValueError(msg)
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

    def snapshot_and_reset(self) -> Metrics | None:
        if self.request_count == 0:
            return None
        samples = sorted(self.samples)
        p99 = samples[min(len(samples) - 1, int(len(samples) * 0.99))] if samples else 0
        result = Metrics(
            request_count=self.request_count,
            success_count=self.success_count,
            error_count=self.error_count,
            avg_latency_ms=self.total_latency_us / self.request_count / 1000.0,
            p99_latency_ms=p99 / 1000.0,
            max_latency_ms=self.max_latency_us / 1000.0,
        )
        self.__init__()
        return result


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
            global_metrics = self.global_bucket.snapshot_and_reset()
            if global_metrics is None:
                return None
            collection_metrics: dict[str, Metrics] = {}
            for name, bucket in self.collections.items():
                metrics = bucket.snapshot_and_reset()
                if metrics is not None and (
                    enabled_collections is None or name in enabled_collections
                ):
                    collection_metrics[name] = metrics
            self.collections = {}
            return OperationMetrics("", global_metrics, collection_metrics)


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
        runtime_client_id: str = "",
    ) -> None:
        self._stub_provider = stub_provider
        self._config = TelemetryConfig.from_value(config)
        self._config_lock = threading.RLock()
        self._user = user or ""
        self._database_provider = database_provider or (lambda: "")
        self._config_provider = config_provider or (dict)
        self._client_id = self._config.client_id or runtime_client_id or str(uuid.uuid4())
        self._client_id_stable = bool(self._config.client_id)

        self._collectors: dict[str, _OperationCollector] = {}
        self._collectors_lock = threading.Lock()
        self._enabled_collections: set[str] = set()
        self._all_collections_enabled = False
        self._collections_lock = threading.RLock()
        self._errors: deque[ErrorInfo] = deque(maxlen=self._config.error_max_count)
        self._errors_lock = threading.Lock()
        self._snapshots: deque[MetricsSnapshot] = deque(maxlen=_SNAPSHOT_LIMIT)
        self._snapshots_lock = threading.Lock()

        self._handlers: dict[str, Callable[[ClientCommand], CommandReply]] = {}
        self._handlers_lock = threading.RLock()
        self._pending_replies: list[common_pb2.CommandReply] = []
        self._pending_lock = threading.Lock()
        self._executed_commands: dict[str, int] = {}
        self._executed_lock = threading.Lock()
        self._last_command_timestamp = 0
        self._config_hash = ""
        self._sampling_counter = 0
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
        return self._unsupported_streak == 0

    def last_heartbeat_error(self) -> BaseException | None:
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
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None

    def register_command_handler(
        self, command_type: str, handler: Callable[[ClientCommand], CommandReply]
    ) -> None:
        with self._handlers_lock:
            self._handlers[command_type] = handler

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
            return list(self._snapshots)

    def process_commands(self, commands: Iterable[Any]) -> None:
        commands = list(commands)
        with self._state_lock:
            last_timestamp = self._last_command_timestamp
        max_timestamp = last_timestamp
        has_persistent = False

        for command in commands:
            local = ClientCommand(
                command_id=command.command_id,
                command_type=command.command_type,
                payload=bytes(command.payload),
                create_time=command.create_time,
                persistent=command.persistent,
                target_scope=command.target_scope,
            )
            max_timestamp = max(max_timestamp, local.create_time)
            has_persistent = has_persistent or local.persistent

            if local.create_time < last_timestamp:
                self._queue_reply(CommandReply(local.command_id, True))
                continue
            with self._executed_lock:
                already_executed = local.command_id in self._executed_commands
            if already_executed:
                self._queue_reply(CommandReply(local.command_id, True))
                continue

            reply = self._handle_command(local)
            with self._executed_lock:
                self._executed_commands[local.command_id] = local.create_time
            if reply is not None:
                self._queue_reply(reply)

        with self._executed_lock:
            self._executed_commands = {
                command_id: timestamp
                for command_id, timestamp in self._executed_commands.items()
                if timestamp > last_timestamp
            }
        with self._state_lock:
            if has_persistent:
                self._config_hash = self.calculate_config_hash(commands)
            self._last_command_timestamp = max(self._last_command_timestamp, max_timestamp)

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
            return self._config.heartbeat_interval or _DEFAULT_HEARTBEAT_INTERVAL

    def _next_heartbeat_delay(self) -> float:
        interval = self._heartbeat_interval()
        if self._unsupported_streak <= 0:
            return interval
        delay = interval * math.pow(2, self._unsupported_streak)
        return max(interval, min(_MAX_UNSUPPORTED_BACKOFF, delay))

    def _heartbeat_loop(self) -> None:
        self._create_snapshot()
        self._send_heartbeat()
        while not self._stop_event.wait(self._next_heartbeat_delay()):
            self._create_snapshot()
            self._send_heartbeat()

    def _send_heartbeat(self) -> None:
        if not self._enabled():
            return
        stub = self._stub_provider()
        if stub is None:
            return

        with self._snapshots_lock:
            latest = self._snapshots[-1] if self._snapshots else None
        metrics = self._to_proto_metrics(latest.metrics if latest else [])
        with self._pending_lock:
            replies = list(self._pending_replies)
        with self._state_lock:
            config_hash = self._config_hash
            last_timestamp = self._last_command_timestamp

        request = milvus_pb2.ClientHeartbeatRequest(
            client_info=self._build_client_info(),
            report_timestamp=int(time.time() * 1000),
            metrics=metrics,
            command_replies=replies,
            config_hash=config_hash,
            last_command_timestamp=last_timestamp,
        )
        try:
            response = stub.ClientHeartbeat(request, timeout=10, wait_for_ready=False)
            if response.status.code != 0 or response.status.error_code != 0:
                raise RuntimeError(response.status.reason or "client telemetry heartbeat failed")
        except grpc.RpcError as exc:
            self._last_heartbeat_error = exc
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                self._unsupported_streak += 1
            return
        except BaseException as exc:  # best-effort background channel
            self._last_heartbeat_error = exc
            return

        self._last_heartbeat_error = None
        self._unsupported_streak = 0
        with self._pending_lock:
            del self._pending_replies[: len(replies)]
        self.process_commands(response.commands)

    def _build_client_info(self) -> common_pb2.ClientInfo:
        reserved = {
            "client_id": self._client_id,
            "client_id_stable": str(self._client_id_stable).lower(),
        }
        database = self._database_provider() or ""
        if database:
            reserved["db_name"] = database
        return common_pb2.ClientInfo(
            sdk_type="Python",
            sdk_version=__version__,
            local_time=time.ctime(),
            user=self._user,
            host=socket.gethostname(),
            reserved=reserved,
        )

    def _should_sample(self, rate: float) -> bool:
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        threshold = int(rate * _SAMPLING_DENOMINATOR)
        if threshold == 0:
            return False
        with self._state_lock:
            self._sampling_counter += 1
            return self._sampling_counter % _SAMPLING_DENOMINATOR < threshold

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
            start = now - int(self._heartbeat_interval() * 1000)
        self._last_snapshot_end = now
        with self._snapshots_lock:
            self._snapshots.append(MetricsSnapshot(start, now, metrics))

    @staticmethod
    def _to_proto_metrics(items: Iterable[OperationMetrics]) -> list[common_pb2.OperationMetrics]:
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
            return handler(command)
        except BaseException as exc:
            return CommandReply(command.command_id, False, error_message=str(exc))

    def _register_default_handlers(self) -> None:
        self.register_command_handler("push_config", self._handle_push_config)
        self.register_command_handler("collection_metrics", self._handle_collection_metrics)
        self.register_command_handler("show_errors", self._handle_show_errors)
        self.register_command_handler("show_latency_history", self._handle_latency_history)
        self.register_command_handler("get_config", self._handle_get_config)

    @staticmethod
    def _payload(command: ClientCommand) -> dict[str, Any]:
        return json.loads(command.payload.decode()) if command.payload else {}

    def _handle_push_config(self, command: ClientCommand) -> CommandReply:
        payload = self._payload(command)
        with self._config_lock:
            if "enabled" in payload:
                self._config.enabled = bool(payload["enabled"])
            if "heartbeat_interval_ms" in payload:
                interval_ms = int(payload["heartbeat_interval_ms"])
                if interval_ms <= 0:
                    msg = "heartbeat_interval_ms must be positive"
                    raise ValueError(msg)
                self._config.heartbeat_interval = interval_ms / 1000.0
            if "sampling_rate" in payload:
                self._config.sampling_rate = min(1.0, max(0.0, float(payload["sampling_rate"])))
        return CommandReply(command.command_id, True)

    def _handle_collection_metrics(self, command: ClientCommand) -> CommandReply:
        if not command.payload:
            with self._collections_lock:
                payload = {
                    "enabled_collections": sorted(self._enabled_collections),
                    "all_collections_enabled": self._all_collections_enabled,
                }
            return CommandReply(command.command_id, True, payload=json.dumps(payload).encode())

        payload = self._payload(command)
        collections = list(payload.get("collections") or [])
        enabled = bool(payload.get("enabled"))
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
        errors = [vars(item) for item in self.get_recent_errors(int(payload.get("max_count", 100)))]
        encoded = json.dumps(errors, separators=(",", ":")).encode()
        while len(encoded) > _MAX_REPLY_PAYLOAD_SIZE and len(errors) > 1:
            errors = errors[: max(1, len(errors) // 2)]
            encoded = json.dumps(errors, separators=(",", ":")).encode()
        if len(encoded) > _MAX_REPLY_PAYLOAD_SIZE and errors:
            while len(encoded) > _MAX_REPLY_PAYLOAD_SIZE and len(errors[0]["error_msg"]) > 1:
                message = errors[0]["error_msg"]
                errors[0]["error_msg"] = message[: max(1, len(message) // 2)] + "...(truncated)"
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
                telemetry_heartbeat_interval_ms=int(self._config.heartbeat_interval * 1000),
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
        start_ms = _parse_rfc3339_ms(payload.get("start_time"))
        end_ms = _parse_rfc3339_ms(payload.get("end_time"))
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
        if payload.get("detail"):
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
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _async_heartbeat_loop(self) -> None:
        self._create_snapshot()
        await self._send_heartbeat_async()
        while not self._stop_event.is_set():
            await asyncio.sleep(self._next_heartbeat_delay())
            self._create_snapshot()
            await self._send_heartbeat_async()

    async def _send_heartbeat_async(self) -> None:
        if not self._enabled():
            return
        stub = self._stub_provider()
        if stub is None:
            return
        with self._snapshots_lock:
            latest = self._snapshots[-1] if self._snapshots else None
        with self._pending_lock:
            replies = list(self._pending_replies)
        with self._state_lock:
            config_hash = self._config_hash
            last_timestamp = self._last_command_timestamp
        request = milvus_pb2.ClientHeartbeatRequest(
            client_info=self._build_client_info(),
            report_timestamp=int(time.time() * 1000),
            metrics=self._to_proto_metrics(latest.metrics if latest else []),
            command_replies=replies,
            config_hash=config_hash,
            last_command_timestamp=last_timestamp,
        )
        try:
            response = await stub.ClientHeartbeat(request, timeout=10, wait_for_ready=False)
            if response.status.code != 0 or response.status.error_code != 0:
                raise RuntimeError(response.status.reason or "client telemetry heartbeat failed")
        except grpc.RpcError as exc:
            self._last_heartbeat_error = exc
            if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                self._unsupported_streak += 1
            return
        except BaseException as exc:
            self._last_heartbeat_error = exc
            return
        self._last_heartbeat_error = None
        self._unsupported_streak = 0
        with self._pending_lock:
            del self._pending_replies[: len(replies)]
        self.process_commands(response.commands)


class TelemetryUnaryUnaryInterceptor(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, manager: ClientTelemetryManager) -> None:
        self._manager = manager

    def intercept_unary_unary(self, continuation: Callable, call_details: Any, request: Any):
        operation = _operation_from_method(call_details.method)
        if operation is None:
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
        if operation is None:
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
            return value.decode() if isinstance(value, bytes) else str(value)
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

    normalized = str(value).replace("Z", "+00:00")
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
    totals: dict[str, dict[str, float]] = {}
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
                    "weighted_p99": 0.0,
                    "max_latency_ms": 0.0,
                },
            )
            total["request_count"] += metrics.request_count
            total["success_count"] += metrics.success_count
            total["error_count"] += metrics.error_count
            total["weighted_avg"] += metrics.avg_latency_ms * metrics.request_count
            total["weighted_p99"] += metrics.p99_latency_ms * metrics.request_count
            total["max_latency_ms"] = max(total["max_latency_ms"], metrics.max_latency_ms)
    result = {}
    for operation, total in totals.items():
        count = int(total["request_count"])
        result[operation] = {
            "request_count": count,
            "success_count": int(total["success_count"]),
            "error_count": int(total["error_count"]),
            "avg_latency_ms": total["weighted_avg"] / count if count else 0.0,
            "p99_latency_ms": total["weighted_p99"] / count if count else 0.0,
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
