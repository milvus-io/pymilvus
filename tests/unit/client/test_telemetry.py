import json
import time

from pymilvus.client.telemetry import (
    ClientCommand,
    ClientTelemetryManager,
    CommandReply,
    TelemetryConfig,
    _response_error,
    new_client_request_id,
)
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


def test_new_client_request_id_is_valid_trace_id():
    request_id = new_client_request_id()
    assert len(request_id) == 32
    assert request_id != "0" * 32
    int(request_id, 16)


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
