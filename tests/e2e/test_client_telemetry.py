import asyncio
import json
import os
import time
import urllib.parse
import urllib.request
import uuid

import pytest
from pymilvus import (
    AsyncMilvusClient,
    MilvusClient,
    TelemetryConfig,
    connections,
    new_client_request_id,
)
from pymilvus.exceptions import MilvusException
from pymilvus.grpc_gen import common_pb2

pytestmark = pytest.mark.skipif(
    os.getenv("MILVUS_TELEMETRY_E2E") != "true",
    reason="set MILVUS_TELEMETRY_E2E=true to run against a local Milvus",
)

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
TELEMETRY_API = os.getenv("MILVUS_TELEMETRY_API", "http://127.0.0.1:9091/api/v1/_telemetry")


def test_default_telemetry_registers_automatically():
    client = MilvusClient(MILVUS_URI, dedicated=True)
    try:
        manager = client.get_telemetry()
        assert manager.ready is True
        _wait_for(
            manager.client_id,
            "default client registration",
            lambda candidate: candidate["status"] == "active",
        )
    finally:
        client.close()


def test_legacy_connection_registers_automatically():
    alias = f"telemetry-e2e-{uuid.uuid4()}"
    client_id = f"e2e-python-legacy-{uuid.uuid4()}"
    try:
        connections.connect(
            alias=alias,
            uri=MILVUS_URI,
            telemetry_config=TelemetryConfig(
                heartbeat_interval=0.5,
                client_id=client_id,
            ),
        )
        _wait_for(
            client_id,
            "legacy client registration",
            lambda candidate: candidate["status"] == "active",
        )
    finally:
        connections.disconnect(alias)


@pytest.mark.asyncio
async def test_async_client_metrics_and_request_id_round_trip():
    client_id = f"e2e-python-async-{uuid.uuid4()}"
    client = AsyncMilvusClient(
        MILVUS_URI,
        telemetry_config=TelemetryConfig(
            heartbeat_interval=0.5,
            client_id=client_id,
        ),
    )
    try:
        manager = await client.get_telemetry()
        await asyncio.to_thread(
            _wait_for,
            client_id,
            "async client registration",
            lambda state: state["status"] == "active",
        )

        analyzer = await client.run_analyzer(
            "hello async telemetry", analyzer_params={"type": "standard"}
        )
        assert analyzer.tokens == ["hello", "async", "telemetry"]
        await asyncio.to_thread(
            _wait_for,
            client_id,
            "async RunAnalyzer metric",
            lambda state: _has_metric(state, "RunAnalyzer", "success_count", 1),
        )

        collections_command = await asyncio.to_thread(
            _push_command,
            client_id,
            "collection_metrics",
            {"collections": ["*"], "enabled": True},
        )
        assert (await asyncio.to_thread(_wait_for_reply, client_id, collections_command))[
            "success"
        ] is True

        request_id = new_client_request_id()
        with pytest.raises(MilvusException):
            await client.query(
                "telemetry_e2e_async_missing",
                filter="id > 0",
                client_request_id=request_id,
            )
        await asyncio.to_thread(
            _wait_for,
            client_id,
            "async failed Query collection metric",
            lambda state: _has_metric(
                state,
                "Query",
                "error_count",
                1,
                collection="telemetry_e2e_async_missing",
            ),
        )

        errors_command = await asyncio.to_thread(
            _push_command, client_id, "show_errors", {"max_count": 10}
        )
        errors_reply = await asyncio.to_thread(_wait_for_reply, client_id, errors_command)
        errors = json.loads(errors_reply["payload"])
        assert any(error.get("request_id") == request_id for error in errors)
        assert manager.last_command_timestamp > 0
    finally:
        await client.close()


def test_metrics_commands_config_and_request_id_round_trip():
    client_id = f"e2e-python-{uuid.uuid4()}"
    config_command = None
    client = MilvusClient(
        MILVUS_URI,
        telemetry_config=TelemetryConfig(
            heartbeat_interval=0.5,
            sampling_rate=1.0,
            client_id=client_id,
        ),
    )
    try:
        manager = client.get_telemetry()
        assert manager.client_id == client_id
        _wait_for(client_id, "client registration", lambda state: state["status"] == "active")

        analyzer = client.run_analyzer(
            "hello milvus telemetry", analyzer_params={"type": "standard"}
        )
        assert analyzer.tokens == ["hello", "milvus", "telemetry"]
        _wait_for(
            client_id,
            "RunAnalyzer metric",
            lambda state: _has_metric(state, "RunAnalyzer", "success_count", 1),
        )

        collections_command = _push_command(
            client_id,
            "collection_metrics",
            {"collections": ["*"], "enabled": True},
        )
        assert _wait_for_reply(client_id, collections_command)["success"] is True

        request_id = new_client_request_id()
        with pytest.raises(MilvusException):
            client.query(
                "telemetry_e2e_missing",
                filter="id > 0",
                client_request_id=request_id,
            )
        _wait_for(
            client_id,
            "failed Query collection metric",
            lambda state: _has_metric(
                state,
                "Query",
                "error_count",
                1,
                collection="telemetry_e2e_missing",
            ),
        )

        errors_command = _push_command(client_id, "show_errors", {"max_count": 10})
        errors_reply = _wait_for_reply(client_id, errors_command)
        assert errors_reply["success"] is True
        errors = json.loads(errors_reply["payload"])
        assert any(
            error.get("operation") == "Query" and error.get("request_id") == request_id
            for error in errors
        )

        config_payload = {"sampling_rate": 0.75, "heartbeat_interval_ms": 600}
        config_command = _push_command(client_id, "push_config", config_payload, persistent=True)
        assert _wait_for_reply(client_id, config_command)["success"] is True
        expected_hash = manager.calculate_config_hash(
            [
                common_pb2.ClientCommand(
                    command_id=config_command,
                    command_type="push_config",
                    payload=json.dumps(config_payload, separators=(",", ":")).encode(),
                    persistent=True,
                )
            ]
        )
        assert manager.config_hash == expected_hash
        assert manager.last_command_timestamp > 0

        config_reply = _wait_for_reply(client_id, _push_command(client_id, "get_config", {}))
        user_config = json.loads(config_reply["payload"])["user_config"]
        assert user_config["telemetry_sampling_rate"] == 0.75
        assert user_config["telemetry_heartbeat_interval_ms"] == 600
        assert user_config["all_collections_enabled"] is True
    finally:
        try:
            if config_command is not None:
                _delete_command(config_command)
        finally:
            client.close()


def _request_json(url, method="GET", body=None):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported telemetry URL scheme: {parsed.scheme}")
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data, method=method)  # noqa: S310
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        return json.load(response)


def _client_state(client_id):
    query = urllib.parse.urlencode({"client_id": client_id, "include_metrics": "true"})
    clients = _request_json(f"{TELEMETRY_API}/clients?{query}").get("clients", [])
    return clients[0] if clients else None


def _wait_for(client_id, label, predicate, timeout=15):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _client_state(client_id)
        if last is not None and predicate(last):
            return last
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {label}; last={last}")


def _push_command(client_id, command_type, payload, persistent=False):
    response = _request_json(
        f"{TELEMETRY_API}/commands",
        method="POST",
        body={
            "command_type": command_type,
            "target_client_id": client_id,
            "payload": payload,
            "ttl_seconds": 30,
            "persistent": persistent,
        },
    )
    return response["command_id"]


def _delete_command(command_id):
    encoded_command_id = urllib.parse.quote(command_id, safe="")
    _request_json(f"{TELEMETRY_API}/commands/{encoded_command_id}", method="DELETE")


def _wait_for_reply(client_id, command_id):
    state = _wait_for(
        client_id,
        f"command reply {command_id}",
        lambda candidate: _find_reply(candidate, command_id) is not None,
    )
    return _find_reply(state, command_id)


def _find_reply(state, command_id):
    return next(
        (
            reply
            for reply in state.get("command_replies") or []
            if reply.get("command_id") == command_id
        ),
        None,
    )


def _has_metric(state, operation, counter, minimum, collection=None):
    for metric in state.get("metrics") or []:
        if metric.get("operation") != operation:
            continue
        if metric.get("global", {}).get(counter, 0) < minimum:
            continue
        return collection is None or collection in metric.get("collection_metrics", {})
    return False
