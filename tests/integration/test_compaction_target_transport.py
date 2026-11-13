"""Exercise Create's no-replay guarantee through a real gRPC transport."""

import asyncio
import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc
import pytest
from pymilvus import AsyncMilvusClient, MilvusClient
from pymilvus.exceptions import ErrorCode, MilvusException
from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc


class TargetService(milvus_pb2_grpc.MilvusServiceServicer):
    def __init__(self):
        self.identifier = 123
        self.creates = []
        self.create_started = threading.Event()
        self.create_release = None

    def Connect(self, request, context):
        return milvus_pb2.ConnectResponse(status=common_pb2.Status(), identifier=self.identifier)

    def CreateCompactionTarget(self, request, context):
        self.creates.append((request, dict(context.invocation_metadata())))
        self.create_started.set()
        if self.create_release is not None:
            self.create_release.wait(timeout=5)
        context.abort(grpc.StatusCode.UNAVAILABLE, "Create response lost")


@pytest.fixture
def target_server():
    service = TargetService()
    with ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool)
        milvus_pb2_grpc.add_MilvusServiceServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            yield f"http://127.0.0.1:{port}", service
        finally:
            server.stop(grace=0).wait()


@pytest.fixture
def create_retry_options():
    return {
        "grpc.enable_retries": 1,
        "grpc.service_config": json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [
                            {
                                "service": "milvus.proto.milvus.MilvusService",
                                "method": "CreateCompactionTarget",
                            }
                        ],
                        "retryPolicy": {
                            "maxAttempts": 3,
                            "initialBackoff": "0.01s",
                            "maxBackoff": "0.02s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    }
                ]
            }
        ),
    }


def test_create_response_loss_disables_configured_transport_replay(
    target_server, create_retry_options
):
    uri, service = target_server

    # A raw stub proves the configured channel reaches the application multiple times.
    with grpc.insecure_channel(
        uri.removeprefix("http://"), options=create_retry_options.items()
    ) as channel:
        raw = milvus_pb2_grpc.MilvusServiceStub(channel)
        with pytest.raises(grpc.RpcError) as error:
            raw.CreateCompactionTarget(milvus_pb2.CreateCompactionTargetRequest(), timeout=3)
    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    assert len(service.creates) == 3
    service.creates.clear()

    client = MilvusClient(
        uri,
        token="user:password",
        db_name="work",
        grpc_options=create_retry_options,
        dedicated=True,
    )
    try:
        service.identifier = 456
        client._handler.reconnect(timeout=3)
        with pytest.raises(MilvusException, match="outcome is unknown") as error:
            client.create_compaction_target(
                "items", segment_ids=[5, 6], client_request_id="create-42", timeout=3
            )
    finally:
        client.close()

    assert error.value.code == ErrorCode.COMPACTION_TARGET_CREATE_OUTCOME_UNKNOWN
    assert not error.value.retriable
    [(request, metadata)] = service.creates
    assert request.db_name == "work"
    assert list(request.rewrite.segment_ids) == [5, 6]
    assert metadata["authorization"] == base64.b64encode(b"user:password").decode()
    assert metadata["identifier"] == "456"
    assert metadata["client-request-id"] == "create-42"


@pytest.mark.asyncio
async def test_async_create_response_loss_disables_configured_transport_replay(
    target_server, create_retry_options
):
    uri, service = target_server
    async with AsyncMilvusClient(uri, grpc_options=create_retry_options, dedicated=True) as client:
        with pytest.raises(MilvusException, match="outcome is unknown") as error:
            await client.create_compaction_target("items", timeout=3)

    assert error.value.code == ErrorCode.COMPACTION_TARGET_CREATE_OUTCOME_UNKNOWN
    assert not error.value.retriable
    [(request, metadata)] = service.creates
    assert request.WhichOneof("target") == "rewrite"
    assert metadata["identifier"] == "123"


@pytest.mark.asyncio
async def test_async_create_cancellation_after_dispatch_reports_unknown(
    target_server, create_retry_options
):
    uri, service = target_server
    service.create_release = threading.Event()
    try:
        async with AsyncMilvusClient(
            uri, grpc_options=create_retry_options, dedicated=True
        ) as client:
            task = asyncio.create_task(client.create_compaction_target("items", timeout=5))
            assert await asyncio.to_thread(service.create_started.wait, 3)
            task.cancel()
            with pytest.raises(MilvusException, match="outcome is unknown") as error:
                await task
    finally:
        service.create_release.set()

    assert error.value.code == ErrorCode.COMPACTION_TARGET_CREATE_OUTCOME_UNKNOWN
    assert not error.value.retriable
    assert len(service.creates) == 1
