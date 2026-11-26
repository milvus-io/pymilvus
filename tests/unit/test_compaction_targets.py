import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest
import pytest_asyncio
from pymilvus import AsyncMilvusClient, MilvusClient
from pymilvus.client import async_grpc_handler, compaction_target, grpc_handler
from pymilvus.client.connection_manager import AsyncGlobalStrategy, GlobalStrategy
from pymilvus.exceptions import ErrorCode, MilvusException, ParamError
from pymilvus.grpc_gen import common_pb2, milvus_pb2, milvus_pb2_grpc
from pymilvus.milvus_client import async_milvus_client, milvus_client


class TransportError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return self._code.name


class TargetTransport:
    """In-process RPC stand-in for the client and handler, using real protobufs."""

    def __init__(self):
        self.creates = []
        self.create_response = milvus_pb2.CreateCompactionTargetResponse(
            status=common_pb2.Status(), target_id=42
        )
        self.create_error = None
        self.describes = []
        self.describe_responses = []
        self.describe_error = None
        self.describe_started = None
        self.describe_release = None
        self.lists = []
        self.list_response = milvus_pb2.ListCompactionTargetsResponse(status=common_pb2.Status())
        self.list_error = None
        self.drops = []
        self.drop_response = milvus_pb2.DropCompactionTargetResponse(status=common_pb2.Status())
        self.drop_error = None

    def connect(self, request, **kwargs):
        return milvus_pb2.ConnectResponse(status=common_pb2.Status(), identifier=123)

    def create(self, request, **kwargs):
        self.creates.append((request, dict(kwargs["metadata"])))
        if self.create_error:
            raise TransportError(self.create_error)
        return self.create_response

    def describe(self, request, **kwargs):
        self.describes.append(request.target_id)
        return self._describe_response()

    async def async_describe(self, request, **kwargs):
        self.describes.append(request.target_id)
        if self.describe_release is not None:
            self.describe_started.set()
            await self.describe_release.wait()
        return self._describe_response()

    def _describe_response(self):
        if self.describe_error:
            raise TransportError(self.describe_error)
        return self.describe_responses.pop(0)

    def list_targets(self, request, **kwargs):
        self.lists.append(request)
        if self.list_error:
            raise TransportError(self.list_error)
        return self.list_response

    def drop(self, request, **kwargs):
        self.drops.append(request.target_id)
        if self.drop_error:
            raise TransportError(self.drop_error)
        return self.drop_response


@pytest.fixture
def target_transport(monkeypatch, mock_grpc_channel, mock_async_channel):
    service = TargetTransport()

    def connect(request, **kwargs):
        return service.connect(request, **kwargs)

    sync_methods = {
        "Connect": connect,
        "CreateCompactionTarget": service.create,
        "DescribeCompactionTarget": service.describe,
        "ListCompactionTargets": service.list_targets,
        "DropCompactionTarget": service.drop,
    }

    def async_method(method):
        async def call(*args, **kwargs):
            return method(*args, **kwargs)

        return call

    async_methods = {name: async_method(method) for name, method in sync_methods.items()}
    async_methods["DescribeCompactionTarget"] = service.async_describe
    sync_channel = mock_grpc_channel
    sync_channel.__enter__.return_value = sync_channel
    async_channel = mock_async_channel
    async_channel.__aenter__.return_value = async_channel
    monkeypatch.setattr(grpc, "insecure_channel", lambda *args, **kwargs: sync_channel)
    monkeypatch.setattr(grpc, "intercept_channel", lambda channel, *args: channel)
    monkeypatch.setattr(grpc.aio, "insecure_channel", lambda *args, **kwargs: async_channel)
    monkeypatch.setattr(
        milvus_pb2_grpc,
        "MilvusServiceStub",
        lambda channel: SimpleNamespace(
            **(async_methods if channel is async_channel else sync_methods)
        ),
    )
    return "http://localhost:19530", service


@pytest.fixture
def fake_clock(monkeypatch):
    # Patch module references instead of the shared time/asyncio modules.
    now = 0.0

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    async def async_sleep(seconds):
        sleep(seconds)

    monkeypatch.setattr(compaction_target, "time", SimpleNamespace(monotonic=monotonic))
    for module in (grpc_handler, milvus_client):
        monkeypatch.setattr(module, "time", SimpleNamespace(sleep=sleep))
    for module in (async_grpc_handler, async_milvus_client):
        monkeypatch.setattr(
            module,
            "asyncio",
            SimpleNamespace(
                sleep=async_sleep,
                Lock=asyncio.Lock,
                wait_for=asyncio.wait_for,
                create_task=asyncio.create_task,
                TimeoutError=asyncio.TimeoutError,
                CancelledError=asyncio.CancelledError,
            ),
        )
    return monotonic


class SyncClientCalls:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        async def call(*args, **kwargs):
            return getattr(self._client, name)(*args, **kwargs)

        return call


@pytest_asyncio.fixture(params=["sync", "async"])
async def target_client(request, target_transport):
    uri, service = target_transport
    if request.param == "async":
        async with AsyncMilvusClient(uri, dedicated=True) as client:
            yield client, service
    else:
        client = MilvusClient(uri, dedicated=True)
        try:
            yield SyncClientCalls(client), service
        finally:
            client.close()


@pytest.mark.asyncio
async def test_create_rewrite_returns_target_and_preserves_request(target_client):
    client, service = target_client
    target = await client.create_compaction_target(
        "items", db_name="work", segment_ids=[5, 6], client_request_id="create-42"
    )
    assert target.target_id == 42
    assert target.total_rows == 0
    assert target.optimize is None
    [(request, metadata)] = service.creates
    assert request.db_name == "work"
    assert request.collection_name == "items"
    assert request.WhichOneof("target") == "rewrite"
    assert list(request.rewrite.segment_ids) == [5, 6]
    assert metadata["client-request-id"] == "create-42"


@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_create_with_caller_channel_fails_before_dispatch(
    async_client, target_transport, mock_grpc_channel, mock_async_channel
):
    uri, service = target_transport
    if async_client:
        async with AsyncMilvusClient(uri, channel=mock_async_channel, dedicated=True) as client:
            with pytest.raises(ParamError, match="caller-supplied gRPC channel"):
                await client.create_compaction_target("items")
    else:
        client = MilvusClient(uri, channel=mock_grpc_channel, dedicated=True)
        try:
            with pytest.raises(ParamError, match="caller-supplied gRPC channel"):
                client.create_compaction_target("items")
        finally:
            client.close()

    assert service.creates == []


@pytest.mark.parametrize("async_client", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_create_with_caller_channel_still_rejects_when_handler_is_shared(
    async_client, target_transport, mock_grpc_channel, mock_async_channel
):
    uri, service = target_transport
    if async_client:
        async with AsyncMilvusClient(uri) as uri_client, AsyncMilvusClient(
            uri, channel=mock_async_channel
        ) as channel_client:
            assert uri_client._handler is channel_client._handler
            with pytest.raises(ParamError, match="caller-supplied gRPC channel"):
                await channel_client.create_compaction_target("items")
    else:
        uri_client = MilvusClient(uri)
        channel_client = MilvusClient(uri, channel=mock_grpc_channel)
        try:
            assert uri_client._handler is channel_client._handler
            with pytest.raises(ParamError, match="caller-supplied gRPC channel"):
                channel_client.create_compaction_target("items")
        finally:
            channel_client.close()
            uri_client.close()

    assert service.creates == []


def test_describe_preserves_unknown_reason_and_absent_progress(target_transport):
    uri, service = target_transport
    service.describe_responses = [
        milvus_pb2.DescribeCompactionTargetResponse(
            status=common_pb2.Status(),
            target=milvus_pb2.CompactionTargetInfo(
                target_id=42,
                db_id=7,
                collection_id=21,
                db_name="work",
                collection_name="items",
                rewrite=milvus_pb2.RewriteTargetSpec(segment_ids=[9]),
                state=milvus_pb2.COMPACTION_TARGET_STATE_INACTIVE,
                created_at=10,
                stopped_at=20,
                stop_reason="future_stop_reason",
                total_rows=0,
            ),
        )
    ]
    client = MilvusClient(uri, dedicated=True)
    try:
        result = client.describe_compaction_target(42, timeout=2)
    finally:
        client.close()

    assert (result.target_id, result.collection_id, result.db_id) == (42, 21, 7)
    assert (result.db_name, result.collection_name, result.family) == ("work", "items", "rewrite")
    assert result.segment_ids == (9,)
    assert (result.state, result.created_at, result.stopped_at) == ("inactive", 10, 20)
    assert result.stop_reason == "future_stop_reason"
    assert result.satisfied is False
    assert (result.total_rows, result.remaining_rows, result.checked_at) == (0, None, 0)
    assert service.describes == [42]


@pytest.mark.parametrize("db_name", ["", "default"])
def test_list_compaction_targets_preserves_filter_and_server_order(target_transport, db_name):
    uri, service = target_transport
    service.list_response.targets.extend(
        [
            milvus_pb2.CompactionTargetInfo(target_id=42, created_at=10),
            milvus_pb2.CompactionTargetInfo(target_id=41, created_at=10),
        ]
    )
    client = MilvusClient(uri, db_name="work", dedicated=True)
    try:
        targets = client.list_compaction_targets(
            db_name=db_name, collection_id=21, states=["active", "inactive"], include_system=True
        )
    finally:
        client.close()

    assert [target.target_id for target in targets] == [42, 41]
    [request] = service.lists
    assert request.db_name == db_name
    assert request.WhichOneof("collection_filter") == "collection_id"
    assert request.collection_id == 21
    assert list(request.states) == [
        milvus_pb2.COMPACTION_TARGET_STATE_ACTIVE,
        milvus_pb2.COMPACTION_TARGET_STATE_INACTIVE,
    ]
    assert request.include_system


def test_drop_compaction_target_uses_target_id_and_preserves_not_found(target_transport):
    uri, service = target_transport
    client = MilvusClient(uri, dedicated=True)
    try:
        assert client.drop_compaction_target(42, timeout=2) is None
        service.drop_response.status.code = ErrorCode.COMPACTION_TARGET_NOT_FOUND
        service.drop_response.status.reason = "Compaction Target does not exist"
        with pytest.raises(MilvusException, match="does not exist") as error:
            client.drop_compaction_target(43, timeout=2)
    finally:
        client.close()

    assert error.value.code == ErrorCode.COMPACTION_TARGET_NOT_FOUND
    assert service.drops == [42, 43]


def describe_result(*, target_id=42, **kwargs):
    return milvus_pb2.DescribeCompactionTargetResponse(
        status=common_pb2.Status(),
        target=milvus_pb2.CompactionTargetInfo(target_id=target_id, **kwargs),
    )


def test_wait_polls_past_active_zero_rows(target_transport, fake_clock):
    uri, service = target_transport
    service.describe_responses = [
        describe_result(remaining_rows=0, checked_at=10),
        describe_result(
            state=milvus_pb2.COMPACTION_TARGET_STATE_INACTIVE,
            stopped_at=20,
            stop_reason="satisfied",
        ),
    ]
    client = MilvusClient(uri, dedicated=True)
    try:
        result = client.wait_for_compaction_target(42, timeout=2, poll_interval=0.01)
    finally:
        client.close()

    assert result.satisfied
    assert result.stop_reason == "satisfied"
    assert service.describes == [42, 42]
    assert not service.creates and not service.drops


@pytest.mark.asyncio
async def test_async_wait_returns_other_stop_reason(target_transport, fake_clock):
    uri, service = target_transport
    service.describe_responses = [
        describe_result(),
        describe_result(
            state=milvus_pb2.COMPACTION_TARGET_STATE_INACTIVE,
            stopped_at=20,
            stop_reason="operator_dropped",
        ),
    ]
    async with AsyncMilvusClient(uri, dedicated=True) as client:
        result = await client.wait_for_compaction_target(42, timeout=2, poll_interval=0.01)

    assert result.satisfied is False
    assert result.stop_reason == "operator_dropped"
    assert service.describes == [42, 42]
    assert not service.creates and not service.drops


@pytest.mark.asyncio
async def test_async_wait_cancellation_only_cancels_read(target_transport):
    uri, service = target_transport
    service.describe_started = asyncio.Event()
    service.describe_release = asyncio.Event()
    service.describe_responses = [describe_result()]
    async with AsyncMilvusClient(uri, dedicated=True) as client:
        task = asyncio.create_task(client.wait_for_compaction_target(42))
        await asyncio.wait_for(service.describe_started.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert service.describes == [42]
    assert not service.creates and not service.drops


@pytest.mark.asyncio
async def test_optimize_uses_literal_bytes_and_projects_saved_sizes(target_client):
    client, service = target_client
    input_size, actual_size = 2**30, 2**29
    optimize = milvus_pb2.OptimizeTargetCreateResult(
        input_target_size_bytes=input_size, actual_target_size_bytes=actual_size
    )
    service.create_response.optimize.CopyFrom(optimize)
    service.describe_responses = [
        describe_result(
            optimize=milvus_pb2.OptimizeTargetInfo(
                input_target_size_bytes=input_size, actual_target_size_bytes=actual_size
            ),
            total_rows=1,
            remaining_rows=2,
            checked_at=10,
        )
    ]

    created = await client.create_compaction_target(
        "items", db_name="default", input_target_size_bytes=input_size, timeout=2
    )
    described = await client.describe_compaction_target(created.target_id, timeout=2)
    [(request, _)] = service.creates
    assert request.WhichOneof("target") == "optimize"
    assert request.optimize.input_target_size_bytes == input_size
    assert request.db_name == "default"
    assert described.family == "optimize"
    assert described.optimize == created.optimize
    assert described.optimize.actual_target_size_bytes == actual_size
    assert described.total_rows == 1
    assert described.remaining_rows == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"collection_name": ""},
        {"db_name": 12},
        {"segment_ids": "5"},
        {"segment_ids": [-1]},
        {"segment_ids": [True]},
        {"segment_ids": [], "input_target_size_bytes": 2**30},
        {"input_target_size_bytes": 0},
        {"input_target_size_bytes": 2**63},
    ],
)
@pytest.mark.asyncio
async def test_create_rejects_invalid_shapes_before_dispatch(target_client, kwargs):
    client, service = target_client
    with pytest.raises(ParamError):
        await client.create_compaction_target(**{"collection_name": "items", **kwargs})
    assert service.creates == []


@pytest.mark.asyncio
async def test_create_rejects_partition_names(target_client):
    client, service = target_client
    with pytest.raises(TypeError, match="partition_names"):
        await client.create_compaction_target("items", partition_names=["part"])
    assert service.creates == []


@pytest.mark.parametrize(
    ("response_fields", "create_args", "error_message"),
    [
        ({"target_id": 42}, {}, "missing response status"),
        ({"status": common_pb2.Status()}, {}, "invalid successful response"),
        (
            {"status": common_pb2.Status(), "target_id": 42, "total_rows": -1},
            {},
            "invalid successful response",
        ),
        (
            {"status": common_pb2.Status(code=2302), "target_id": 42},
            {},
            "error response contains a successful result",
        ),
        (
            {"status": common_pb2.Status(), "target_id": 42},
            {"input_target_size_bytes": 1024},
            "invalid successful response",
        ),
        (
            {
                "status": common_pb2.Status(),
                "target_id": 42,
                "optimize": milvus_pb2.OptimizeTargetCreateResult(
                    input_target_size_bytes=2048, actual_target_size_bytes=1024
                ),
            },
            {"input_target_size_bytes": 1024},
            "invalid OPTIMIZE sizes",
        ),
        (
            {
                "status": common_pb2.Status(),
                "target_id": 42,
                "optimize": milvus_pb2.OptimizeTargetCreateResult(
                    input_target_size_bytes=1024, actual_target_size_bytes=1024
                ),
            },
            {},
            "invalid successful response",
        ),
    ],
)
@pytest.mark.asyncio
async def test_create_rejects_untrustworthy_response(
    target_client, response_fields, create_args, error_message
):
    client, service = target_client
    service.create_response = milvus_pb2.CreateCompactionTargetResponse(**response_fields)
    with pytest.raises(MilvusException, match=error_message) as error:
        await client.create_compaction_target("items", timeout=2, **create_args)

    assert error.value.code == ErrorCode.COMPACTION_TARGET_CREATE_OUTCOME_UNKNOWN
    assert error.value.retriable is False
    assert len(service.creates) == 1


@pytest.mark.parametrize(
    "status",
    [
        common_pb2.Status(
            code=ErrorCode.COMPACTION_TARGET_CREATE_OUTCOME_UNKNOWN,
            error_code=common_pb2.UnexpectedError,
            reason="Cannot confirm candidate Target 42",
        ),
        common_pb2.Status(code=2302, reason="Active OPTIMIZE Target 42"),
        common_pb2.Status(code=8, reason="Rate limited", retriable=True),
        common_pb2.Status(
            code=1100,
            error_code=common_pb2.IllegalArgument,
            reason="Invalid size",
            extra_info={"is_input_error": "true"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_create_preserves_business_status_without_replay(target_client, status):
    client, service = target_client
    service.create_response = milvus_pb2.CreateCompactionTargetResponse(status=status)
    with pytest.raises(MilvusException, match=status.reason) as error:
        await client.create_compaction_target("items", timeout=2)

    assert error.value.code == status.code
    assert error.value.compatible_code == status.error_code
    assert error.value.retriable == status.retriable
    assert error.value.is_input_error == (status.extra_info.get("is_input_error") == "true")
    assert len(service.creates) == 1


@pytest.mark.parametrize(
    "transport_code", [grpc.StatusCode.UNIMPLEMENTED, grpc.StatusCode.PERMISSION_DENIED]
)
@pytest.mark.asyncio
async def test_create_preserves_definitive_grpc_errors(target_client, transport_code):
    client, service = target_client
    service.create_error = transport_code
    with pytest.raises(grpc.RpcError) as error:
        await client.create_compaction_target("items", timeout=2)
    assert error.value.code() == transport_code
    assert len(service.creates) == 1


@pytest.mark.parametrize(
    "status",
    [
        common_pb2.Status(code=4, reason="Service not ready", retriable=True),
        common_pb2.Status(code=ErrorCode.RATE_LIMIT, reason="Rate limited"),
        common_pb2.Status(error_code=common_pb2.RateLimit, reason="Legacy rate limit"),
    ],
)
@pytest.mark.asyncio
async def test_target_read_preserves_zero_progress_and_retries_retriable_status(
    target_client, fake_clock, status
):
    client, service = target_client
    service.describe_responses = [
        milvus_pb2.DescribeCompactionTargetResponse(status=status),
        describe_result(remaining_rows=0, checked_at=10),
    ]
    result = await client.describe_compaction_target(42, timeout=2)

    assert result.total_rows is None
    assert result.remaining_rows == 0
    assert result.checked_at == 10
    assert service.describes == [42, 42]


@pytest.mark.asyncio
async def test_wait_preserves_not_found_and_other_terminal_read_errors(target_client):
    client, service = target_client
    service.describe_responses = [
        milvus_pb2.DescribeCompactionTargetResponse(
            status=common_pb2.Status(
                code=ErrorCode.COMPACTION_TARGET_NOT_FOUND,
                reason="Compaction Target expired",
            )
        )
    ]
    with pytest.raises(MilvusException, match="expired") as error:
        await client.wait_for_compaction_target(42, timeout=2)
    assert error.value.code == ErrorCode.COMPACTION_TARGET_NOT_FOUND
    assert service.describes == [42]
    assert not service.creates and not service.drops

    service.describe_error = grpc.StatusCode.UNIMPLEMENTED
    with pytest.raises(grpc.RpcError) as error:
        await client.wait_for_compaction_target(42, timeout=2)
    assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert service.describes == [42, 42]


@pytest.mark.parametrize(
    "response",
    [
        milvus_pb2.DescribeCompactionTargetResponse(
            target=milvus_pb2.CompactionTargetInfo(target_id=42)
        ),
        milvus_pb2.DescribeCompactionTargetResponse(status=common_pb2.Status()),
        describe_result(target_id=43),
        describe_result(remaining_rows=-1),
        describe_result(state=20),
    ],
)
@pytest.mark.asyncio
async def test_describe_rejects_invalid_response(target_client, response):
    client, service = target_client
    service.describe_responses = [response]
    with pytest.raises(MilvusException, match=r"Missing|Invalid|unexpected target"):
        await client.describe_compaction_target(42, timeout=2)
    assert service.describes == [42]


@pytest.mark.asyncio
async def test_wait_budget_caps_poll_sleep(target_client, fake_clock):
    client, service = target_client
    service.describe_responses = [describe_result(remaining_rows=0)]
    with pytest.raises(MilvusException, match="Timed out waiting for Compaction Target 42"):
        await client.wait_for_compaction_target(42, timeout=1, poll_interval=10)
    assert fake_clock() == pytest.approx(1)
    assert service.describes == [42]
    assert not service.creates and not service.drops


@pytest.mark.asyncio
async def test_wait_budget_caps_unavailable_read_retries(target_client, fake_clock):
    client, service = target_client
    service.describe_error = grpc.StatusCode.UNAVAILABLE
    with pytest.raises(grpc.RpcError) as error:
        await client.wait_for_compaction_target(42, timeout=0.2)
    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    assert fake_clock() == pytest.approx(0.2)
    assert len(service.describes) >= 2
    assert not service.creates and not service.drops


@pytest.mark.asyncio
async def test_list_preserves_a_system_target_and_absent_observations(target_client):
    client, service = target_client
    service.list_response.targets.add(
        target_id=45,
        other=milvus_pb2.OtherTargetSpec(
            condition="size",
            cluster_wide=True,
            parameters=[common_pb2.KeyValuePair(key="planner", value="two_tier")],
        ),
    )
    (target,) = await client.list_compaction_targets(include_system=True, timeout=2)
    assert (target.family, target.condition, target.cluster_wide) == ("other", "size", True)
    assert target.parameters == {"planner": "two_tier"}
    assert (target.total_rows, target.remaining_rows) == (None, None)
    assert service.describes == []

    service.list_response.ClearField("status")
    with pytest.raises(MilvusException, match="Missing Compaction Target response status"):
        await client.list_compaction_targets(include_system=True, timeout=2)


@pytest.mark.asyncio
async def test_drop_requires_status(target_client):
    client, service = target_client
    await client.drop_compaction_target(42, timeout=2)
    service.drop_response.ClearField("status")
    with pytest.raises(MilvusException, match="Missing Compaction Target response status"):
        await client.drop_compaction_target(43, timeout=2)
    assert service.drops == [42, 43]


@pytest.mark.parametrize(
    "filters",
    [
        {"db_name": 12},
        {"collection_name": ""},
        {"collection_name": "items", "collection_id": 21},
        {"collection_id": 0},
        {"states": "active"},
        {"states": ["pending"]},
        {"include_system": 1},
    ],
)
@pytest.mark.asyncio
async def test_list_validates_filters_before_dispatch(target_client, filters):
    client, service = target_client
    with pytest.raises(ParamError):
        await client.list_compaction_targets(**filters)
    assert service.lists == []


@pytest.mark.asyncio
async def test_list_accepts_collection_name(target_client):
    client, service = target_client
    await client.list_compaction_targets(collection_name="items")
    [request] = service.lists
    assert request.WhichOneof("collection_filter") == "collection_name"
    assert request.collection_name == "items"
    assert list(request.states) == []


@pytest.mark.parametrize(("name", "value"), [("timeout", -1), ("poll_interval", 0)])
@pytest.mark.asyncio
async def test_wait_validates_timeout_and_poll_interval_before_dispatch(target_client, name, value):
    client, service = target_client
    with pytest.raises(ParamError, match=name):
        await client.wait_for_compaction_target(42, **{name: value})
    assert service.describes == []


@pytest.mark.asyncio
async def test_wait_recovers_unavailable_connection_within_budget(target_client, fake_clock):
    client, service = target_client
    service.describe_error = grpc.StatusCode.UNAVAILABLE

    def connect_replacement(request, **kwargs):
        service.describe_error = None
        return milvus_pb2.ConnectResponse(status=common_pb2.Status(), identifier=456)

    service.connect = connect_replacement
    service.describe_responses = [
        describe_result(state=milvus_pb2.COMPACTION_TARGET_STATE_INACTIVE, stop_reason="satisfied"),
    ]
    result = await client.wait_for_compaction_target(42, timeout=2)

    assert result.satisfied
    assert service.describes == [42, 42]
    assert 0 < fake_clock() < 2


@pytest.mark.parametrize(
    ("method", "args", "error_attribute", "requests_attribute"),
    [
        ("list_compaction_targets", (), "list_error", "lists"),
        ("drop_compaction_target", (42,), "drop_error", "drops"),
    ],
)
@pytest.mark.asyncio
async def test_list_and_drop_recover_unavailable_connection(
    target_client, method, args, error_attribute, requests_attribute
):
    client, service = target_client
    setattr(service, error_attribute, grpc.StatusCode.UNAVAILABLE)

    def connect_replacement(request, **kwargs):
        setattr(service, error_attribute, None)
        return milvus_pb2.ConnectResponse(status=common_pb2.Status(), identifier=456)

    service.connect = connect_replacement
    await getattr(client, method)(*args)
    assert len(getattr(service, requests_attribute)) == 2


@pytest.mark.asyncio
async def test_wait_keeps_budget_when_global_topology_refresh_is_unavailable(
    target_client, fake_clock, monkeypatch
):
    client, service = target_client
    underlying = client._client if isinstance(client, SyncClientCalls) else client
    strategy = (
        AsyncGlobalStrategy() if isinstance(underlying, AsyncMilvusClient) else GlobalStrategy()
    )
    managed = underlying._manager._get_managed(underlying._handler)
    monkeypatch.setattr(managed, "strategy", strategy)
    refresh_topology = Mock(return_value=False)
    monkeypatch.setattr(strategy, "on_unavailable", refresh_topology)
    service.describe_error = grpc.StatusCode.UNAVAILABLE

    with pytest.raises(grpc.RpcError) as error:
        await client.wait_for_compaction_target(42, timeout=0.2)

    assert error.value.code() == grpc.StatusCode.UNAVAILABLE
    assert fake_clock() == pytest.approx(0.2)
    refresh_topology.assert_not_called()
