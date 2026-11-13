"""Request and response handling for the Compaction Target APIs."""

import math
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import grpc

from pymilvus.decorators import IGNORE_RETRY_CODES
from pymilvus.exceptions import ErrorCode, MilvusException, ParamError
from pymilvus.grpc_gen import common_pb2
from pymilvus.grpc_gen import milvus_pb2 as milvus_types

from .constants import INT64_MAX
from .utils import is_successful


@dataclass(frozen=True)
class OptimizeTargetSize:
    """Requested and effective output segment sizes in bytes."""

    input_target_size_bytes: int
    actual_target_size_bytes: int


@dataclass(frozen=True)
class CreateCompactionTargetResult:
    """The Target ID, initial matching row count and optional OPTIMIZE sizes."""

    target_id: int
    total_rows: int
    optimize: Optional[OptimizeTargetSize] = None


CompactionTargetState = Literal["active", "inactive"]
_TARGET_STATES = {
    "active": milvus_types.COMPACTION_TARGET_STATE_ACTIVE,
    "inactive": milvus_types.COMPACTION_TARGET_STATE_INACTIVE,
}
_TARGET_STATE_NAMES = {state: name for name, state in _TARGET_STATES.items()}


@dataclass
class CompactionTargetInfo:
    """Persisted target details and an optional observation of matching rows."""

    target_id: int
    db_id: int
    collection_id: int
    db_name: str
    collection_name: str
    family: str
    state: CompactionTargetState
    created_at: int
    stopped_at: int
    stop_reason: str
    total_rows: Optional[int]
    remaining_rows: Optional[int]
    checked_at: int
    message: str
    segment_ids: Tuple[int, ...] = ()
    optimize: Optional[OptimizeTargetSize] = None
    condition: Optional[str] = None
    cluster_wide: bool = False
    parameters: Optional[Dict[str, str]] = None

    @property
    def satisfied(self) -> bool:
        """Whether the target stopped because its condition was satisfied."""
        return self.state == "inactive" and self.stop_reason == "satisfied"


def positive_int64(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= INT64_MAX:
        raise ParamError(message=f"{name} must be a positive signed 64-bit integer")


def create_target_request(
    collection_name: str,
    db_name: str,
    segment_ids: Optional[Sequence[int]],
    input_target_size_bytes: Optional[int],
) -> milvus_types.CreateCompactionTargetRequest:
    if not isinstance(collection_name, str) or not collection_name:
        raise ParamError(message="collection_name must be a non-empty string")
    if not isinstance(db_name, str):
        raise ParamError(message="db_name must be a string")

    request = milvus_types.CreateCompactionTargetRequest(
        collection_name=collection_name, db_name=db_name
    )
    if input_target_size_bytes is not None:
        if segment_ids is not None:
            raise ParamError(message="OPTIMIZE does not accept segment_ids")
        positive_int64(input_target_size_bytes, "input_target_size_bytes")
        request.optimize.input_target_size_bytes = input_target_size_bytes
    else:
        if segment_ids is not None and (
            not isinstance(segment_ids, Sequence) or isinstance(segment_ids, (str, bytes))
        ):
            raise ParamError(message="segment_ids must be a sequence of positive integers")
        roots = list(segment_ids) if segment_ids is not None else []
        for root in roots:
            positive_int64(root, "segment_ids")
        request.rewrite.CopyFrom(milvus_types.RewriteTargetSpec(segment_ids=roots))
    return request


def require_retry_configurable_channel(caller_channel: bool) -> None:
    if caller_channel:
        raise ParamError(
            message="CreateCompactionTarget cannot safely use a caller-supplied gRPC channel"
        )


def create_outcome_unknown(*, details: str) -> MilvusException:
    return MilvusException(
        code=ErrorCode.COMPACTION_TARGET_CREATE_OUTCOME_UNKNOWN,
        message=f"Compaction Target creation outcome is unknown: {details}",
    )


def create_transport_outcome_unknown(error: grpc.RpcError) -> bool:
    # These codes identify explicit rejections before a Target can be saved.
    return error.code() not in {
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.INVALID_ARGUMENT,
    }


def parse_create_target_result(
    request: milvus_types.CreateCompactionTargetRequest,
    response: Optional[milvus_types.CreateCompactionTargetResponse],
) -> CreateCompactionTargetResult:
    if response is None or not response.HasField("status"):
        raise create_outcome_unknown(details="missing response status")

    has_optimize_result = response.HasField("optimize")
    if not is_successful(response.status):
        if response.target_id or response.total_rows or has_optimize_result:
            raise create_outcome_unknown(details="error response contains a successful result")
        raise MilvusException.from_status(response.status)

    is_optimize = request.WhichOneof("target") == "optimize"
    if response.target_id <= 0 or response.total_rows < 0 or is_optimize != has_optimize_result:
        raise create_outcome_unknown(details="invalid successful response")

    optimize = None
    if is_optimize:
        sizes = response.optimize
        if (
            sizes.input_target_size_bytes != request.optimize.input_target_size_bytes
            or sizes.actual_target_size_bytes <= 0
        ):
            raise create_outcome_unknown(details="invalid OPTIMIZE sizes")
        optimize = OptimizeTargetSize(sizes.input_target_size_bytes, sizes.actual_target_size_bytes)
    return CreateCompactionTargetResult(response.target_id, response.total_rows, optimize)


def list_targets_request(
    db_name: str,
    collection_name: Optional[str],
    collection_id: Optional[int],
    states: Sequence[CompactionTargetState],
    include_system: bool,
) -> milvus_types.ListCompactionTargetsRequest:
    if not isinstance(db_name, str):
        raise ParamError(message="db_name must be a string")
    if collection_name is not None and (
        not isinstance(collection_name, str) or not collection_name
    ):
        raise ParamError(message="collection_name must be a non-empty string")
    if collection_id is not None:
        if collection_name is not None:
            raise ParamError(message="Specify only one of collection_name and collection_id")
        positive_int64(collection_id, "collection_id")
    if not isinstance(include_system, bool):
        raise ParamError(message="include_system must be a boolean")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise ParamError(message="states must be a sequence containing active or inactive")
    try:
        state_values = [_TARGET_STATES[state] for state in states]
    except (KeyError, TypeError) as error:
        raise ParamError(message="states must contain only active or inactive") from error
    return milvus_types.ListCompactionTargetsRequest(
        db_name=db_name,
        collection_name=collection_name,
        collection_id=collection_id,
        states=state_values,
        include_system=include_system,
    )


def check_target_response(
    response: Union[
        milvus_types.DescribeCompactionTargetResponse,
        milvus_types.ListCompactionTargetsResponse,
        milvus_types.DropCompactionTargetResponse,
    ],
) -> None:
    if response is None or not response.HasField("status"):
        raise MilvusException(message="Missing Compaction Target response status")
    if not is_successful(response.status):
        raise MilvusException.from_status(response.status)


def parse_target_info(target: milvus_types.CompactionTargetInfo) -> CompactionTargetInfo:
    state = _TARGET_STATE_NAMES.get(target.state)
    total_rows = target.total_rows if target.HasField("total_rows") else None
    remaining_rows = target.remaining_rows if target.HasField("remaining_rows") else None
    if target.target_id <= 0 or state is None:
        raise MilvusException(message="Invalid Compaction Target response")
    if any(rows is not None and rows < 0 for rows in (total_rows, remaining_rows)):
        raise MilvusException(message="Invalid Compaction Target row count")

    family = target.WhichOneof("target") or "unknown"
    result = CompactionTargetInfo(
        target_id=target.target_id,
        db_id=target.db_id,
        collection_id=target.collection_id,
        db_name=target.db_name,
        collection_name=target.collection_name,
        family=family,
        state=state,
        created_at=target.created_at,
        stopped_at=target.stopped_at,
        stop_reason=target.stop_reason,
        total_rows=total_rows,
        remaining_rows=remaining_rows,
        checked_at=target.checked_at,
        message=target.message,
    )
    if family == "rewrite":
        result.segment_ids = tuple(target.rewrite.segment_ids)
    elif family == "optimize":
        result.optimize = OptimizeTargetSize(
            target.optimize.input_target_size_bytes, target.optimize.actual_target_size_bytes
        )
    elif family == "other":
        result.condition = target.other.condition
        result.cluster_wide = target.other.cluster_wide
        result.parameters = {pair.key: pair.value for pair in target.other.parameters}
    return result


def parse_describe_target_result(
    target_id: int, response: milvus_types.DescribeCompactionTargetResponse
) -> CompactionTargetInfo:
    check_target_response(response)
    if not response.HasField("target") or response.target.target_id != target_id:
        raise MilvusException(message="DescribeCompactionTarget returned an unexpected target")
    return parse_target_info(response.target)


def retryable_target_read(error: Union[grpc.RpcError, MilvusException]) -> bool:
    if isinstance(error, grpc.RpcError):
        return error.code() not in (*IGNORE_RETRY_CODES, grpc.StatusCode.CANCELLED)
    return not error.is_input_error and (
        error.retriable
        or error.code == ErrorCode.RATE_LIMIT
        or error.compatible_code == common_pb2.RateLimit
    )


class TargetReadRetry:
    """Apply the RPC timeout to all Describe attempts and backoff together."""

    def __init__(self, timeout: Optional[float]) -> None:
        self._deadline = time.monotonic() + timeout if timeout is not None else None
        self._backoff = 0.01
        self._retries_left = 75

    def remaining(self) -> Optional[float]:
        return None if self._deadline is None else max(0, self._deadline - time.monotonic())

    def can_retry(self) -> bool:
        return self.remaining() != 0 and self._retries_left > 0

    def delay_after(
        self, error: Union[grpc.RpcError, MilvusException], *, recovered: bool
    ) -> Optional[float]:
        remaining = self.remaining()
        if (
            remaining == 0
            or (remaining is None and self._retries_left == 0)
            or not (recovered or retryable_target_read(error))
        ):
            return None
        if remaining is None:
            self._retries_left -= 1
        delay = self._backoff if remaining is None else min(remaining, self._backoff)
        self._backoff = min(self._backoff * 3, 3.0)
        return delay


def compaction_target_deadline(timeout: Optional[float], poll_interval: float) -> Optional[float]:
    if (
        not isinstance(poll_interval, (int, float))
        or not (0 < poll_interval < math.inf)
        or isinstance(poll_interval, bool)
    ):
        raise ParamError(message="poll_interval must be a finite positive number")
    if timeout is None:
        return None
    if (
        not isinstance(timeout, (int, float))
        or not (0 <= timeout < math.inf)
        or isinstance(timeout, bool)
    ):
        raise ParamError(message="timeout must be a finite non-negative number")
    return time.monotonic() + timeout


def compaction_target_wait_timeout(target_id: int) -> MilvusException:
    return MilvusException(message=f"Timed out waiting for Compaction Target {target_id}")


def compaction_target_time_left(deadline: Optional[float], target_id: int) -> Optional[float]:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise compaction_target_wait_timeout(target_id)
    return remaining
