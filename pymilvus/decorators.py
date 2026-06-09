import asyncio
import contextvars
import functools
import inspect
import logging
import time
import traceback
import warnings
from typing import Any, Callable, Optional, Set

import grpc

from .exceptions import (
    DataNotMatchException,
    ErrorCode,
    MilvusException,
    ParamError,
    SchemaMismatchRetryableException,
)
from .grpc_gen import common_pb2

LOGGER = logging.getLogger(__name__)
WARNING_COLOR = "\033[93m{}\033[0m"

# gRPC connectivity state: int -> enum mapping
_CONNECTIVITY_INT_TO_ENUM = {state.value[0]: state for state in grpc.ChannelConnectivity}
_DEPRECATED_WARNING_DEPTH = contextvars.ContextVar("deprecated_warning_depth", default=0)


def _get_rpc_error_info(e: grpc.RpcError, channel: grpc.Channel = None) -> str:
    """Extract full error info from gRPC error including debug_error_string and channel state.

    Args:
        e: The gRPC RpcError exception
        channel: Optional gRPC channel to get connectivity state from
    """
    parts = [f"{e.code()}", e.details() or ""]

    # Include channel connectivity state for better diagnostics
    # This helps distinguish between connection-level vs application-level timeouts
    if channel is not None:
        try:
            state = channel._channel.check_connectivity_state(False)
            # Handle both enum and integer return types
            if isinstance(state, int):
                state = _CONNECTIVITY_INT_TO_ENUM.get(state)
            state_name = state.name if state else str(state)
            parts.append(f"channel_state={state_name}")
        except Exception:  # noqa: S110
            pass

    # Append debug_error_string for TCP-level diagnostics
    if hasattr(e, "debug_error_string"):
        try:
            debug_str = e.debug_error_string()
            if debug_str:
                parts.append(f"debug={debug_str}")
        except Exception:  # noqa: S110
            pass

    return ", ".join(p for p in parts if p)


def _try_get_channel(args: tuple) -> grpc.Channel:
    """Try to get channel from the first argument (self) if it's a GrpcHandler."""
    if args and hasattr(args[0], "_channel"):
        return args[0]._channel
    return None


def retry_on_schema_mismatch():
    """
    Decorator that handles schema mismatch errors with automatic retry.

    Catches:
    - DataNotMatchException: Client-side schema validation failed
    - SchemaMismatchRetryableException: Server returned SchemaMismatch error

    On catch:
    1. Invalidates schema cache for the collection
    2. Retries the operation once

    Usage:
        @retry_on_rpc_failure()
        @retry_on_schema_mismatch()
        def insert_rows(self, collection_name, ...):
            ...
    """

    def wrapper(func: Callable):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_handler(self: Any, collection_name: str, *args, **kwargs):
                for attempt in range(2):  # max 2 attempts
                    try:
                        return await func(self, collection_name, *args, **kwargs)
                    except (DataNotMatchException, SchemaMismatchRetryableException) as e:
                        if attempt == 0:
                            LOGGER.debug(
                                f"[{func.__name__}] Schema mismatch detected, "
                                f"invalidating cache and retrying: {e}"
                            )
                            context = kwargs.get("context")
                            if not context:
                                raise ParamError(
                                    message="context is required but not provided"
                                ) from None
                            db_name = context.get_db_name()
                            self._invalidate_schema(collection_name, db_name=db_name)
                            continue  # retry once
                        raise e from e
                return None  # unreachable, for type checker

            return async_handler

        @functools.wraps(func)
        def handler(self: Any, collection_name: str, *args, **kwargs):
            for attempt in range(2):  # max 2 attempts
                try:
                    return func(self, collection_name, *args, **kwargs)
                except (DataNotMatchException, SchemaMismatchRetryableException) as e:
                    if attempt == 0:
                        LOGGER.debug(
                            f"[{func.__name__}] Schema mismatch detected, "
                            f"invalidating cache and retrying: {e}"
                        )
                        context = kwargs.get("context")
                        if not context:
                            raise ParamError(
                                message="context is required but not provided"
                            ) from None
                        db_name = context.get_db_name()
                        self._invalidate_schema(collection_name, db_name=db_name)
                        continue  # retry once
                    raise e from e
            return None  # unreachable, for type checker

        return handler

    return wrapper


class PyMilvusDeprecationWarning(FutureWarning):
    """Warning category for PyMilvus APIs with a scheduled removal."""


def warn_deprecated(
    api_name: str,
    replacement: str = "MilvusClient",
    stacklevel: int = 2,
    reason: str = "is an ORM-style PyMilvus API",
) -> None:
    warnings.warn(
        f"`{api_name}` {reason} and will be removed in PyMilvus 3.1. "
        f"Use `{replacement}` instead.",
        category=PyMilvusDeprecationWarning,
        stacklevel=stacklevel,
    )


def _deprecated_warning(
    api_name: Optional[str],
    replacement: str = "MilvusClient",
    reason: str = "is an ORM-style PyMilvus API",
) -> Callable[[Any], Any]:
    def decorator(func: Any):
        resolved_api_name = api_name or func.__qualname__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                depth = _DEPRECATED_WARNING_DEPTH.get()
                if depth == 0:
                    warn_deprecated(
                        resolved_api_name,
                        replacement=replacement,
                        reason=reason,
                        stacklevel=3,
                    )
                token = _DEPRECATED_WARNING_DEPTH.set(depth + 1)
                try:
                    return await func(*args, **kwargs)
                finally:
                    _DEPRECATED_WARNING_DEPTH.reset(token)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            depth = _DEPRECATED_WARNING_DEPTH.get()
            if depth == 0:
                warn_deprecated(
                    resolved_api_name,
                    replacement=replacement,
                    reason=reason,
                    stacklevel=3,
                )
            token = _DEPRECATED_WARNING_DEPTH.set(depth + 1)
            try:
                return func(*args, **kwargs)
            finally:
                _DEPRECATED_WARNING_DEPTH.reset(token)

        return wrapper

    return decorator


def deprecated(
    api_name: Optional[str] = None,
    *,
    replacement: str = "MilvusClient",
    reason: str = "is an ORM-style PyMilvus API",
) -> Callable[[Any], Any]:
    if callable(api_name):
        return _deprecated_warning(None, replacement=replacement, reason=reason)(api_name)
    return _deprecated_warning(api_name, replacement=replacement, reason=reason)


def _deprecated_property(api_name: str, prop: property) -> property:
    fget = deprecated(api_name)(prop.fget) if prop.fget is not None else None
    fset = deprecated(api_name)(prop.fset) if prop.fset is not None else None
    fdel = deprecated(api_name)(prop.fdel) if prop.fdel is not None else None
    return property(fget, fset, fdel, prop.__doc__)


def deprecated_class(
    api_prefix: str,
    *,
    exclude_methods: Optional[Set[str]] = None,
    warn_properties: Optional[Set[str]] = None,
) -> Callable[[Any], Any]:
    exclude_methods = exclude_methods or set()
    warn_properties = warn_properties or set()

    def decorator(cls: Any):
        if hasattr(cls, "__init__") and "__init__" not in exclude_methods:
            cls.__init__ = deprecated(api_prefix)(cls.__init__)

        for name, attr in list(cls.__dict__.items()):
            if name.startswith("_") or name in exclude_methods:
                continue

            api_name = f"{api_prefix}.{name}"

            if isinstance(attr, property):
                if name in warn_properties:
                    setattr(cls, name, _deprecated_property(api_name, attr))
                continue

            if isinstance(attr, classmethod):
                setattr(cls, name, classmethod(deprecated(api_name)(attr.__func__)))
                continue

            if isinstance(attr, staticmethod):
                setattr(cls, name, staticmethod(deprecated(api_name)(attr.__func__)))
                continue

            if callable(attr):
                setattr(cls, name, deprecated(api_name)(attr))

        return cls

    return decorator


# Reference: https://grpc.github.io/grpc/python/grpc.html#grpc-status-code
IGNORE_RETRY_CODES = (
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.PERMISSION_DENIED,
    grpc.StatusCode.UNAUTHENTICATED,
    grpc.StatusCode.INVALID_ARGUMENT,
    grpc.StatusCode.ALREADY_EXISTS,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.UNIMPLEMENTED,
)


def retry_on_rpc_failure(
    *,
    retry_times: int = 75,
    initial_back_off: float = 0.01,
    max_back_off: float = 3,
    back_off_multiplier: int = 3,
):
    def wrapper(func: Any):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            @error_handler(func_name=func.__name__)
            @tracing_request()
            async def async_handler(*args, **kwargs):
                _timeout = kwargs.get("timeout")
                _retry_times = kwargs.get("retry_times")
                _retry_on_rate_limit = kwargs.get("retry_on_rate_limit", True)

                retry_timeout = (
                    _timeout
                    if _timeout is not None and isinstance(_timeout, (int, float))
                    else None
                )
                final_retry_times = (
                    _retry_times
                    if _retry_times is not None and isinstance(_retry_times, int)
                    else retry_times
                )
                counter = 1
                back_off = initial_back_off
                start_time = time.time()
                channel = _try_get_channel(args)

                def is_timeout(start_time: Optional[float] = None) -> bool:
                    if retry_timeout is not None:
                        return time.time() - start_time >= retry_timeout
                    return counter > final_retry_times

                to_msg = (
                    f"[{func.__name__}] Retry timeout: {retry_timeout}s"
                    if retry_timeout is not None
                    else f"[{func.__name__}] Retry run out of {final_retry_times} retry times"
                )

                while True:
                    try:
                        if retry_timeout is not None:
                            remaining = retry_timeout - (time.time() - start_time)
                            if remaining <= 0:
                                raise MilvusException(message=to_msg)
                            return await asyncio.wait_for(func(*args, **kwargs), timeout=remaining)
                        return await func(*args, **kwargs)
                    except grpc.RpcError as e:
                        if e.code() in IGNORE_RETRY_CODES:
                            raise e from e
                        # Notify connection manager of retryable RPC errors for recovery
                        if args and hasattr(args[0], "_on_rpc_error"):
                            await args[0]._on_rpc_error(e)
                        if is_timeout(start_time):
                            raise MilvusException(
                                e.code(), f"{to_msg}, {_get_rpc_error_info(e, channel)}"
                            ) from e

                        if counter > 3:
                            LOGGER.info(
                                f"[{func.__name__}] retry:{counter}, cost: {back_off:.2f}s, "
                                f"reason: <{_get_rpc_error_info(e, channel)}>"
                            )

                        await asyncio.sleep(back_off)
                        back_off = min(back_off * back_off_multiplier, max_back_off)
                    except MilvusException as e:
                        if is_timeout(start_time):
                            LOGGER.warning(WARNING_COLOR.format(to_msg))
                            raise MilvusException(
                                code=e.code, message=f"{to_msg}, message={e.message}"
                            ) from e
                        # Retry on rate-limit, or on errors handled by
                        # connection manager (e.g. REPLICATE_VIOLATION)
                        should_retry = _retry_on_rate_limit and (
                            e.code == ErrorCode.RATE_LIMIT
                            or e.compatible_code == common_pb2.RateLimit
                        )
                        if not should_retry and args and hasattr(args[0], "_on_rpc_error"):
                            should_retry = await args[0]._on_rpc_error(e)
                        if should_retry:
                            await asyncio.sleep(back_off)
                            back_off = min(back_off * back_off_multiplier, max_back_off)
                        else:
                            raise e from e
                    except asyncio.TimeoutError as e:
                        raise MilvusException(message=to_msg) from e
                    except Exception as e:
                        raise e from e
                    finally:
                        counter += 1

            return async_handler

        @functools.wraps(func)
        @error_handler(func_name=func.__name__)
        @tracing_request()
        def handler(*args, **kwargs):
            # This has to make sure every timeout parameter is passing
            # throught kwargs form as `timeout=10`
            _timeout = kwargs.get("timeout")
            _retry_times = kwargs.get("retry_times")
            _retry_on_rate_limit = kwargs.get("retry_on_rate_limit", True)

            retry_timeout = (
                _timeout if _timeout is not None and isinstance(_timeout, (int, float)) else None
            )
            final_retry_times = (
                _retry_times
                if _retry_times is not None and isinstance(_retry_times, int)
                else retry_times
            )
            counter = 1
            back_off = initial_back_off
            start_time = time.time()
            channel = _try_get_channel(args)

            def timeout(start_time: Optional[float] = None) -> bool:
                """If timeout is valid, use timeout as the retry limits,
                If timeout is None, use final_retry_times as the retry limits.
                """
                if retry_timeout is not None:
                    return time.time() - start_time >= retry_timeout
                return counter > final_retry_times

            to_msg = (
                f"[{func.__name__}] Retry timeout: {retry_timeout}s"
                if retry_timeout is not None
                else f"[{func.__name__}] Retry run out of {final_retry_times} retry times"
            )

            while True:
                try:
                    return func(*args, **kwargs)
                except grpc.RpcError as e:
                    if e.code() in IGNORE_RETRY_CODES:
                        raise e from e
                    # Notify connection manager of retryable RPC errors for recovery
                    if args and hasattr(args[0], "_on_rpc_error"):
                        args[0]._on_rpc_error(e)
                    if timeout(start_time):
                        raise MilvusException(
                            e.code(), f"{to_msg}, {_get_rpc_error_info(e, channel)}"
                        ) from e

                    if counter > 3:
                        LOGGER.info(
                            f"[{func.__name__}] retry:{counter}, cost: {back_off:.2f}s, "
                            f"reason: <{_get_rpc_error_info(e, channel)}>"
                        )

                    time.sleep(back_off)
                    back_off = min(back_off * back_off_multiplier, max_back_off)
                except MilvusException as e:
                    if timeout(start_time):
                        LOGGER.warning(WARNING_COLOR.format(to_msg))
                        raise MilvusException(
                            code=e.code, message=f"{to_msg}, message={e.message}"
                        ) from e
                    # Retry on rate-limit, or on errors handled by
                    # connection manager (e.g. REPLICATE_VIOLATION)
                    should_retry = _retry_on_rate_limit and (
                        e.code == ErrorCode.RATE_LIMIT or e.compatible_code == common_pb2.RateLimit
                    )
                    if not should_retry and args and hasattr(args[0], "_on_rpc_error"):
                        should_retry = args[0]._on_rpc_error(e)
                    if should_retry:
                        time.sleep(back_off)
                        back_off = min(back_off * back_off_multiplier, max_back_off)
                    else:
                        raise e from e
                except Exception as e:
                    raise e from e
                finally:
                    counter += 1

        return handler

    return wrapper


def _log_rpc_error(inner_name: str, label: str, msg: str, start_ts: float):
    if not LOGGER.isEnabledFor(logging.ERROR):
        return
    elapsed_ms = (time.monotonic() - start_ts) * 1000
    tb_str = traceback.format_exc()
    LOGGER.error(
        f"{label}: [{inner_name}], {msg}, <elapsed:{elapsed_ms:.1f}ms>\nTraceback:\n{tb_str}"
    )


def error_handler(func_name: str = ""):
    def wrapper(func: Callable):
        if inspect.iscoroutinefunction(func):
            inner_name = func_name or func.__name__

            @functools.wraps(func)
            async def async_handler(*args, **kwargs):
                start_ts = time.monotonic()
                try:
                    return await func(*args, **kwargs)
                except MilvusException as e:
                    _log_rpc_error(inner_name, "RPC error", str(e), start_ts)
                    raise e from e
                except grpc.FutureTimeoutError as e:
                    _log_rpc_error(
                        inner_name,
                        "grpc Timeout",
                        f"<{e.__class__.__name__}: {e.code()}, {e.details()}>",
                        start_ts,
                    )
                    raise e from e
                except grpc.RpcError as e:
                    _log_rpc_error(
                        inner_name,
                        "grpc RpcError",
                        f"<{e.__class__.__name__}: {e.code()}, {e.details()}>",
                        start_ts,
                    )
                    raise e from e
                except Exception as e:
                    _log_rpc_error(inner_name, "Unexpected error", str(e), start_ts)
                    raise MilvusException(message=f"Unexpected error, message=<{e!s}>") from e

            return async_handler

        inner_name = func_name or func.__name__

        @functools.wraps(func)
        def handler(*args, **kwargs):
            start_ts = time.monotonic()
            try:
                return func(*args, **kwargs)
            except MilvusException as e:
                _log_rpc_error(inner_name, "RPC error", str(e), start_ts)
                raise e from e
            except grpc.FutureTimeoutError as e:
                _log_rpc_error(
                    inner_name,
                    "grpc Timeout",
                    f"<{e.__class__.__name__}: {e.code()}, {e.details()}>",
                    start_ts,
                )
                raise e from e
            except grpc.RpcError as e:
                _log_rpc_error(
                    inner_name,
                    "grpc RpcError",
                    f"<{e.__class__.__name__}: {e.code()}, {e.details()}>",
                    start_ts,
                )
                raise e from e
            except Exception as e:
                _log_rpc_error(inner_name, "Unexpected error", str(e), start_ts)
                raise MilvusException(message=f"Unexpected error, message=<{e!s}>") from e

        return handler

    return wrapper


def tracing_request():
    def wrapper(func: Callable):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_handler(self: Callable, *args, **kwargs):
                level = kwargs.get("log-level", kwargs.get("log_level"))
                if level:
                    self.set_onetime_loglevel(level)
                return await func(self, *args, **kwargs)

            return async_handler

        @functools.wraps(func)
        def handler(self: Callable, *args, **kwargs):
            level = kwargs.get("log-level", kwargs.get("log_level"))
            if level:
                self.set_onetime_loglevel(level)
            return func(self, *args, **kwargs)

        return handler

    return wrapper


def ignore_unimplemented(default_return_value: Any):
    def wrapper(func: Callable):
        @functools.wraps(func)
        def handler(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    LOGGER.debug(f"{func.__name__} unimplemented, ignore it")
                    return default_return_value
                raise e from e
            except Exception as e:
                raise e from e

        return handler

    return wrapper


def upgrade_reminder(func: Callable):
    @functools.wraps(func)
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                msg = (
                    "Received UNIMPLEMENTED from server. "
                    "Please first verify that your uri points to a Milvus gRPC port "
                    "(default: 19530) instead of an HTTP/proxy port. "
                    "If the port is correct, your SDK may be incompatible with the server; "
                    "upgrade Milvus server or use a matching pymilvus version."
                )
                raise MilvusException(message=msg) from e
            raise e from e
        except Exception as e:
            raise e from e

    return handler
