import asyncio
import datetime
import functools
import logging
import time
import traceback
from typing import Any, Callable, Optional

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
        if asyncio.iscoroutinefunction(func):

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


def deprecated(func: Any):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        LOGGER.warning(
            WARNING_COLOR.format(
                "[WARNING] PyMilvus: ",
                "class Milvus will be deprecated soon, please use Collection/utility instead",
            )
        )
        return func(*args, **kwargs)

    return inner


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
        if asyncio.iscoroutinefunction(func):

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
                        return await func(*args, **kwargs)
                    except grpc.RpcError as e:
                        if e.code() in IGNORE_RETRY_CODES:
                            raise e from e
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
                        if _retry_on_rate_limit and (
                            e.code == ErrorCode.RATE_LIMIT
                            or e.compatible_code == common_pb2.RateLimit
                        ):
                            await asyncio.sleep(back_off)
                            back_off = min(back_off * back_off_multiplier, max_back_off)
                        else:
                            raise e from e
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
                    # Do not retry on these codes
                    if e.code() in IGNORE_RETRY_CODES:
                        raise e from e
                    if timeout(start_time):
                        raise MilvusException(
                            e.code, f"{to_msg}, {_get_rpc_error_info(e, channel)}"
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
                    if _retry_on_rate_limit and (
                        e.code == ErrorCode.RATE_LIMIT or e.compatible_code == common_pb2.RateLimit
                    ):
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


def error_handler(func_name: str = ""):
    def wrapper(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_handler(*args, **kwargs):
                inner_name = func_name or func.__name__
                record_dict = {}
                try:
                    record_dict["RPC start"] = str(datetime.datetime.now())
                    return await func(*args, **kwargs)
                except MilvusException as e:
                    record_dict["RPC error"] = str(datetime.datetime.now())
                    tb_str = traceback.format_exc()
                    LOGGER.error(
                        f"RPC error: [{inner_name}], {e}, <Time:{record_dict}>\n"
                        f"Traceback:\n{tb_str}"
                    )
                    raise e from e
                except grpc.FutureTimeoutError as e:
                    record_dict["gRPC timeout"] = str(datetime.datetime.now())
                    tb_str = traceback.format_exc()
                    LOGGER.error(
                        f"grpc Timeout: [{inner_name}], <{e.__class__.__name__}: "
                        f"{e.code()}, {e.details()}>, <Time:{record_dict}>\n"
                        f"Traceback:\n{tb_str}"
                    )
                    raise e from e
                except grpc.RpcError as e:
                    record_dict["gRPC error"] = str(datetime.datetime.now())
                    tb_str = traceback.format_exc()
                    LOGGER.error(
                        f"grpc RpcError: [{inner_name}], <{e.__class__.__name__}: "
                        f"{e.code()}, {e.details()}>, <Time:{record_dict}>\n"
                        f"Traceback:\n{tb_str}"
                    )
                    raise e from e
                except Exception as e:
                    record_dict["Exception"] = str(datetime.datetime.now())
                    tb_str = traceback.format_exc()
                    LOGGER.error(
                        f"Unexpected error: [{inner_name}], {e}, <Time: {record_dict}>\n"
                        f"Traceback:\n{tb_str}"
                    )
                    raise MilvusException(message=f"Unexpected error, message=<{e!s}>") from e

            return async_handler

        @functools.wraps(func)
        def handler(*args, **kwargs):
            inner_name = func_name
            if inner_name == "":
                inner_name = func.__name__
            record_dict = {}
            try:
                record_dict["RPC start"] = str(datetime.datetime.now())
                return func(*args, **kwargs)
            except MilvusException as e:
                record_dict["RPC error"] = str(datetime.datetime.now())
                tb_str = traceback.format_exc()
                LOGGER.error(
                    f"RPC error: [{inner_name}], {e}, <Time:{record_dict}>\n"
                    f"Traceback:\n{tb_str}"
                )
                raise e from e
            except grpc.FutureTimeoutError as e:
                record_dict["gRPC timeout"] = str(datetime.datetime.now())
                tb_str = traceback.format_exc()
                LOGGER.error(
                    f"grpc Timeout: [{inner_name}], <{e.__class__.__name__}: "
                    f"{e.code()}, {e.details()}>, <Time:{record_dict}>\n"
                    f"Traceback:\n{tb_str}"
                )
                raise e from e
            except grpc.RpcError as e:
                record_dict["gRPC error"] = str(datetime.datetime.now())
                tb_str = traceback.format_exc()
                LOGGER.error(
                    f"grpc RpcError: [{inner_name}], <{e.__class__.__name__}: "
                    f"{e.code()}, {e.details()}>, <Time:{record_dict}>\n"
                    f"Traceback:\n{tb_str}"
                )
                raise e from e
            except Exception as e:
                record_dict["Exception"] = str(datetime.datetime.now())
                tb_str = traceback.format_exc()
                LOGGER.error(
                    f"Unexpected error: [{inner_name}], {e}, <Time: {record_dict}>\n"
                    f"Traceback:\n{tb_str}"
                )
                raise MilvusException(message=f"Unexpected error, message=<{e!s}>") from e

        return handler

    return wrapper


def tracing_request():
    def wrapper(func: Callable):
        if asyncio.iscoroutinefunction(func):

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
                    "Incorrect port or sdk is incompatible with server, "
                    "please check your port or downgrade your sdk or upgrade your server"
                )
                raise MilvusException(message=msg) from e
            raise e from e
        except Exception as e:
            raise e from e

    return handler
