import time
import datetime
import logging
import functools

import grpc

from .exceptions import MilvusException, MilvusUnavaliableException
from .client.types import Status

LOGGER = logging.getLogger(__name__)
WARNING_COLOR = "\033[93m{}\033[0m"


def deprecated(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        dup_msg = "[WARNING] PyMilvus: class Milvus will be deprecated soon, please use Collection/utility instead"
        LOGGER.warning(WARNING_COLOR.format(dup_msg))
        return func(*args, **kwargs)
    return inner


def retry_on_rpc_failure(retry_times=10, initial_back_off=0.01, max_back_off=60, back_off_multiplier=3, retry_on_deadline=True):
    # the default 7 retry_times will cost about 26s
    def wrapper(func):
        @functools.wraps(func)
        @error_handler(func_name=func.__name__)
        def handler(self, *args, **kwargs):
            # This has to make sure every timeout parameter is passing throught kwargs form as `timeout=10`
            _timeout = kwargs.get("timeout", None)

            retry_timeout = _timeout if _timeout is not None and isinstance(_timeout, int) else None
            counter = 1
            back_off = initial_back_off
            start_time = time.time()

            def timeout(start_time) -> bool:
                """ If timeout is valid, use timeout as the retry limits,
                    If timeout is None, use retry_times as the retry limits.
                """
                if retry_timeout is not None:
                    return time.time() - start_time >= retry_timeout
                return counter > retry_times

            while True:
                try:
                    return func(self, *args, **kwargs)
                except grpc.RpcError as e:
                    # DEADLINE_EXCEEDED means that the task wat not completed
                    # UNAVAILABLE means that the service is not reachable currently
                    # Reference: https://grpc.github.io/grpc/python/grpc.html#grpc-status-code
                    if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED and e.code() != grpc.StatusCode.UNAVAILABLE:
                        raise MilvusException(Status.UNEXPECTED_ERROR, str(e))
                    if not retry_on_deadline and e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        raise MilvusException(Status.UNEXPECTED_ERROR, str(e))
                    if timeout(start_time):
                        timeout_msg = f"Retry timeout: {retry_timeout}s" if retry_timeout is not None \
                            else f"Retry run out of {retry_times} retry times"

                        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                            raise MilvusException(Status.UNEXPECTED_ERROR, f"rpc deadline exceeded: {timeout_msg}")
                        if e.code() == grpc.StatusCode.UNAVAILABLE:
                            raise MilvusUnavaliableException(Status.UNEXPECTED_ERROR, f"server unavaliable: {timeout_msg}")
                        raise MilvusException(Status.UNEXPECTED_ERROR, str(e))

                    if counter > 3:
                        retry_msg = f"[{func.__name__}] retry:{counter}, cost: {back_off}s, reason: <{e.__class__.__name__}: {e.code()}, {e.details()}>"
                        LOGGER.warning(WARNING_COLOR.format(retry_msg))

                    time.sleep(back_off)
                    back_off = min(back_off * back_off_multiplier, max_back_off)
                except Exception as e:
                    raise e
                finally:
                    counter += 1

        return handler
    return wrapper


def error_handler(func_name=""):
    def wrapper(func):
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
                LOGGER.error(f"RPC error: [{inner_name}], {e}, <Time:{record_dict}>")
                raise e
            except grpc.FutureTimeoutError as e:
                record_dict["gRPC timeout"] = str(datetime.datetime.now())
                LOGGER.error(f"grpc Timeout: [{inner_name}], <{e.__class__.__name__}: {e.code()}, {e.details()}>, <Time:{record_dict}>")
                raise e
            except grpc.RpcError as e:
                record_dict["gRPC error"] = str(datetime.datetime.now())
                LOGGER.error(f"grpc RpcError: [{inner_name}], <{e.__class__.__name__}: {e.code()}, {e.details()}>, <Time:{record_dict}>")
                raise e
            except Exception as e:
                record_dict["Exception"] = str(datetime.datetime.now())
                LOGGER.error(f"Unexcepted error: [{inner_name}], {e}, <Time: {record_dict}>")
                raise e
        return handler
    return wrapper
