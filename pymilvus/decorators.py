import time
import datetime
import logging
import functools

import grpc

from .client.exceptions import CollectionNotExistException, BaseException
from .client.types import ErrorCode, Status

LOGGER = logging.getLogger(__name__)


def check_has_collection(func):
    @functools.wraps(func)
    def handler(self, *args, **kwargs):
        collection_name = args[0]
        if not self.has_collection(collection_name):
            raise CollectionNotExistException(ErrorCode.CollectionNotExists,
                                              f"collection {collection_name} doesn't exist!")
        return func(self, *args, **kwargs)
    return handler


def deprecated(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        dup_msg = f"[WARNING] PyMilvus: class Milvus will be deprecated soon, please use Collection/utility instead"
        LOGGER.warning(f"\033[93m{dup_msg}\033[0m")
        return func(*args, **kwargs)
    return inner


def retry_on_rpc_failure(retry_times=10, wait=1, retry_on_deadline=True):
    def wrapper(func):
        @functools.wraps(func)
        def handler(self, *args, **kwargs):
            counter = 1
            while True:
                try:
                    return func(self, *args, **kwargs)
                except grpc.RpcError as e:
                    # DEADLINE_EXCEEDED means that the task wat not completed
                    # UNAVAILABLE means that the service is not reachable currently
                    # Reference: https://grpc.github.io/grpc/python/grpc.html#grpc-status-code
                    if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED and e.code() != grpc.StatusCode.UNAVAILABLE:
                        raise e
                    if not retry_on_deadline and e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                        raise e
                    if counter >= retry_times:
                        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                            raise BaseException(Status.UNEXPECTED_ERROR, "rpc timeout")
                        raise e
                    time.sleep(wait)
                except Exception as e:
                    raise e
                finally:
                    counter += 1

        return handler
    return wrapper


def error_handler(func):
    @functools.wraps(func)
    def handler(*args, **kwargs):
        record_dict = {}
        try:
            record_dict["RPC start"] = str(datetime.datetime.now())
            return func(*args, **kwargs)
        except BaseException as e:
            record_dict["RPC error"] = str(datetime.datetime.now())
            LOGGER.error(f"RPC error: [{func.__name__}], {e}, <Time:{record_dict}>")
            raise e
        except grpc.FutureTimeoutError as e:
            record_dict["gRPC timeout"] = str(datetime.datetime.now())
            LOGGER.error(f"grpc Timeout: [{func.__name__}], <{e.__class__.__name__}: {e.code()}, {e.details()}>, <Time:{record_dict}>")
            raise e
        except grpc.RpcError as e:
            record_dict["gRPC error"] = str(datetime.datetime.now())
            LOGGER.error(f"grpc RpcError: [{func.__name__}], <{e.__class__.__name__}: {e.code()}, {e.details()}>, <Time:{record_dict}>")
            raise e
        except Exception as e:
            record_dict["Exception"] = str(datetime.datetime.now())
            LOGGER.error(f"Unexcepted error: [{func.__name__}], {e}, <Time: {record_dict}>")
            raise e
    return handler
