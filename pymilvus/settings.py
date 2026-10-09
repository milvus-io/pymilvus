import logging
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # legacy env MILVUS_DEFAULT_CONNECTION, not recommended
    LEGACY_URI = str(os.getenv("MILVUS_DEFAULT_CONNECTION", ""))
    MILVUS_URI = str(os.getenv("MILVUS_URI", LEGACY_URI))

    MILVUS_CONN_ALIAS = str(os.getenv("MILVUS_CONN_ALIAS", "default"))
    MILVUS_CONN_TIMEOUT = float(os.getenv("MILVUS_CONN_TIMEOUT", "10.0"))

    # legacy configs:
    DEFAULT_USING = MILVUS_CONN_ALIAS
    DEFAULT_CONNECT_TIMEOUT = MILVUS_CONN_TIMEOUT

    # TODO tidy the following configs
    GRPC_PORT = "19530"
    GRPC_ADDRESS = "127.0.0.1:19530"
    GRPC_URI = f"tcp://{GRPC_ADDRESS}"

    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = "19530"

    WaitTimeDurationWhenLoad = 0.2  # in seconds
    MaxVarCharLengthKey = "max_length"
    MaxVarCharLength = 65535
    EncodeProtocol = "utf-8"
    IndexName = ""


# logging
LOG_FORMAT = "%(asctime)s [%(levelname)s][%(funcName)s]: %(message)s (%(filename)s:%(lineno)s)"
LOG_HANDLER_NAME = "pymilvus_console"
_LOGGER_LEVELS = (
    ("pymilvus.milvus_client", "INFO"),
    ("pymilvus.bulk_writer", "INFO"),
)


def _get_console_handler() -> logging.Handler:
    """Return the shared pymilvus console handler, creating it on first use.

    The handler is looked up on the ``pymilvus`` logger tree by name so that
    re-running :func:`init_log` (e.g. ``importlib.reload``) does not stack
    duplicate handlers.
    """
    for name in ("pymilvus", *(name for name, _ in _LOGGER_LEVELS)):
        for handler in logging.getLogger(name).handlers:
            if handler.get_name() == LOG_HANDLER_NAME:
                return handler
    handler = logging.StreamHandler()
    handler.set_name(LOG_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def init_log(log_level: str):
    """Configure logging for the ``pymilvus`` logger tree only.

    Only the ``pymilvus*`` loggers are touched. In particular, this must not go
    through ``logging.config.dictConfig``: in its default (non-incremental)
    mode that helper closes every handler already registered in the process
    and strips handlers the application attached to the ``pymilvus`` logger,
    silently breaking logging configured before ``import pymilvus``.

    The shared console handler is only attached to a logger that has no other
    handlers: if the application already routes a ``pymilvus*`` logger to its
    own handler, adding the console handler as well would emit every record
    twice.
    """
    handler = _get_console_handler()
    for name, level in (("pymilvus", log_level), *_LOGGER_LEVELS):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        has_app_handler = any(h.get_name() != LOG_HANDLER_NAME for h in logger.handlers)
        if has_app_handler:
            logger.removeHandler(handler)
        elif handler not in logger.handlers:
            logger.addHandler(handler)


init_log("WARNING")
