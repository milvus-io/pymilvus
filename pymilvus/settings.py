import contextlib
import logging.config

import environs

env = environs.Env()

with contextlib.suppress(Exception):
    env.read_env(".env")


class Config:
    # legacy env MILVUS_DEFAULT_CONNECTION, not recommended
    LEGACY_URI = env.str("MILVUS_DEFAULT_CONNECTION", "")
    MILVUS_URI = env.str("MILVUS_URI", LEGACY_URI)

    MILVUS_CONN_ALIAS = env.str("MILVUS_CONN_ALIAS", "default")
    MILVUS_CONN_TIMEOUT = env.float("MILVUS_CONN_TIMEOUT", 10)

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
COLORS = {
    "HEADER": "\033[95m",
    "INFO": "\033[92m",
    "DEBUG": "\033[94m",
    "WARNING": "\033[93m",
    "ERROR": "\033[95m",
    "CRITICAL": "\033[91m",
    "ENDC": "\033[0m",
}


class ColorFulFormatColMixin:
    def format_col(self, message_str: str, level_name: str):
        if level_name in COLORS:
            message_str = COLORS.get(level_name) + message_str + COLORS.get("ENDC")
        return message_str


class ColorfulFormatter(logging.Formatter, ColorFulFormatColMixin):
    def format(self, record: str):
        message_str = super().format(record)

        return self.format_col(message_str, level_name=record.levelname)


LOG_LEVEL = "WARNING"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
        },
    },
    "loggers": {
        "milvus": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
        },
    },
}

if LOG_LEVEL == "DEBUG":
    LOGGING["formatters"] = {
        "colorful_console": {
            "format": "[%(asctime)s-%(levelname)s-%(name)s]: %(message)s (%(filename)s:%(lineno)s)",
            "()": ColorfulFormatter,
        },
    }
    LOGGING["handlers"]["milvus_console"] = {
        "class": "logging.StreamHandler",
        "formatter": "colorful_console",
    }
    LOGGING["loggers"]["milvus"] = {
        "handlers": ["milvus_console"],
        "level": LOG_LEVEL,
    }

logging.config.dictConfig(LOGGING)

DEBUG_LOG_LEVEL = "debug"
INFO_LOG_LEVEL = "info"
WARN_LOG_LEVEL = "warn"
ERROR_LOG_LEVEL = "error"
