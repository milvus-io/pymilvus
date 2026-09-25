import logging
import os


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


def init_log(log_level: str):
    """Configure PyMilvus loggers without modifying unrelated logging state."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s][%(funcName)s]: %(message)s (%(filename)s:%(lineno)s)"
        )
    )
    for name, level in (
        ("pymilvus", log_level),
        ("pymilvus.milvus_client", "INFO"),
        ("pymilvus.bulk_writer", "INFO"),
    ):
        logger = logging.getLogger(name)
        for existing in logger.handlers[:]:
            logger.removeHandler(existing)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


init_log("WARNING")
