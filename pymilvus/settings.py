import logging.config


class DefaultConfig:
    GRPC_PORT = "19530"
    GRPC_ADDRESS = "127.0.0.1:19530"
    GRPC_URI = f"tcp://{GRPC_ADDRESS}"

    HTTP_PORT = "19121"
    HTTP_ADDRESS = "127.0.0.1:19121"
    HTTP_URI = f"http://{HTTP_ADDRESS}"

    CALC_DIST_METRIC = "L2"


# logging
COLORS = {
    'HEADER': '\033[95m',
    'INFO': '\033[92m',
    'DEBUG': '\033[94m',
    'WARNING': '\033[93m',
    'ERROR': '\033[95m',
    'CRITICAL': '\033[91m',
    'ENDC': '\033[0m',
}


class ColorFulFormatColMixin:
    def format_col(self, message_str, level_name):
        if level_name in COLORS:
            message_str = COLORS.get(level_name) + message_str + COLORS.get('ENDC')
        return message_str


class ColorfulFormatter(logging.Formatter, ColorFulFormatColMixin):
    def format(self, record):
        message_str = super().format(record)

        return self.format_col(message_str, level_name=record.levelname)


LOG_LEVEL = 'WARNING'
# LOG_LEVEL = 'DEBUG'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': LOG_LEVEL,
        },
    },
    'loggers': {
        'milvus': {
            'handlers': ['console'],
            'level': LOG_LEVEL,
        },
    },
}

if LOG_LEVEL == 'DEBUG':
    LOGGING['formatters'] = {
        'colorful_console': {
            'format': '[%(asctime)s-%(levelname)s-%(name)s]: %(message)s (%(filename)s:%(lineno)s)',
            '()': ColorfulFormatter,
        },
    }
    LOGGING['handlers']['milvus_console'] = {
        'class': 'logging.StreamHandler',
        'formatter': 'colorful_console',
    }
    LOGGING['loggers']['milvus'] = {
        'handlers': ['milvus_console'],
        'level': LOG_LEVEL,
    }

logging.config.dictConfig(LOGGING)

DEBUG_LOG_LEVEL = "debug"
INFO_LOG_LEVEL = "info"
WARN_LOG_LEVEL = "warn"
ERROR_LOG_LEVEL = "error"
