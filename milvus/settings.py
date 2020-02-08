import logging.config


class DefaultConfig:
    GRPC_PORT = "19530"
    GRPC_ADDRESS = "127.0.0.1:19530"
    GRPC_URI = "tcp://{}".format(GRPC_ADDRESS)

    HTTP_PORT = "19121"
    HTTP_ADDRESS = "127.0.0.1:19121"
    HTTP_URI = "http://{}".format(HTTP_ADDRESS)


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
        if level_name in COLORS.keys():
            message_str = COLORS.get(level_name) + message_str + COLORS.get(
                'ENDC')
        return message_str


class ColorfulFormatter(logging.Formatter, ColorFulFormatColMixin):
    def format(self, record):
        message_str = super(ColorfulFormatter, self).format(record)

        return self.format_col(message_str, level_name=record.levelname)


LOG_LEVEL = 'WARNING'
# LOG_LEVEL = 'INFO'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
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
