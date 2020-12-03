# -*- coding: utf-8 -*-

from .client.stub import Milvus
from .client.types import Status, DataType, RangeType
from .client.exceptions import (
    ParamError,
    ConnectError,
    NotConnectError,
    RepeatingConnectError,
    VersionError,
    BaseError
)
from .client import __version__

__all__ = ['Milvus', 'Status', 'DataType',
           'ParamError', 'ConnectError', 'NotConnectError', 'RepeatingConnectError', 'VersionError', 'BaseError',
           '__version__']
