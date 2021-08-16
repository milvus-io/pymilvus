# -*- coding: utf-8 -*-

from .client.stub import Milvus
from .client.prepare import Prepare
from .client.types import Status, DataType, RangeType, IndexType
from .client.exceptions import (
    ParamError,
    ConnectError,
    NotConnectError,
    RepeatingConnectError,
    VersionError,
    BaseException
)
#  comment for dup
#  from .client.asynch import MutationFuture
from .client import __version__


"""client module"""
from .orm.pymilvus_orm.collection import Collection
from .orm.pymilvus_orm.connections import (
    Connections,
    connections,
    add_connection,
    list_connections,
    get_connection_addr,
    remove_connection,
    connect,
    get_connection,
    disconnect
)

from .orm.pymilvus_orm.index import Index
from .orm.pymilvus_orm.partition import Partition
from .orm.pymilvus_orm.utility import (
    loading_progress,
    index_building_progress,
    wait_for_loading_complete,
    wait_for_index_building_complete,
    has_collection,
    has_partition,
    list_collections,
)

from .orm.pymilvus_orm import utility
from .orm.pymilvus_orm.default_config import DefaultConfig

from .orm.pymilvus_orm.search import SearchResult, Hits, Hit
from .orm.pymilvus_orm.schema import FieldSchema, CollectionSchema
from .orm.pymilvus_orm.future import SearchFuture, MutationFuture
from .orm.pymilvus_orm.exceptions import ExceptionsMessage

__all__ = [
    # pymilvus orm'styled APIs
    'Collection', 'Index', 'Partition',
    'Connections', 'connections', 'add_connection', 'list_connections', 'get_connection_addr', 'remove_connection', 'connect', 'get_connection', 'disconnect',
    'loading_progress', 'index_building_progress', 'wait_for_loading_complete', 'has_collection', 'has_partition', 'list_collections',
    'SearchResult', 'Hits', 'Hit',
    'FieldSchema', 'CollectionSchema',
    'SearchFuture', 'MutationFuture',
    'utility', 'DefaultConfig', 'ExceptionsMessage',

    # pymilvus old style APIs
    'Milvus', 'Prepare', 'Status', 'DataType',
    'ParamError', 'ConnectError', 'NotConnectError', 'RepeatingConnectError', 'VersionError', 'BaseException',
    '__version__'
]
