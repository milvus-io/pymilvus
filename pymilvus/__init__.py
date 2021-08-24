# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

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
from .orm.pymilvus_orm.connections import connections, Connections

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
    drop_collection,
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
    'connections',
    'loading_progress', 'index_building_progress', 'wait_for_loading_complete', 'has_collection', 'has_partition', 'list_collections', 'wait_for_loading_complete', 'wait_for_index_building_complete', 'drop_collection',
    'SearchResult', 'Hits', 'Hit',
    'FieldSchema', 'CollectionSchema',
    'SearchFuture', 'MutationFuture',
    'utility', 'DefaultConfig', 'ExceptionsMessage',

    # pymilvus old style APIs
    'Milvus', 'Prepare', 'Status', 'DataType',
    'ParamError', 'ConnectError', 'NotConnectError', 'RepeatingConnectError', 'VersionError', 'BaseException',
    '__version__'
]
