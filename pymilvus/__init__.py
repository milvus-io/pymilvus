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
from .client.types import Status, DataType, RangeType, IndexType, Replica, Group, Shard, BulkLoadState
from .exceptions import (
    ParamError,
    ConnectError,
    NotConnectError,
    RepeatingConnectError,
    VersionError,
    MilvusException,
    MilvusUnavailableException,
    ExceptionsMessage
)
#  comment for dup
#  from .client.asynch import MutationFuture
from .client import __version__

"""client module"""
from .orm.collection import Collection
from .orm.connections import connections, Connections

from .orm.index import Index
from .orm.partition import Partition
from .orm.utility import (
    loading_progress,
    index_building_progress,
    wait_for_loading_complete,
    wait_for_index_building_complete,
    has_collection,
    has_partition,
    list_collections,
    drop_collection,
    get_query_segment_info,
    load_balance,
    mkts_from_hybridts, mkts_from_unixtime, mkts_from_datetime,
    hybridts_to_unixtime, hybridts_to_datetime,
    # bulk_load, get_bulk_load_state, list_bulk_load_tasks,
    reset_password, create_credential, update_credential, delete_credential, list_cred_users,
)

from .orm import utility
from .orm.default_config import DefaultConfig

from .orm.search import SearchResult, Hits, Hit
from .orm.schema import FieldSchema, CollectionSchema
from .orm.future import SearchFuture, MutationFuture

__all__ = [
    # pymilvus orm'styled APIs
    'Collection', 'Index', 'Partition',
    'connections',
    'loading_progress', 'index_building_progress', 'wait_for_loading_complete', 'has_collection', 'has_partition',
    'list_collections', 'wait_for_loading_complete', 'wait_for_index_building_complete', 'drop_collection',
    'mkts_from_hybridts', 'mkts_from_unixtime', 'mkts_from_datetime',
    'hybridts_to_unixtime', 'hybridts_to_datetime',
    'reset_password', 'create_credential', 'update_credential', 'delete_credential', 'list_cred_users',
    'SearchResult', 'Hits', 'Hit', 'Replica', 'Group', 'Shard',
    'FieldSchema', 'CollectionSchema',
    'SearchFuture', 'MutationFuture',
    'utility', 'DefaultConfig', 'ExceptionsMessage', 'MilvusUnavailableException',
    'BulkLoadState',

    # pymilvus old style APIs
    'Milvus', 'Prepare', 'Status', 'DataType',
    'ParamError', 'ConnectError', 'NotConnectError', 'RepeatingConnectError', 'VersionError', 'MilvusException',
    '__version__'
]
