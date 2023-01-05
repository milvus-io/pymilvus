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

from .client import __version__
from .client.prepare import Prepare
from .client.stub import Milvus
from .client.types import (
    BulkInsertState,
    DataType,
    Group,
    IndexType,
    RangeType,
    Replica,
    Shard,
    Status,
)
from .exceptions import (
    ExceptionsMessage,
    MilvusException,
    MilvusUnavailableException,
    ParamError,
)
from .orm import utility
from .orm.collection import Collection
from .orm.connections import Connections, connections
from .orm.default_config import ENV_CONNECTION_CONF, DefaultConfig
from .orm.future import MutationFuture, SearchFuture
from .orm.index import Index
from .orm.partition import Partition
from .orm.role import Role
from .orm.schema import CollectionSchema, FieldSchema
from .orm.search import Hit, Hits, SearchResult
from .orm.utility import (
    create_user,
    delete_user,
    do_bulk_insert,
    drop_collection,
    get_bulk_insert_state,
    get_query_segment_info,
    has_collection,
    has_partition,
    hybridts_to_datetime,
    hybridts_to_unixtime,
    index_building_progress,
    list_bulk_insert_tasks,
    list_collections,
    list_usernames,
    load_balance,
    loading_progress,
    mkts_from_datetime,
    mkts_from_hybridts,
    mkts_from_unixtime,
    reset_password,
    update_password,
    wait_for_index_building_complete,
    wait_for_loading_complete,
)
from .settings import DEBUG_LOG_LEVEL, ERROR_LOG_LEVEL, INFO_LOG_LEVEL, WARN_LOG_LEVEL


__all__ = [
    "Collection",
    "Index",
    "Partition",
    "connections",
    "loading_progress",
    "index_building_progress",
    "wait_for_loading_complete",
    "has_collection",
    "has_partition",
    "list_collections",
    "wait_for_loading_complete",
    "wait_for_index_building_complete",
    "drop_collection",
    "mkts_from_hybridts",
    "mkts_from_unixtime",
    "mkts_from_datetime",
    "hybridts_to_unixtime",
    "hybridts_to_datetime",
    "reset_password",
    "create_user",
    "update_password",
    "delete_user",
    "list_usernames",
    "SearchResult",
    "Hits",
    "Hit",
    "Replica",
    "Group",
    "Shard",
    "FieldSchema",
    "CollectionSchema",
    "SearchFuture",
    "MutationFuture",
    "utility",
    "DefaultConfig",
    "ExceptionsMessage",
    "MilvusUnavailableException",
    "BulkInsertState",
    "Role",
    "Milvus",
    "Prepare",
    "Status",
    "DataType",
    "MilvusException",
    "__version__",
]
