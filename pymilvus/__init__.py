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
from .client.abstract import AnnSearchRequest, Hit, Hits, RRFRanker, SearchResult, WeightedRanker
from .client.asynch import SearchFuture
from .client.prepare import Prepare
from .client.stub import Milvus
from .client.types import (
    BulkInsertState,
    DataType,
    Group,
    IndexType,
    Replica,
    ResourceGroupInfo,
    Shard,
    Status,
)
from .exceptions import (
    ExceptionsMessage,
    MilvusException,
    MilvusUnavailableException,
)
from .milvus_client import MilvusClient
from .orm import db, utility
from .orm.collection import Collection
from .orm.connections import Connections, connections
from .orm.future import MutationFuture
from .orm.index import Index
from .orm.partition import Partition
from .orm.role import Role
from .orm.schema import CollectionSchema, FieldSchema
from .orm.utility import (
    create_resource_group,
    create_user,
    delete_user,
    describe_resource_group,
    drop_collection,
    drop_resource_group,
    has_collection,
    has_partition,
    hybridts_to_datetime,
    hybridts_to_unixtime,
    index_building_progress,
    list_collections,
    list_resource_groups,
    list_usernames,
    loading_progress,
    mkts_from_datetime,
    mkts_from_hybridts,
    mkts_from_unixtime,
    reset_password,
    transfer_node,
    transfer_replica,
    update_password,
    update_resource_groups,
    wait_for_index_building_complete,
    wait_for_loading_complete,
)

# Compatiable
from .settings import Config as DefaultConfig

__all__ = [
    "Collection",
    "Index",
    "Partition",
    "connections",
    "loading_progress",
    "index_building_progress",
    "wait_for_index_building_complete",
    "drop_collection",
    "has_collection",
    "list_collections",
    "wait_for_loading_complete",
    "has_partition",
    "mkts_from_hybridts",
    "mkts_from_unixtime",
    "mkts_from_datetime",
    "hybridts_to_unixtime",
    "hybridts_to_datetime",
    "reset_password",
    "create_user",
    "update_password",
    "update_resource_groups",
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
    "db",
    "DefaultConfig",
    "Role",
    "ExceptionsMessage",
    "MilvusUnavailableException",
    "BulkInsertState",
    "create_resource_group",
    "drop_resource_group",
    "describe_resource_group",
    "list_resource_groups",
    "transfer_node",
    "transfer_replica",
    "Milvus",
    "Prepare",
    "Status",
    "DataType",
    "MilvusException",
    "__version__",
    "MilvusClient",
    "ResourceGroupInfo",
    "Connections",
    "IndexType",
    "AnnSearchRequest",
    "RRFRanker",
    "WeightedRanker",
]
