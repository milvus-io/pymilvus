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

"""client module"""
from pkg_resources import get_distribution, DistributionNotFound

from .collection import Collection
from .connections import (
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

from .index import Index
from .partition import Partition
from .utility import (
        loading_progress,
        index_building_progress,
        wait_for_loading_complete,
        wait_for_index_building_complete,
        has_collection,
        has_partition,
        list_collections,
)

from .search import SearchResult, Hits, Hit
from .types import DataType
from .schema import FieldSchema, CollectionSchema
from .future import SearchFuture, MutationFuture

__version__ = '0.0.0.dev'

try:
    __version__ = get_distribution('pymilvus-orm').version
except DistributionNotFound:
    # package is not installed
    pass
