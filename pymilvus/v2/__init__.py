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

# APIs that V2 provided
from .grpc_handler import connect, insecured_connect

from .types import CollectionSchema, ResourceGroupInfo

from .apis import ( # alias
    create_alias,
    drop_alias,
    alter_alias,
    list_aliases,
)
from .apis import ( # collection
    create_collection,
    drop_collection,
    has_collection,
    describe_collection,
    alter_collection,
    get_collection_statistics,
    list_collections
)
from .apis import ( # partition
    create_partition,
    drop_partition,
    has_partition,
    describe_partition,
    get_partition_statistics,
    list_partitions
)
from .apis import ( # index
    drop_index,
    has_index,
    list_indexes,
    describe_index,
)
from .apis import ( # resource group
    create_resource_group,
    drop_resource_group,
    describe_resource_group,
    list_resource_groups,
    transfer_node,
    transfer_replica,
)
from .apis import ( # others
    get_server_version,
)

from .async_apis import ( # async APIs
    load_collection,
    load_partitions,
    release_collection,
    release_partitions,
    flush,
    create_index,
    insert,
    insert_by_rows,
    bulk_insert,
    upsert,
    delete,
    delete_by_expression,
    search,
    query,
)


# TODO
__all__ = [ ]
