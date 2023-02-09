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

from typing import Dict, List, Any

from .types import CollectionSchema, ResourceGroupInfo
#  from .grpc_handler import handler


# Alias
# TODO
def create_alias(alias: str, collection_name: str, timeout: float=None) -> None:
    pass


# TODO
def drop_alias(alias: str, timeout: float=None) -> None:
    pass


# TODO
def alter_alias(alias: str, collection_name: str, timeout: float=None) -> None:
    pass


# TODO
def list_aliases(collection_name: str, timeout: float=None) -> List[str]:
    pass

# Collection
# TODO
def create_collection(collection_name: str, schema: CollectionSchema, num_shards: int=2, timeout: float=None, **kwargs) -> None:
    pass

# TODO
def drop_collection(collection_name: str, timeout: float=None, **kwargs) -> None:
    pass

# TODO
def has_collection(collection_name: str, timeout: float=None, **kwargs) -> bool:
    pass

# TODO
def describe_collection(collection_name: str, timeout: float=None, **kwargs) -> Dict[str, Any]:
    pass

# TODO
def alter_collection(collection_name: str, properties: Dict[str, Any], timeout: float=None, **kwargs) -> None:
    pass

# TODO
def get_collection_statistics(collection_name: str, timeout: float=None, **kwargs) -> Dict[str, Any]:
    pass

# TODO
def list_collections(timeout: float=None) -> List[str]:
    pass

# Partition
# TODO
def create_partition(collection_name: str, partition_name: str, timeout: float=None) -> None:
    pass

# TODO
def drop_partition(collection_name: str, partition_name: str, timeout: float=None) -> None:
    pass

# TODO
def has_partition(collection_name: str, partition_name: str, timeout: float=None) -> bool:
    pass

# TODO
def describe_partition(collection_name: str, partition_name: str, timeout: float=None) -> Dict[str, Any]:
    pass

# TODO
def get_partition_statistics(collection_name: str, partition_name: str, timeout: float=None) -> Dict[str, Any]:
    pass

# TODO
def list_partitions(collection_name: str, timeout: float=None) -> List[str]:
    pass

# Index
# TODO
def drop_index(collection_name: str, field_name: str, index_name: str, timeout: float=None) ->None:
    pass

# TODO
def has_index(collection_name: str, field_name: str, index_name: str, timeout: float=None) -> bool:
    pass

# TODO
def list_indexes(collection_name: str, timeout: float=None, **kwargs) -> List[str]:
    pass

# TODO
def describe_index(collection_name: str, index_name: str, timeout: float=None, **kwargs) -> Dict[str, Any]:
    pass

# ResouceGroup
# TODO
def create_resource_group(group_name: str, timeout: float=None, **kwargs) -> None:
    pass

# TODO
def drop_resource_group(group_name: str, timeout: float=None, **kwargs) -> None:
    pass

# TODO
def describe_resource_group(group_name: str, timeout: float=None, **kwargs) -> ResourceGroupInfo:
    pass

# TODO
def list_resource_groups(timeout: float=None, **kwargs) -> List[str]:
    pass

# TODO
def transfer_node(source_group: str, target_group: str, num_node: int, timeout: float=None, **kwargs) -> None:
    pass

# TODO
def transfer_replica(source_group: str, target_group: str, collection_name: str, num_replica: int, timeout: float=None, **kwargs) -> None:
    pass

# RBAC TODO


# Others
# TODO
def get_server_version(timeout: float=None, **kwargs) -> str:
    pass
