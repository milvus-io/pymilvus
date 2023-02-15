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

from typing import Dict, List, Any, Union

import grpc

from .grpc_handler import GrpcHandler
from ..exceptions import MilvusException

from ..types import (
    CollectionSchema,
    ResourceGroupInfo,
    AliasInfo,
    CollectionInfo,
    IndexInfo,
    PartitionInfo,
)
from ..settings import DefaultConfig
from ..grpc_gen import milvus_pb2 as milvus_types
from ..grpc_gen import common_pb2, milvus_pb2_grpc
from ..utils import generate_address


class MilvusClient:
    handler: GrpcHandler

    # TODO secured channel
    def __init__(self, host: str, port: Union[str, int], **kwargs):
        # ut only channel
        _channel = kwargs.get("_channel")
        if isinstance(_channel, grpc.Channel):
            self.handler = milvus_pb2_grpc.MilvusServiceStub(_channel)
            return

        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)) and timeout > 0:
            connection_timeout = timeout
        else:
            connection_timeout = DefaultConfig.CONNECTION_TIMEOUT

        address = generate_address(host, port)

        # set up GrpcHandler
        self.handler = GrpcHandler(address, timeout=connection_timeout, **kwargs)

    def get_server_version(self, **kwargs) -> str:
        req = milvus_types.GetVersionRequest()
        resp = self.handler.GetVersion(req, **kwargs)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.version

    @NotImplementedError
    def create_alias(self, alias: str, collection_name: str, **kwargs) -> None:
        """ Create an alias for a collection

        Args:
            alias (``str``): Specifies an alias desired for the collection.
            collection_name (``str``): Specifies the name of a target collection.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.create_alias("Gandalf", "hello_milvus")
        """
        pass

    @NotImplementedError
    def alter_alias(self, alias: str, collection_name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    def describe_alias(self, alias: str, **kwargs) -> AliasInfo:
        pass

    @NotImplementedError
    def drop_alias(self, alias: str, **kwargs) -> None:
        pass

    @NotImplementedError
    def has_alias(self, alias: str, **kwargs) -> bool:
        pass

    @NotImplementedError
    def list_aliases(self, collection_name: str, **kwargs) -> List[str]:
        pass

    @NotImplementedError
    def create_collection(self, name: str, schema: CollectionSchema, **kwargs) -> None:
        pass

    @NotImplementedError
    def create_schema(self) -> CollectionSchema:
        pass

    @NotImplementedError
    def describe_collection(self, name: str, **kwargs) -> CollectionInfo:
        pass

    @NotImplementedError
    def drop_collection(self, name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    def has_collection(self, name: str, **kwargs) -> bool:
        pass

    def list_collections(self, **kwargs) -> List[str]:
        """ List all collection names in the Database.

        Args:
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Returns:
           List[str]: list of collection names.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.list_collections()
            ['lord_of_the_rings', 'hello_milvus']
        """
        req = milvus_types.ShowCollectionsRequest()
        resp = self.handler.ShowCollections(req, **kwargs)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.collection_names

    @NotImplementedError
    def alter_collection(self, collection_name: str, properties: Dict[str, Any], **kwargs) -> None:
        # TODO
        pass

    @NotImplementedError
    def release_collection(self, name, **kwargs) -> None:
        pass

    @NotImplementedError
    def create_partition(self, collection_name: str, partition_name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    def describe_partition(self, collection_name: str, partition_name: str, **kwargs) -> PartitionInfo:
        # TODO
        pass

    @NotImplementedError
    def drop_partition(self, collection_name: str, partition_name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    def has_partition(self, collection_name: str, partition_name: str, **kwargs) -> bool:
        pass

    @NotImplementedError
    def list_partitions(self, collection_name: str, **kwargs) -> List[str]:
        pass

    @NotImplementedError
    def describe_index(self, collection_name: str, index_name: str, **kwargs) -> IndexInfo:
        pass

    @NotImplementedError
    def drop_index(self, collection_name: str, index_name: str, **kwargs) ->None:
        pass

    @NotImplementedError
    def has_index(self, collection_name: str, index_name: str, **kwargs) -> bool:
        pass

    @NotImplementedError
    def list_indexes(self, collection_name: str, field_name: str, **kwargs) -> List[str]:
        pass

    # ResouceGroup
    # TODO
    def create_resource_group(self, group_name: str, timeout: float=None, **kwargs) -> None:
        pass

    # TODO
    def drop_resource_group(self, group_name: str, timeout: float=None, **kwargs) -> None:
        pass

    # TODO
    def describe_resource_group(self, group_name: str, timeout: float=None, **kwargs) -> ResourceGroupInfo:
        pass

    # TODO
    def list_resource_groups(self, timeout: float=None, **kwargs) -> List[str]:
        pass

    # TODO
    def transfer_node(self, source_group: str, target_group: str, num_node: int, timeout: float=None, **kwargs) -> None:
        pass

    # TODO
    def transfer_replica(self, source_group: str, target_group: str, collection_name: str, num_replica: int, timeout: float=None, **kwargs) -> None:
        pass

    # RBAC TODO


    # Others

    # TODO
    def insert(self, collection_name: str, column_based_entities: List[list], partition_name: str="_default", timeout: float=None, **kwargs):
        pass

    # TODO
    def insert_by_rows(self, collection_name: str, row_based_entities: List[list], partition_name: str="_default", timeout: float=None, **kwargs):
        pass

    # TODO
    def bulk_insert(self, collection_name: str, partition_name: str, files: List[str], timeout: float=None, **kwargs) -> int:
        pass

    # TODO
    def upsert(self, timeout: float=None):
        pass

    # TODO
    def delete(self, collection_name: str, pks: List[Union[int, str]], partition_name: str="_default", timeout: float=None, **kwargs):
        pass

    # TODO
    def delete_by_expression(self, collection_name: str, expression: str, partition_name: str="_default", timeout: float=None, **kwargs):
        pass


    # TODO
    def release_partitions(self, collection_name: str, partition_names: List[str], timeout: float=None, **kwargs):
        pass

    # TODO
    def flush(self, collection_name: str, timeout: float=None, **kwargs):
        pass

    # TODO
    def create_index(self, collection_name: str, field_name: str, index_name: str, timeout: float=None):
        pass


    # TODO
    def search(self, collection_name: str, data: List[Any], anns_field: str, param: Dict[str, Any], limit: int,
        expression: str=None, partition_names: List[str]=None, output_fields: List[str]=None,
        round_decimal: int=-1, timeout: float=None, schema: Any=None, **kwargs):
        pass

    # TODO
    def query(self, collection_name: str, expr: str, output_fields: List[str]=None, partition_names: List[str]=None, timeout: float=None, **kwargs):
        pass

    # TODO
    def load_collection(self, collection_name: str, num_replica: int=1, timeout: float=None, **kwargs):
        pass

    # TODO
    def load_partitions(self, collection_name: str, partition_names: List[str], timeout: float=None, **kwargs):
        pass
