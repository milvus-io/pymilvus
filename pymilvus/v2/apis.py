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
from ..utils import generate_address, TypeChecker


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
        resp = self.handler.GetVersion(req)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.version

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
        TypeChecker.check(alias=alias, collection_name=collection_name)
        req = milvus_types.CreateAliasRequest(alias=alias, collection_name=collection_name)
        status = self.handler.CreateAlias(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    def alter_alias(self, alias: str, collection_name: str, **kwargs) -> None:
        """ Alter this alias to a new collection

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
            >>> client.alter_alias("Gandalf", "hello_milvus")
        """
        TypeChecker.check(alias=alias, collection_name=collection_name)
        req = milvus_types.AlterAliasRequest(alias=alias, collection_name=collection_name)
        status = self.handler.AlterAlias(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    @NotImplementedError
    def describe_alias(self, alias: str, **kwargs) -> AliasInfo:
        pass

    def drop_alias(self, alias: str, **kwargs) -> None:
        """ Drop the alias

        Args:
            alias (``str``): Specifies an alias desired for the collection.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.drop_alias("Gandalf")
        """
        TypeChecker.check(alias=alias)
        req = milvus_types.DropAliasRequest(alias=alias)
        status = self.handler.DropAlias(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    @NotImplementedError
    def has_alias(self, alias: str, **kwargs) -> bool:
        """ Check if this alias exists

        Args:
            alias (``str``): Specifies an alias desired for the collection.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.has_alias("Gandalf")
            False
        """
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

    def drop_collection(self, name: str, **kwargs) -> None:
        """ Drop the collection

        Args:
            name (``str``): Specifies the name of the collection.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.drop_collection("hello_milvus")
        """
        TypeChecker.check(name=name)
        req = milvus_types.DropCollectionRequest(collection_name=name)
        status = self.handler.DropCollection(req)

        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    def has_collection(self, name: str, **kwargs) -> bool:
        """ Check if this collection exists

        Args:
            name (``str``): Specifies the name of the collection.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.has_collection("hello_milvus")
            False
        """
        TypeChecker.check(name=name)
        req = milvus_types.DescribeCollectionRequest(collection_name=name)
        resp = self.handler.DescribeCollection(req)

        def is_status_abnormal(status: common_pb2.Status) -> bool:
            if status.error_code in (common_pb2.Success, common_pb2.CollectionNotExists) \
                or (status.error_code == common_pb2.UnexpectedError and "can\'t find collection" in status.reason):
                return False
            return True

        if is_status_abnormal(resp.status):
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.status.error_code == common_pb2.Success


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
        resp = self.handler.ShowCollections(req)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.collection_names

    @NotImplementedError
    def alter_collection(self, collection_name: str, properties: Dict[str, Any], **kwargs) -> None:
        # TODO
        pass

    def release_collection(self, name, **kwargs) -> None:
        """ Release the loaded collection

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
            >>> client.release_collection("hello_milvus")
        """
        req = milvus_types.ReleaseCollectionRequest(collection_name=name)
        status = self.handler.ReleaseCollection(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    def create_partition(self, collection_name: str, partition_name: str, **kwargs) -> None:
        """ create a partition for the collection

        Args:
            collection_name (``str``): Name of the collection
            partition_name (``str``): Name of the partition
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.create_partition("hello_milvus", "movies")
        """
        TypeChecker.check(collection_name=collection_name, partition_name=partition_name)
        req = milvus_types.CreatePartitionRequest(collection_name=collection_name, partition_name=partition_name)
        status = self.handler.CreatePartition(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)


    @NotImplementedError
    def describe_partition(self, collection_name: str, partition_name: str, **kwargs) -> PartitionInfo:
        # TODO
        pass

    def drop_partition(self, collection_name: str, partition_name: str, **kwargs) -> None:
        """ Drop a partition in the collection

        Args:
            collection_name (``str``): Specifies the name of the collection.
            partition_name (``str``): Specifies the name of the partition.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.drop_partition("hello_milvus", "movies")
        """
        TypeChecker.check(collection_name=collection_name, partition_name=partition_name)
        req = milvus_types.DropPartitionRequest(collection_name=collection_name, partition_name=partition_name)
        status = self.handler.DropPartition(req)

        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    def has_partition(self, collection_name: str, partition_name: str, **kwargs) -> bool:
        """ Check if a partition exist in the collection

        Args:
            collection_name (``str``): Specifies the name of the collection.
            partition_name (``str``): Specifies the name of the partition.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Returns:
            bool: whether the partition exists.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.has_partition("hello_milvus", "movies")
            False
        """
        TypeChecker.check(collection_name=collection_name, partition_name=partition_name)
        req = milvus_types.HasPartitionRequest(collection_name=collection_name, partition_name=partition_name)
        resp = self.handler.HasPartition(req)

        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.value

    @NotImplementedError
    def list_partitions(self, collection_name: str, **kwargs) -> List[str]:
        pass

    def release_partitions(self, collection_name: str, partition_names: List[str],  **kwargs):
        """ Release multiple partitions off the memory in the collection

        Args:
            collection_name (``str``): Specifies the name of the collection.
            partition_names (``List[str]``): Specifies the list of names of the partitions.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.release_partitions("hello_milvus", "movies")
        """
        TypeChecker.check(collection_name=collection_name, partition_names=partition_names)
        req = milvus_types.ReleasePartitionsRequest(collection_name=collection_name, partition_names=partition_names)
        status = self.handler.ReleasePartitions(req)

        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    @NotImplementedError
    def describe_index(self, collection_name: str, index_name: str, **kwargs) -> IndexInfo:
        pass

    def drop_index(self, collection_name: str, index_name: str, **kwargs) ->None:
        """ Drop a index in the collection

        Args:
            collection_name (``str``): Specifies the name of the collection.
            index_name (``str``): Specifies the name of the index.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.drop_index("hello_milvus", "index_ivf")
        """
        TypeChecker.check(collection_name=collection_name, index_name=index_name)
        req = milvus_types.DropIndexRequest(collection_name=collection_name, index_name=index_name)
        status = self.handler.DropIndex(req)

        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    @NotImplementedError
    def has_index(self, collection_name: str, index_name: str, **kwargs) -> bool:
        pass

    def list_indexes(self, collection_name: str, field_name: str, **kwargs) -> List[str]:
        """ List all indexes' name of the collection field

        Args:
            collection_name (``str``): Name of the collection
            field_name (``str``): Name of the field
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Returns:
           List[str]: list of index names.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus import MilvusClient
            >>> client = MilvusClient("localhost", "19530")
            >>> client.list_indexes("hello_milvus", "vector_field")
            ["index_ivf"]
        """
        TypeChecker.check(collection_name=collection_name, field_name=field_name)
        req = milvus_types.DescribeIndexRequest(collection_name=collection_name, field_name=field_name)
        resp = self.handler.DescribeIndex(req)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        index_names = [d.index_name for d in resp.index_descriptions]

        return index_names

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
