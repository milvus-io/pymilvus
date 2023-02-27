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

from .grpc_handler import AsyncGrpcHandler
from ..settings import DefaultConfig
from ..utils import generate_address, TypeChecker
from ..grpc_gen import milvus_pb2 as milvus_types
from ..grpc_gen import common_pb2
from ..exceptions import MilvusException
from ..types import (
    CollectionSchema,
    AliasInfo,
    CollectionInfo,
    IndexInfo,
    PartitionInfo,
)

class AsyncMilvusClient:
    handler: AsyncGrpcHandler

    # TODO secured channel
    def __init__(self, host: str, port: Union[str, int], **kwargs):

        timeout = kwargs.get("timeout")
        if isinstance(timeout, (int, float)) and timeout > 0:
            connection_timeout = timeout
        else:
            connection_timeout = DefaultConfig.CONNECTION_TIMEOUT

        address = generate_address(host, port)

        # set up GrpcHandler
        self.handler = AsyncGrpcHandler(address, timeout=connection_timeout, **kwargs)

    async def get_server_version(self, **kwargs) -> str:
        req = milvus_types.GetVersionRequest()
        resp = await self.handler.GetVersion(req, **kwargs)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.version

    async def create_alias(self, alias: str, collection_name: str, **kwargs) -> None:
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
            >>> from pymilvus.aio import AsyncMilvusClient
            >>> client = AsyncMilvusClient("localhost", "19530")
            >>> await client.create_alias("Gandalf", "hello_milvus")
        """
        TypeChecker.check(alias=alias, collection_name=collection_name)
        req = milvus_types.CreateAliasRequest(alias=alias, collection_name=collection_name)
        status = await self.handler.CreateAlias(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    async def alter_alias(self, alias: str, collection_name: str, **kwargs) -> None:
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
            >>> from pymilvus.aio import AsyncMilvusClient
            >>> client = AsyncMilvusClient("localhost", "19530")
            >>> await client.alter_alias("Gandalf", "hello_milvus")
        """
        TypeChecker.check(alias=alias, collection_name=collection_name)
        req = milvus_types.AlterAliasRequest(alias=alias, collection_name=collection_name)
        status = await self.handler.AlterAlias(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    @NotImplementedError
    async def describe_alias(self, alias: str, **kwargs) -> AliasInfo:
        pass

    async def drop_alias(self, alias: str, **kwargs) -> None:
        """ Drop the alias

        Args:
            alias (``str``): Specifies an alias desired for the collection.
            **kwargs (``dict``, optional):

                * *timeout*(``float``): Specifies the timeout duration of this operation in seconds.
                    The value defaults to ``None``, indicating that no such limit applies.

        Raises:
            MilvusException: If anything goes wrong.

        Examples:
            >>> from pymilvus.aio import AsyncMilvusClient
            >>> client = AsyncMilvusClient("localhost", "19530")
            >>> await client.drop_alias("Gandalf")
        """
        TypeChecker.check(alias=alias)
        req = milvus_types.DropAliasRequest(alias=alias)
        status = await self.handler.DropAlias(req)
        if status.error_code != common_pb2.Success:
            raise MilvusException(status.error_code, status.reason)

    @NotImplementedError
    async def has_alias(self, alias: str, **kwargs) -> bool:
        pass

    @NotImplementedError
    async def list_aliases(self, collection_name: str, **kwargs) -> List[str]:
        pass

    @NotImplementedError
    async def create_collection(self, name: str, schema: CollectionSchema, **kwargs) -> None:
        pass

    @NotImplementedError
    async def create_schema(self) -> CollectionSchema:
        pass

    @NotImplementedError
    async def describe_collection(self, name: str, **kwargs) -> CollectionInfo:
        pass

    @NotImplementedError
    async def drop_collection(self, name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    async def has_collection(self, name: str, **kwargs) -> bool:
        pass

    async def list_collections(self, **kwargs) -> List[str]:
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
        resp = await self.handler.ShowCollections(req, **kwargs)
        if resp.status.error_code != common_pb2.Success:
            raise MilvusException(resp.status.error_code, resp.status.reason)

        return resp.collection_names

    @NotImplementedError
    async def alter_collection(self, name: str, properties: Dict[str, Any], **kwargs) -> None:
        # TODO
        pass

    @NotImplementedError
    async def release_collection(self, name, **kwargs) -> None:
        pass

    @NotImplementedError
    async def create_partition(self, collection_name: str, partition_name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    async def describe_partition(self, collection_name: str, partition_name: str, **kwargs) -> PartitionInfo:
        # TODO
        pass

    @NotImplementedError
    async def drop_partition(self, collection_name: str, partition_name: str, **kwargs) -> None:
        pass

    @NotImplementedError
    async def has_partition(self, collection_name: str, partition_name: str, **kwargs) -> bool:
        pass

    @NotImplementedError
    async def list_partitions(self, collection_name: str, **kwargs) -> List[str]:
        pass

    @NotImplementedError
    async def describe_index(self, collection_name: str, index_name: str, **kwargs) -> IndexInfo:
        pass

    @NotImplementedError
    async def drop_index(self, collection_name: str, index_name: str, **kwargs) ->None:
        pass

    @NotImplementedError
    async def has_index(self, collection_name: str, index_name: str, **kwargs) -> bool:
        pass

    @NotImplementedError
    async def list_indexes(self, collection_name: str, field_name: str, **kwargs) -> List[str]:
        pass
