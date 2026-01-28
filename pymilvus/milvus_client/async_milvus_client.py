import asyncio
import time
from typing import Dict, List, Optional, Union

from pymilvus.client.abstract import AnnSearchRequest, BaseRanker
from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.types import (
    ExceptionsMessage,
    LoadState,
    OmitZeroDict,
    ResourceGroupConfig,
    RoleInfo,
    SegmentInfo,
    UserInfo,
)
from pymilvus.client.utils import (
    convert_struct_fields_to_user_format,
    is_vector_type,
)
from pymilvus.exceptions import (
    DataTypeNotMatchException,
    MilvusException,
    ParamError,
    PrimaryKeyException,
)
from pymilvus.orm.collection import CollectionSchema, Function, FunctionScore
from pymilvus.orm.connections import connections
from pymilvus.orm.types import DataType

from ._utils import create_connection
from .async_optimize_task import AsyncOptimizeTask
from .base import BaseMilvusClient
from .check import validate_param
from .index import IndexParam, IndexParams
from .optimize_task import OptimizeResult, ProgressStage


class AsyncMilvusClient(BaseMilvusClient):
    """AsyncMilvusClient is an EXPERIMENTAL class
    which only provides part of MilvusClient's methods"""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        self._db_name = self._extract_db_name_from_uri(uri, db_name)
        self._using = create_connection(
            uri,
            token,
            self._db_name,
            use_async=True,
            user=user,
            password=password,
            timeout=timeout,
            **kwargs,
        )
        self.is_self_hosted = bool(self.get_server_type() == "milvus")

    async def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        primary_field_name: str = "id",  # default is "id"
        id_type: str = "int",  # or "string",
        vector_field_name: str = "vector",  # default is  "vector"
        metric_type: str = "COSINE",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        schema: Optional[CollectionSchema] = None,
        index_params: Optional[IndexParams] = None,
        **kwargs,
    ):
        if schema is None:
            return await self._fast_create_collection(
                collection_name,
                dimension,
                primary_field_name=primary_field_name,
                id_type=id_type,
                vector_field_name=vector_field_name,
                metric_type=metric_type,
                auto_id=auto_id,
                timeout=timeout,
                **kwargs,
            )

        return await self._create_collection_with_schema(
            collection_name, schema, index_params, timeout=timeout, **kwargs
        )

    async def _fast_create_collection(
        self,
        collection_name: str,
        dimension: int,
        primary_field_name: str = "id",  # default is "id"
        id_type: Union[DataType, str] = DataType.INT64,  # or "string",
        vector_field_name: str = "vector",  # default is  "vector"
        metric_type: str = "COSINE",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if dimension is None:
            msg = "missing requried argument: 'dimension'"
            raise TypeError(msg)
        if "enable_dynamic_field" not in kwargs:
            kwargs["enable_dynamic_field"] = True

        schema = self.create_schema(auto_id=auto_id, **kwargs)

        if id_type in ("int", DataType.INT64):
            pk_data_type = DataType.INT64
        elif id_type in ("string", "str", DataType.VARCHAR):
            pk_data_type = DataType.VARCHAR
        else:
            raise PrimaryKeyException(message=ExceptionsMessage.PrimaryFieldType)

        pk_args = {}
        if "max_length" in kwargs and pk_data_type == DataType.VARCHAR:
            pk_args["max_length"] = kwargs["max_length"]

        schema.add_field(primary_field_name, pk_data_type, is_primary=True, **pk_args)
        vector_type = DataType.FLOAT_VECTOR
        schema.add_field(vector_field_name, vector_type, dim=dimension)
        schema.verify()

        conn = self._get_connection()
        if "consistency_level" not in kwargs:
            kwargs["consistency_level"] = DEFAULT_CONSISTENCY_LEVEL
        await conn.create_collection(
            collection_name,
            schema,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

        index_params = IndexParams()
        index_params.add_index(vector_field_name, index_type="AUTOINDEX", metric_type=metric_type)
        await self.create_index(collection_name, index_params, timeout=timeout)
        await self.load_collection(collection_name, timeout=timeout)

    async def _create_collection_with_schema(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        schema.verify()

        conn = self._get_connection()
        if "consistency_level" not in kwargs:
            kwargs["consistency_level"] = DEFAULT_CONSISTENCY_LEVEL
        await conn.create_collection(
            collection_name,
            schema,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

        if index_params:
            await self.create_index(collection_name, index_params, timeout=timeout)
            await self.load_collection(collection_name, timeout=timeout)

    async def drop_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_collection(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def truncate_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.truncate_collection(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def rename_collection(
        self,
        old_name: str,
        new_name: str,
        target_db: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.rename_collection(
            old_name,
            new_name,
            target_db,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def load_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.load_collection(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def release_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.release_collection(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def create_index(
        self,
        collection_name: str,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        validate_param("collection_name", collection_name, str)
        validate_param("index_params", index_params, IndexParams)
        if len(index_params) == 0:
            raise ParamError(message="IndexParams is empty, no index can be created")

        for index_param in index_params:
            await self._create_index(collection_name, index_param, timeout=timeout, **kwargs)

    async def _create_index(
        self,
        collection_name: str,
        index_param: IndexParam,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.create_index(
            collection_name,
            index_param.field_name,
            index_param.get_index_configs(),
            timeout=timeout,
            index_name=index_param.index_name,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_index(
        self, collection_name: str, index_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_index(
            collection_name,
            "",
            index_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def create_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.create_partition(
            collection_name,
            partition_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_partition(
            collection_name,
            partition_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def load_partitions(
        self,
        collection_name: str,
        partition_names: Union[str, List[str]],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(partition_names, str):
            partition_names = [partition_names]

        conn = self._get_connection()
        await conn.load_partitions(
            collection_name,
            partition_names,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def release_partitions(
        self,
        collection_name: str,
        partition_names: Union[str, List[str]],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(partition_names, str):
            partition_names = [partition_names]
        conn = self._get_connection()
        await conn.release_partitions(
            collection_name,
            partition_names,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def has_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ) -> bool:
        conn = self._get_connection()
        return await conn.has_partition(
            collection_name,
            partition_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def list_partitions(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> List[str]:
        conn = self._get_connection()
        return await conn.list_partitions(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        # If no data provided, we cannot input anything
        if isinstance(data, Dict):
            data = [data]

        msg = "wrong type of argument 'data',"
        msg += f"expected 'Dict' or list of 'Dict', got '{type(data).__name__}'"

        if not isinstance(data, List):
            raise TypeError(msg)

        if len(data) == 0:
            return {"insert_count": 0, "ids": []}

        conn = self._get_connection()
        # Insert into the collection.
        res = await conn.insert_rows(
            collection_name,
            data,
            partition_name=partition_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        return OmitZeroDict(
            {
                "insert_count": res.insert_count,
                "ids": res.primary_keys,
                "cost": res.cost,
            }
        )

    async def upsert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Upsert data into the collection asynchronously.

        Args:
            collection_name (str): Name of the collection to upsert into.
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (float, optional): The timeout to use, will override init timeout. Defaults
                to None.
            partition_name (str, optional): Name of the partition to upsert into.
            **kwargs (dict): Extra keyword arguments.

                * *partial_update* (bool, optional): Whether this is a partial update operation.
                    If True, only the specified fields will be updated while others remain unchanged
                    Default is False.

        Raises:
            DataNotMatchException: If the data has missing fields an exception will be thrown.
            MilvusException: General Milvus error on upsert.

        Returns:
            Dict: Number of rows that were upserted.
        """
        # If no data provided, we cannot input anything
        if isinstance(data, Dict):
            data = [data]

        msg = "wrong type of argument 'data',"
        msg += f"expected 'Dict' or list of 'Dict', got '{type(data).__name__}'"

        if not isinstance(data, List):
            raise TypeError(msg)

        if len(data) == 0:
            return {"upsert_count": 0, "ids": []}

        conn = self._get_connection()
        # Upsert into the collection.
        try:
            res = await conn.upsert_rows(
                collection_name,
                data,
                partition_name=partition_name,
                timeout=timeout,
                context=self._generate_call_context(**kwargs),
                **kwargs,
            )
        except Exception as ex:
            raise ex from ex

        return OmitZeroDict(
            {
                "upsert_count": res.upsert_count,
                "cost": res.cost,
                "ids": res.primary_keys,
            }
        )

    async def hybrid_search(
        self,
        collection_name: str,
        reqs: List[AnnSearchRequest],
        ranker: Union[BaseRanker, Function],
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[List[dict]]:
        conn = self._get_connection()
        return await conn.hybrid_search(
            collection_name,
            reqs,
            ranker,
            limit=limit,
            partition_names=partition_names,
            output_fields=output_fields,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def search(
        self,
        collection_name: str,
        data: Optional[Union[List[list], list]] = None,
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        ranker: Optional[Union[Function, FunctionScore]] = None,
        ids: Optional[Union[List[int], List[str], str, int]] = None,
        **kwargs,
    ) -> List[List[dict]]:
        conn = self._get_connection()
        return await conn.search(
            collection_name=collection_name,
            anns_field=anns_field or "",
            param=search_params or {},
            expression=filter,
            limit=limit,
            data=data,
            ids=ids,
            output_fields=output_fields,
            partition_names=partition_names,
            expr_params=kwargs.pop("filter_params", {}),
            timeout=timeout,
            ranker=ranker,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        ids: Optional[Union[List, str, int]] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        if filter and not isinstance(filter, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))

        if filter and ids is not None:
            raise ParamError(message=ExceptionsMessage.AmbiguousQueryFilterParam)

        if isinstance(ids, (int, str)):
            ids = [ids]

        conn = self._get_connection()

        if ids:
            try:
                schema_dict, _ = await conn._get_schema(
                    collection_name,
                    timeout=timeout,
                    context=self._generate_call_context(**kwargs),
                    **kwargs,
                )
            except Exception as ex:
                raise ex from ex
            filter = self._pack_pks_expr(schema_dict, ids)

        if not output_fields:
            output_fields = ["*"]

        return await conn.query(
            collection_name,
            expr=filter,
            output_fields=output_fields,
            partition_names=partition_names,
            timeout=timeout,
            expr_params=kwargs.pop("filter_params", {}),
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def get(
        self,
        collection_name: str,
        ids: Union[list, str, int],
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        if not isinstance(ids, list):
            ids = [ids]

        if len(ids) == 0:
            return []

        conn = self._get_connection()
        try:
            schema_dict, _ = await conn._get_schema(
                collection_name,
                timeout=timeout,
                context=self._generate_call_context(**kwargs),
                **kwargs,
            )
        except Exception as ex:
            raise ex from ex

        if not output_fields:
            output_fields = ["*"]

        expr = self._pack_pks_expr(schema_dict, ids)
        return await conn.query(
            collection_name,
            expr=expr,
            output_fields=output_fields,
            partition_names=partition_names,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def delete(
        self,
        collection_name: str,
        ids: Optional[Union[list, str, int]] = None,
        timeout: Optional[float] = None,
        filter: Optional[str] = None,
        partition_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, int]:
        pks = kwargs.get("pks", [])
        if isinstance(pks, (int, str)):
            pks = [pks]

        for pk in pks:
            if not isinstance(pk, (int, str)):
                msg = f"wrong type of argument pks, expect list, int or str, got '{type(pk).__name__}'"
                raise TypeError(msg)

        if ids is not None:
            if isinstance(ids, (int, str)):
                pks.append(ids)
            elif isinstance(ids, list):
                for id in ids:
                    if not isinstance(id, (int, str)):
                        msg = f"wrong type of argument ids, expect list, int or str, got '{type(id).__name__}'"
                        raise TypeError(msg)
                pks.extend(ids)
            else:
                msg = f"wrong type of argument ids, expect list, int or str, got '{type(ids).__name__}'"
                raise TypeError(msg)

        # validate ambiguous delete filter param before describe collection rpc
        if filter and len(pks) > 0:
            raise ParamError(message=ExceptionsMessage.AmbiguousDeleteFilterParam)

        expr = ""
        conn = self._get_connection()
        if len(pks) > 0:
            try:
                schema_dict, _ = await conn._get_schema(
                    collection_name,
                    timeout=timeout,
                    context=self._generate_call_context(**kwargs),
                    **kwargs,
                )
            except Exception as ex:
                raise ex from ex
            expr = self._pack_pks_expr(schema_dict, pks)
        else:
            if not isinstance(filter, str):
                raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))
            expr = filter

        ret_pks = []
        res = await conn.delete(
            collection_name=collection_name,
            expression=expr,
            partition_name=partition_name,
            expr_params=kwargs.pop("filter_params", {}),
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        if res.primary_keys:
            ret_pks.extend(res.primary_keys)

        # compatible with deletions that returns primary keys
        if ret_pks:
            return ret_pks

        return OmitZeroDict({"delete_count": res.delete_count, "cost": res.cost})

    async def describe_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> dict:
        conn = self._get_connection()
        result = await conn.describe_collection(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        # Convert internal struct_array_fields to user-friendly format
        if isinstance(result, dict) and "struct_array_fields" in result:
            converted_fields = convert_struct_fields_to_user_format(result["struct_array_fields"])
            result["fields"].extend(converted_fields)
            # Remove internal struct_array_fields from user-facing response
            result.pop("struct_array_fields")
        return result

    async def has_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> bool:
        conn = self._get_connection()
        return await conn.has_collection(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def list_collections(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_collections(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def get_collection_stats(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        stats = await conn.get_collection_stats(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        result = {stat.key: stat.value for stat in stats}
        if "row_count" in result:
            result["row_count"] = int(result["row_count"])
        return result

    async def get_partition_stats(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        stats = await conn.get_partition_stats(
            collection_name,
            partition_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        result = {stat.key: stat.value for stat in stats}
        if "row_count" in result:
            result["row_count"] = int(result["row_count"])
        return result

    async def get_load_state(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        state = await conn.get_load_state(
            collection_name,
            partition_names,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

        ret = {"state": state}
        if state == LoadState.Loading:
            progress = await conn.get_loading_progress(
                collection_name,
                partition_names,
                timeout=timeout,
                context=self._generate_call_context(**kwargs),
                **kwargs,
            )
            ret["progress"] = progress

        return ret

    async def refresh_load(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        return await conn.refresh_load(
            collection_name,
            partition_names,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def get_server_version(
        self, timeout: Optional[float] = None, detail: bool = False, **kwargs
    ) -> Union[str, Dict]:
        """Get the running server's version

        Args:
            timeout: A duration of time in seconds to allow for the RPC.
            detail: If True, return detailed server info. Defaults to False.

        Returns:
            str: Server version when detail=False.
            dict: Detailed server info when detail=True.
        """
        conn = self._get_connection()
        return await conn.get_server_version(
            timeout=timeout, detail=detail, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def describe_replica(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        return await conn.describe_replica(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def alter_collection_properties(
        self, collection_name: str, properties: dict, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.alter_collection_properties(
            collection_name,
            properties=properties,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_collection_properties(
        self,
        collection_name: str,
        property_keys: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.drop_collection_properties(
            collection_name,
            property_keys=property_keys,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def alter_collection_field(
        self,
        collection_name: str,
        field_name: str,
        field_params: dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.alter_collection_field(
            collection_name,
            field_name=field_name,
            field_params=field_params,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def add_collection_field(
        self,
        collection_name: str,
        field_name: str,
        data_type: DataType,
        desc: str = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if is_vector_type(data_type) and not kwargs.get("nullable", False):
            raise ParamError(
                message="Adding vector field to existing collection requires nullable=True"
            )
        field_schema = self.create_field_schema(field_name, data_type, desc, **kwargs)
        conn = self._get_connection()
        await conn.add_collection_field(
            collection_name,
            field_schema,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def add_collection_function(
        self, collection_name: str, function: Function, timeout: Optional[float] = None, **kwargs
    ):
        """Add a new function to the collection.

        Args:
            collection_name(``string``): The name of collection.
            function(``Function``):  The function schema.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC.
                If timeout is set to None, the client keeps waiting until the server
                responds or an error occurs.
            **kwargs (``dict``): Optional field params

        Raises:
            MilvusException: If anything goes wrong
        """
        conn = self._get_connection()
        await conn.add_collection_function(
            collection_name,
            function,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def alter_collection_function(
        self,
        collection_name: str,
        function_name: str,
        function: Function,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """Alter a function in the collection.

        Args:
            collection_name(``string``): The name of collection.
            function_name(``string``): The function name that needs to be modified
            function(``Function``):  The function schema.
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC.
                If timeout is set to None, the client keeps waiting until the server
                responds or an error occurs.
            **kwargs (``dict``): Optional field params

        Raises:
            MilvusException: If anything goes wrong
        """
        conn = self._get_connection()
        await conn.alter_collection_function(
            collection_name,
            function_name,
            function,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_collection_function(
        self, collection_name: str, function_name: str, timeout: Optional[float] = None, **kwargs
    ):
        """Drop a function from the collection.

        Args:
            collection_name(``string``): The name of collection.
            function_name(``string``): The function name that needs to be dropped
            timeout (``float``, optional): A duration of time in seconds to allow for the RPC.
                If timeout is set to None, the client keeps waiting until the server
                responds or an error occurs.
            **kwargs (``dict``): Optional field params

        Raises:
            MilvusException: If anything goes wrong
        """
        conn = self._get_connection()
        await conn.drop_collection_function(
            collection_name,
            function_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def close(self):
        await connections.async_remove_connection(self._using)

    async def list_indexes(self, collection_name: str, field_name: Optional[str] = "", **kwargs):
        conn = self._get_connection()
        indexes = await conn.list_indexes(
            collection_name, context=self._generate_call_context(**kwargs), **kwargs
        )
        index_name_list = []
        for index in indexes:
            if not index:
                continue
            if not field_name or index.field_name == field_name:
                index_name_list.append(index.index_name)
        return index_name_list

    async def describe_index(
        self, collection_name: str, index_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        return await conn.describe_index(
            collection_name,
            index_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def alter_index_properties(
        self,
        collection_name: str,
        index_name: str,
        properties: dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.alter_index_properties(
            collection_name,
            index_name,
            properties=properties,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_index_properties(
        self,
        collection_name: str,
        index_name: str,
        property_keys: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.drop_index_properties(
            collection_name,
            index_name,
            property_keys=property_keys,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def create_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.create_alias(
            collection_name,
            alias,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_alias(self, alias: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.drop_alias(
            alias, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def alter_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.alter_alias(
            collection_name,
            alias,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def describe_alias(self, alias: str, timeout: Optional[float] = None, **kwargs) -> Dict:
        conn = self._get_connection()
        return await conn.describe_alias(
            alias, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def list_aliases(
        self, collection_name: str = "", timeout: Optional[float] = None, **kwargs
    ) -> List[str]:
        conn = self._get_connection()
        return await conn.list_aliases(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def using_database(self, db_name: str, **kwargs):
        """Deprecated: Use use_database instead."""
        await self.use_database(db_name, **kwargs)

    async def use_database(self, db_name: str, **kwargs):
        """Switch to a different database. Future operations will use this database.

        Args:
            db_name (str): The name of the database to switch to.
            **kwargs: Additional keyword arguments (e.g., timeout).

        Raises:
            MilvusException: If the database does not exist (error code 800).
        """
        await self.describe_database(db_name, **kwargs)
        self._db_name = db_name

    async def create_database(
        self,
        db_name: str,
        properties: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.create_database(
            db_name=db_name,
            properties=properties,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_database(self, db_name: str, **kwargs):
        conn = self._get_connection()
        await conn.drop_database(db_name, context=self._generate_call_context(**kwargs), **kwargs)

    async def list_databases(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_database(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def describe_database(self, db_name: str, **kwargs) -> dict:
        conn = self._get_connection()
        return await conn.describe_database(
            db_name, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def alter_database_properties(self, db_name: str, properties: dict, **kwargs):
        conn = self._get_connection()
        await conn.alter_database(
            db_name, properties, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def drop_database_properties(self, db_name: str, property_keys: List[str], **kwargs):
        conn = self._get_connection()
        await conn.drop_database_properties(
            db_name, property_keys, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def create_user(
        self, user_name: str, password: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.create_user(
            user_name,
            password,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def drop_user(self, user_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.drop_user(
            user_name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def update_password(
        self,
        user_name: str,
        old_password: str,
        new_password: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.update_password(
            user_name,
            old_password,
            new_password,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def list_users(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_users(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def describe_user(
        self, user_name: str, timeout: Optional[float] = None, **kwargs
    ) -> dict:
        conn = self._get_connection()
        res = await conn.describe_user(
            user_name,
            True,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        if hasattr(res, "results") and res.results:
            user_info = UserInfo(res.results)
            if user_info.groups:
                item = user_info.groups[0]
                return {"user_name": user_name, "roles": item.roles}
        return {}

    async def create_privilege_group(
        self,
        group_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.create_privilege_group(
            group_name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def drop_privilege_group(
        self,
        group_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.drop_privilege_group(
            group_name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def list_privilege_groups(
        self,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        conn = self._get_connection()
        res = await conn.list_privilege_groups(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )
        ret = []
        for g in res:
            privileges = []
            for p in g.privileges:
                privileges.append(p.name)
            ret.append({"privilege_group": g.group_name, "privileges": privileges})
        return ret

    async def add_privileges_to_group(
        self,
        group_name: str,
        privileges: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.add_privileges_to_group(
            group_name,
            privileges,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def remove_privileges_from_group(
        self,
        group_name: str,
        privileges: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.remove_privileges_from_group(
            group_name,
            privileges,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def create_role(self, role_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.create_role(
            role_name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def drop_role(
        self, role_name: str, force_drop: bool = False, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_role(
            role_name,
            force_drop=force_drop,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def grant_role(
        self, user_name: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.grant_role(
            user_name,
            role_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def revoke_role(
        self, user_name: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.revoke_role(
            user_name,
            role_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def grant_privilege(
        self,
        role_name: str,
        object_type: str,
        privilege: str,
        object_name: str,
        db_name: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.grant_privilege(
            role_name,
            object_type,
            object_name,
            privilege,
            db_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def revoke_privilege(
        self,
        role_name: str,
        object_type: str,
        privilege: str,
        object_name: str,
        db_name: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.revoke_privilege(
            role_name,
            object_type,
            object_name,
            privilege,
            db_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def grant_privilege_v2(
        self,
        role_name: str,
        privilege: str,
        collection_name: str,
        db_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.grant_privilege_v2(
            role_name,
            privilege,
            collection_name,
            db_name=db_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def revoke_privilege_v2(
        self,
        role_name: str,
        privilege: str,
        collection_name: str,
        db_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.revoke_privilege_v2(
            role_name,
            privilege,
            collection_name,
            db_name=db_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def describe_role(
        self, role_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        context = self._generate_call_context(**kwargs)
        db_name = kwargs.pop("db_name", "")
        res = await conn.select_grant_for_one_role(
            role_name, db_name, timeout=timeout, context=context, **kwargs
        )
        ret = {}
        ret["role"] = role_name
        ret["privileges"] = [dict(i) for i in res.groups]
        return ret

    async def list_roles(self, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        res = await conn.list_roles(
            False, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

        role_info = RoleInfo(res)
        return [g.role_name for g in role_info.groups]

    async def create_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.create_resource_group(
            name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def drop_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.drop_resource_group(
            name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def update_resource_groups(
        self, configs: Dict[str, ResourceGroupConfig], timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.update_resource_groups(
            configs, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def describe_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return await conn.describe_resource_group(
            name, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def list_resource_groups(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_resource_groups(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def transfer_replica(
        self,
        source: str,
        target: str,
        collection_name: str,
        num_replica: int,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.transfer_replica(
            source,
            target,
            collection_name,
            num_replica,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def flush(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.flush(
            [collection_name],
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def flush_all(self, timeout: Optional[float] = None, **kwargs) -> None:
        """Flush all collections.

        Args:
            timeout (Optional[float]): An optional duration of time in seconds to allow for the RPC.
            **kwargs: Additional arguments.
        """
        conn = self._get_connection()
        await conn.flush_all(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def get_flush_all_state(self, timeout: Optional[float] = None, **kwargs) -> bool:
        """Get the flush all state.

        Args:
            timeout (Optional[float]): An optional duration of time in seconds to allow for the RPC.
            **kwargs: Additional arguments.

        Returns:
            bool: True if flush all operation is completed, False otherwise.
        """
        conn = self._get_connection()
        return await conn.get_flush_all_state(
            timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def list_persistent_segments(
        self,
        collection_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[SegmentInfo]:
        """List persistent segments for a collection.

        Args:
            collection_name (str): The name of the collection.
            timeout (Optional[float]): An optional duration of time in seconds to allow for the RPC.
            **kwargs: Additional arguments.

        Returns:
            List[SegmentInfo]: A list of persistent segment information.
        """
        validate_param("collection_name", collection_name, str)
        infos = await self._get_connection().get_persistent_segment_infos(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        return [
            SegmentInfo(
                segment_id=info.segmentID,
                collection_id=info.collectionID,
                collection_name=collection_name,
                num_rows=info.num_rows,
                is_sorted=info.is_sorted,
                state=info.state,
                level=info.level,
                storage_version=info.storage_version,
            )
            for info in infos
        ]

    async def compact(
        self,
        collection_name: str,
        is_clustering: Optional[bool] = False,
        is_l0: Optional[bool] = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> int:
        conn = self._get_connection()
        return await conn.compact(
            collection_name,
            is_clustering=is_clustering,
            is_l0=is_l0,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def get_compaction_state(
        self, job_id: int, timeout: Optional[float] = None, **kwargs
    ) -> str:
        conn = self._get_connection()
        result = await conn.get_compaction_state(
            job_id, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )
        return result.state_name

    async def get_compaction_plans(
        self,
        job_id: int,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """Get compaction plans for a specific job.

        Args:
            job_id (int): The ID of the compaction job.
            timeout (Optional[float]): An optional duration of time in seconds to allow for the RPC.
            **kwargs: Additional arguments.

        Returns:
            CompactionPlans: The compaction plans for the specified job.
        """
        conn = self._get_connection()
        return await conn.get_compaction_plans(
            job_id, timeout=timeout, context=self._generate_call_context(**kwargs), **kwargs
        )

    async def run_analyzer(
        self,
        texts: Union[str, List[str]],
        analyzer_params: Optional[Union[str, Dict]] = None,
        with_hash: bool = False,
        with_detail: bool = False,
        collection_name: Optional[str] = None,
        field_name: Optional[str] = None,
        analyzer_names: Optional[Union[str, List[str]]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        return await conn.run_analyzer(
            texts,
            analyzer_params=analyzer_params,
            with_hash=with_hash,
            with_detail=with_detail,
            collection_name=collection_name,
            field_name=field_name,
            analyzer_names=analyzer_names,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def update_replicate_configuration(
        self,
        clusters: Optional[List[Dict]] = None,
        cross_cluster_topology: Optional[List[Dict]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Update replication configuration across Milvus clusters.

        Args:
            clusters (List[Dict], optional): List of cluster configurations.
            Each dict should contain:
                - cluster_id (str): Unique identifier for the cluster
                - connection_param (Dict): Connection parameters with 'uri' and 'token'
                - pchannels (List[str], optional): Physical channels for the cluster

            cross_cluster_topology (List[Dict], optional): List of replication relationships.
            Each dict should contain:
                - source_cluster_id (str): ID of the source cluster
                - target_cluster_id (str): ID of the target cluster

            timeout (float, optional): An optional duration of time in seconds to allow for the RPC
            **kwargs: Additional arguments

        Returns:
            Status: The status of the operation

        Raises:
            ParamError: If neither clusters nor cross_cluster_topology is provided
            MilvusException: If the operation fails
        """
        conn = self._get_connection()
        return await conn.update_replicate_configuration(
            clusters=clusters,
            cross_cluster_topology=cross_cluster_topology,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )

    async def _is_collection_loaded(
        self, collection_name: str, timeout: Optional[float] = None
    ) -> bool:
        state_dict = await self.get_load_state(collection_name, timeout=timeout)
        return state_dict.get("state") == LoadState.Loaded

    async def _list_vector_indexes(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> List[str]:
        schema_dict = await self._get_connection()._get_schema(
            collection_name,
            timeout=timeout,
            context=self._generate_call_context(**kwargs),
            **kwargs,
        )
        vector_fields = {
            field["name"]
            for field in schema_dict.get("fields", [])
            if is_vector_type(field.get("type"))
        }

        if not vector_fields:
            return []

        all_indexes = await self.list_indexes(collection_name, **kwargs)

        vector_indexes = []
        if all_indexes:
            describe_tasks = [
                self.describe_index(collection_name, index_name, timeout=timeout, **kwargs)
                for index_name in all_indexes
            ]
            index_infos = await asyncio.gather(*describe_tasks)

            vector_indexes = [
                all_indexes[i]
                for i, index_info in enumerate(index_infos)
                if index_info and index_info.get("field_name") in vector_fields
            ]
        return vector_indexes

    async def _wait_for_indexes(
        self,
        task: AsyncOptimizeTask,
        collection_name: str,
        index_names: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        if not index_names:
            return

        task.check_cancelled()

        conn = self._get_connection()
        wait_tasks = [
            conn.wait_for_creating_index(
                collection_name, index_name, context=self._generate_call_context(**kwargs), **kwargs
            )
            for index_name in index_names
        ]

        try:
            await asyncio.wait_for(asyncio.gather(*wait_tasks), timeout=timeout)
        except asyncio.TimeoutError as e:
            raise MilvusException(
                message=f"Timeout waiting for indexes to complete on collection {collection_name}"
            ) from e

    async def _wait_for_compaction_with_cancel(
        self, task: AsyncOptimizeTask, compaction_id: int, timeout: Optional[float] = None, **kwargs
    ) -> None:
        start = time.time()
        conn = self._get_connection()
        while True:
            task.check_cancelled()

            if timeout is not None:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    raise MilvusException(
                        message=f"Timeout waiting for compaction {compaction_id} to complete"
                    )
                remaining_timeout = timeout - elapsed
            else:
                remaining_timeout = None

            state = await conn.get_compaction_state(
                compaction_id,
                timeout=remaining_timeout,
                context=self._generate_call_context(**kwargs),
                **kwargs,
            )
            if state.state == 2:
                break
            if state.state == 3:
                raise MilvusException(message=f"Compaction {compaction_id} failed")

            await asyncio.sleep(0.5)

    async def optimize(
        self,
        collection_name: str,
        target_size: Optional[str] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Union[OptimizeResult, AsyncOptimizeTask]:
        """Optimize collection to adjust segment sizes for better query performance.

        WARNING: This is a Preview version feature for non-production use only (Benchmark, POC).

        This method performs the following operations:
        1. Waits for all indexes to complete building
        2. Triggers force merge compaction with optional target size
        3. Waits for compaction to complete
        4. Waits for index rebuild to complete
        5. Refreshes collection load if collection is loaded

        Args:
            collection_name (str): The name of the collection to optimize.
            target_size (Optional[str]): Target segment size. Format: '1000MB', '1GB', '1.2gb'.
                If not provided, uses system default.
            wait (bool): Whether to wait for optimization to complete. Defaults to True.
                If False, returns an OptimizeTask for async tracking.
            timeout (Optional[float]): Maximum time in seconds to wait for optimization.
                Only applies when wait=True.
            **kwargs: Additional arguments.

        Returns:
            Union[OptimizeResult, AsyncOptimizeTask]: If wait=True, returns OptimizeResult with:
                - status: "success"
                - collection_name: str
                - compaction_id: int
                - target_size: str or None
                - progress: list of progress stages completed
            If wait=False, returns AsyncOptimizeTask for progress tracking.

        Raises:
            ParamError: If collection_name is invalid or target_size format is incorrect.
            MilvusException: If index build fails, compaction fails, or timeout occurs.

        Examples:
            Synchronous usage:
            >>> await client.optimize("my_collection", target_size="512MB")

            Asynchronous usage:
            >>> task = await client.optimize("my_collection", target_size="512MB", wait=False)
            >>> while not task.done():
            ...     print(f"Progress: {task.progress()}")
            ...     await asyncio.sleep(1)
            >>> result = task.result()

            Cancelling a task:
            >>> task = await client.optimize("my_collection", target_size="512MB", wait=False)
            >>> # ... do some work ...
            >>> if task.cancel():
            ...     print("Task cancelled successfully")
        """
        validate_param("collection_name", collection_name, str)

        task = AsyncOptimizeTask(
            collection_name=collection_name,
            target_size=target_size,
            task_timeout=timeout,
            execute_fn=self._execute_optimize,
            **kwargs,
        )
        task.start()

        if wait:
            return await task.result(timeout=timeout)

        return task

    async def _execute_optimize(
        self,
        task: AsyncOptimizeTask,
        collection_name: str,
        size_mb: Optional[int],
        timeout: Optional[float],
        **kwargs,
    ) -> OptimizeResult:
        start = time.time()

        def remaining_timeout() -> Optional[float]:
            if timeout is None:
                return None
            elapsed = time.time() - start
            return max(0, timeout - elapsed)

        task.check_cancelled()
        vector_indexes = await self._list_vector_indexes(
            collection_name, timeout=remaining_timeout(), **kwargs
        )

        task.check_cancelled()
        task.set_progress(ProgressStage.WAITING_FOR_INDEXES)
        await self._wait_for_indexes(
            task, collection_name, vector_indexes, timeout=remaining_timeout(), **kwargs
        )

        task.check_cancelled()
        task.set_progress(ProgressStage.COMPACTING)
        conn = self._get_connection()
        compaction_id = await conn.compact(
            collection_name=collection_name,
            target_size=size_mb,
            timeout=remaining_timeout(),
            **kwargs,
        )

        task.check_cancelled()
        task.set_progress(ProgressStage.WAITING_FOR_COMPACTION)
        await self._wait_for_compaction_with_cancel(
            task, compaction_id, timeout=remaining_timeout(), **kwargs
        )

        task.check_cancelled()
        task.set_progress(ProgressStage.WAITING_FOR_INDEX_REBUILD)
        await self._wait_for_indexes(
            task, collection_name, vector_indexes, timeout=remaining_timeout(), **kwargs
        )

        task.check_cancelled()
        if await self._is_collection_loaded(collection_name, timeout=remaining_timeout()):
            task.set_progress(ProgressStage.REFRESHING_LOAD)
            await self.refresh_load(collection_name, timeout=remaining_timeout(), **kwargs)

        return OptimizeResult(
            status="success",
            collection_name=collection_name,
            compaction_id=compaction_id,
            target_size=task._target_size,
            progress=task.progress_history(),
        )
