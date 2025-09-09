import logging
from typing import Dict, List, Optional, Union

from pymilvus.client.abstract import AnnSearchRequest, BaseRanker
from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.types import (
    ExceptionsMessage,
    OmitZeroDict,
    ResourceGroupConfig,
    RoleInfo,
    UserInfo,
)
from pymilvus.exceptions import (
    DataTypeNotMatchException,
    MilvusException,
    ParamError,
    PrimaryKeyException,
)
from pymilvus.orm import utility
from pymilvus.orm.collection import CollectionSchema
from pymilvus.orm.connections import connections
from pymilvus.orm.schema import FieldSchema
from pymilvus.orm.types import DataType

from ._utils import create_connection
from .check import validate_param
from .index import IndexParam, IndexParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AsyncMilvusClient:
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
        self._using = create_connection(
            uri,
            token,
            db_name,
            use_async=True,
            user=user,
            password=password,
            timeout=timeout,
            **kwargs,
        )
        self.is_self_hosted = bool(utility.get_server_type(using=self._using) == "milvus")

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
        try:
            await conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

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
        try:
            await conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

        if index_params:
            await self.create_index(collection_name, index_params, timeout=timeout)
            await self.load_collection(collection_name, timeout=timeout)

    async def drop_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_collection(collection_name, timeout=timeout, **kwargs)
        logger.debug("Successfully dropped collection: %s", collection_name)

    async def rename_collection(
        self,
        old_name: str,
        new_name: str,
        target_db: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.rename_collection(old_name, new_name, target_db, timeout=timeout, **kwargs)
        logger.debug("Successfully renamed collection from %s to %s", old_name, new_name)

    async def load_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        try:
            await conn.load_collection(collection_name, timeout=timeout, **kwargs)
        except MilvusException as ex:
            logger.error("Failed to load collection: %s", collection_name)
            raise ex from ex

    async def release_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        try:
            await conn.release_collection(collection_name, timeout=timeout, **kwargs)
        except MilvusException as ex:
            logger.error("Failed to load collection: %s", collection_name)
            raise ex from ex

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
        try:
            await conn.create_index(
                collection_name,
                index_param.field_name,
                index_param.get_index_configs(),
                timeout=timeout,
                index_name=index_param.index_name,
                **kwargs,
            )
            logger.debug("Successfully created an index on collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create an index on collection: %s", collection_name)
            raise ex from ex

    async def drop_index(
        self, collection_name: str, index_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_index(collection_name, "", index_name, timeout=timeout, **kwargs)

    async def create_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.create_partition(collection_name, partition_name, timeout=timeout, **kwargs)

    async def drop_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_partition(collection_name, partition_name, timeout=timeout, **kwargs)

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
        await conn.load_partitions(collection_name, partition_names, timeout=timeout, **kwargs)

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
        await conn.release_partitions(collection_name, partition_names, timeout=timeout, **kwargs)

    async def has_partition(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ) -> bool:
        conn = self._get_connection()
        try:
            return await conn.has_partition(
                collection_name, partition_name, timeout=timeout, **kwargs
            )
        except Exception as ex:
            logger.error(
                "Failed to check partition existence: %s.%s", collection_name, partition_name
            )
            raise ex from ex

    async def list_partitions(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> List[str]:
        conn = self._get_connection()
        try:
            return await conn.list_partitions(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to list partitions: %s", collection_name)
            raise ex from ex

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
        try:
            res = await conn.insert_rows(
                collection_name, data, partition_name=partition_name, timeout=timeout
            )
        except Exception as ex:
            raise ex from ex
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
            return {"upsert_count": 0}

        conn = self._get_connection()
        # Upsert into the collection.
        try:
            res = await conn.upsert_rows(
                collection_name, data, partition_name=partition_name, timeout=timeout, **kwargs
            )
        except Exception as ex:
            raise ex from ex

        return OmitZeroDict(
            {
                "upsert_count": res.upsert_count,
                "cost": res.cost,
            }
        )

    async def hybrid_search(
        self,
        collection_name: str,
        reqs: List[AnnSearchRequest],
        ranker: BaseRanker,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[List[dict]]:
        conn = self._get_connection()
        try:
            res = await conn.hybrid_search(
                collection_name,
                reqs,
                ranker,
                limit=limit,
                partition_names=partition_names,
                output_fields=output_fields,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to hybrid search collection: %s", collection_name)
            raise ex from ex
        return res

    async def search(
        self,
        collection_name: str,
        data: Union[List[list], list],
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        anns_field: Optional[str] = None,
        **kwargs,
    ) -> List[List[dict]]:
        conn = self._get_connection()
        try:
            res = await conn.search(
                collection_name,
                data,
                anns_field or "",
                search_params or {},
                expression=filter,
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                expr_params=kwargs.pop("filter_params", {}),
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to search collection: %s", collection_name)
            raise ex from ex
        return res

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
                schema_dict = await conn.describe_collection(
                    collection_name, timeout=timeout, **kwargs
                )
            except Exception as ex:
                logger.error("Failed to describe collection: %s", collection_name)
                raise ex from ex
            filter = self._pack_pks_expr(schema_dict, ids)

        if not output_fields:
            output_fields = ["*"]

        try:
            res = await conn.query(
                collection_name,
                expr=filter,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                expr_params=kwargs.pop("filter_params", {}),
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to query collection: %s", collection_name)
            raise ex from ex

        return res

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
            schema_dict = await conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

        if not output_fields:
            output_fields = ["*"]

        expr = self._pack_pks_expr(schema_dict, ids)
        try:
            res = await conn.query(
                collection_name,
                expr=expr,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to get collection: %s", collection_name)
            raise ex from ex

        return res

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
                schema_dict = await conn.describe_collection(
                    collection_name, timeout=timeout, **kwargs
                )
            except Exception as ex:
                logger.error("Failed to describe collection: %s", collection_name)
                raise ex from ex
            expr = self._pack_pks_expr(schema_dict, pks)
        else:
            if not isinstance(filter, str):
                raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))
            expr = filter

        ret_pks = []
        try:
            res = await conn.delete(
                collection_name=collection_name,
                expression=expr,
                partition_name=partition_name,
                expr_params=kwargs.pop("filter_params", {}),
                timeout=timeout,
                **kwargs,
            )
            if res.primary_keys:
                ret_pks.extend(res.primary_keys)
        except Exception as ex:
            logger.error("Failed to delete primary keys in collection: %s", collection_name)
            raise ex from ex

        # compatible with deletions that returns primary keys
        if ret_pks:
            return ret_pks

        return OmitZeroDict({"delete_count": res.delete_count, "cost": res.cost})

    async def describe_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> dict:
        conn = self._get_connection()
        try:
            return await conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

    async def has_collection(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> bool:
        conn = self._get_connection()
        try:
            return await conn.has_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to check collection existence: %s", collection_name)
            raise ex from ex

    async def list_collections(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        try:
            return await conn.list_collections(timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to list collections")
            raise ex from ex

    async def get_collection_stats(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        try:
            stats = await conn.get_collection_stats(collection_name, timeout=timeout, **kwargs)
            result = {stat.key: stat.value for stat in stats}
            if "row_count" in result:
                result["row_count"] = int(result["row_count"])
        except Exception as ex:
            logger.error("Failed to get collection stats: %s", collection_name)
            raise ex from ex
        else:
            return result

    async def get_partition_stats(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        try:
            stats = await conn.get_partition_stats(
                collection_name, partition_name, timeout=timeout, **kwargs
            )
            result = {stat.key: stat.value for stat in stats}
            if "row_count" in result:
                result["row_count"] = int(result["row_count"])
        except Exception as ex:
            logger.error("Failed to get partition stats: %s.%s", collection_name, partition_name)
            raise ex from ex
        else:
            return result

    async def get_load_state(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        try:
            return await conn.get_load_state(
                collection_name, partition_names, timeout=timeout, **kwargs
            )
        except Exception as ex:
            logger.error("Failed to get load state: %s", collection_name)
            raise ex from ex

    async def refresh_load(
        self,
        collection_name: str,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        try:
            return await conn.refresh_load(
                collection_name, partition_names, timeout=timeout, **kwargs
            )
        except Exception as ex:
            logger.error("Failed to refresh load: %s", collection_name)
            raise ex from ex

    async def get_server_version(self, timeout: Optional[float] = None, **kwargs) -> str:
        conn = self._get_connection()
        try:
            return await conn.get_server_version(timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to get server version")
            raise ex from ex

    async def describe_replica(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        try:
            return await conn.describe_replica(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe replica: %s", collection_name)
            raise ex from ex

    async def alter_collection_properties(
        self, collection_name: str, properties: dict, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        try:
            await conn.alter_collection_properties(
                collection_name,
                properties=properties,
                timeout=timeout,
                **kwargs,
            )
            logger.debug("Successfully altered collection properties: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to alter collection properties: %s", collection_name)
            raise ex from ex

    async def drop_collection_properties(
        self,
        collection_name: str,
        property_keys: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        try:
            await conn.drop_collection_properties(
                collection_name, property_keys=property_keys, timeout=timeout, **kwargs
            )
            logger.debug("Successfully dropped collection properties: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to drop collection properties: %s", collection_name)
            raise ex from ex

    async def alter_collection_field(
        self,
        collection_name: str,
        field_name: str,
        field_params: dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        try:
            await conn.alter_collection_field(
                collection_name,
                field_name=field_name,
                field_params=field_params,
                timeout=timeout,
                **kwargs,
            )
            logger.debug(
                "Successfully altered collection field properties: %s.%s",
                collection_name,
                field_name,
            )
        except Exception as ex:
            logger.error(
                "Failed to alter collection field properties: %s.%s", collection_name, field_name
            )
            raise ex from ex

    async def add_collection_field(
        self,
        collection_name: str,
        field_name: str,
        data_type: DataType,
        desc: str = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        field_schema = self.create_field_schema(field_name, data_type, desc, **kwargs)
        conn = self._get_connection()
        try:
            await conn.add_collection_field(
                collection_name,
                field_schema,
                timeout=timeout,
                **kwargs,
            )
            logger.debug("Successfully added collection field: %s.%s", collection_name, field_name)
        except Exception as ex:
            logger.error("Failed to add collection field: %s.%s", collection_name, field_name)
            raise ex from ex

    @classmethod
    def create_schema(cls, **kwargs):
        kwargs["check_fields"] = False  # do not check fields for now
        return CollectionSchema([], **kwargs)

    @classmethod
    def create_field_schema(
        cls, name: str, data_type: DataType, desc: str = "", **kwargs
    ) -> FieldSchema:
        return FieldSchema(name, data_type, desc, **kwargs)

    @classmethod
    def prepare_index_params(cls, field_name: str = "", **kwargs) -> IndexParams:
        index_params = IndexParams()
        if field_name:
            validate_param("field_name", field_name, str)
            index_params.add_index(field_name, **kwargs)
        return index_params

    async def close(self):
        await connections.async_remove_connection(self._using)

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    def _extract_primary_field(self, schema_dict: Dict) -> dict:
        fields = schema_dict.get("fields", [])
        if not fields:
            return {}

        for field_dict in fields:
            if field_dict.get("is_primary", None) is not None:
                return field_dict

        return {}

    def _pack_pks_expr(self, schema_dict: Dict, pks: List) -> str:
        primary_field = self._extract_primary_field(schema_dict)
        pk_field_name = primary_field["name"]
        data_type = primary_field["type"]

        # Varchar pks need double quotes around the values
        if data_type == DataType.VARCHAR:
            ids = ["'" + str(entry) + "'" for entry in pks]
            expr = f"""{pk_field_name} in [{','.join(ids)}]"""
        else:
            ids = [str(entry) for entry in pks]
            expr = f"{pk_field_name} in [{','.join(ids)}]"
        return expr

    async def list_indexes(self, collection_name: str, field_name: Optional[str] = "", **kwargs):
        conn = self._get_connection()
        indexes = await conn.list_indexes(collection_name, **kwargs)
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
        return await conn.describe_index(collection_name, index_name, timeout=timeout, **kwargs)

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
            collection_name, index_name, properties=properties, timeout=timeout, **kwargs
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
            collection_name, index_name, property_keys=property_keys, timeout=timeout, **kwargs
        )

    async def create_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.create_alias(collection_name, alias, timeout=timeout, **kwargs)

    async def drop_alias(self, alias: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.drop_alias(alias, timeout=timeout, **kwargs)

    async def alter_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.alter_alias(collection_name, alias, timeout=timeout, **kwargs)

    async def describe_alias(self, alias: str, timeout: Optional[float] = None, **kwargs) -> Dict:
        conn = self._get_connection()
        return await conn.describe_alias(alias, timeout=timeout, **kwargs)

    async def list_aliases(
        self, collection_name: str = "", timeout: Optional[float] = None, **kwargs
    ) -> List[str]:
        conn = self._get_connection()
        return await conn.list_aliases(collection_name, timeout=timeout, **kwargs)

    def using_database(self, db_name: str, **kwargs):
        conn = self._get_connection()
        conn.reset_db_name(db_name)

    def use_database(self, db_name: str, **kwargs):
        conn = self._get_connection()
        conn.reset_db_name(db_name)

    async def create_database(
        self,
        db_name: str,
        properties: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.create_database(
            db_name=db_name, properties=properties, timeout=timeout, **kwargs
        )

    async def drop_database(self, db_name: str, **kwargs):
        conn = self._get_connection()
        await conn.drop_database(db_name, **kwargs)

    async def list_databases(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_database(timeout=timeout, **kwargs)

    async def describe_database(self, db_name: str, **kwargs) -> dict:
        conn = self._get_connection()
        return await conn.describe_database(db_name, **kwargs)

    async def alter_database_properties(self, db_name: str, properties: dict, **kwargs):
        conn = self._get_connection()
        await conn.alter_database(db_name, properties, **kwargs)

    async def drop_database_properties(self, db_name: str, property_keys: List[str], **kwargs):
        conn = self._get_connection()
        await conn.drop_database_properties(db_name, property_keys, **kwargs)

    async def create_user(
        self, user_name: str, password: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.create_user(user_name, password, timeout=timeout, **kwargs)

    async def drop_user(self, user_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.drop_user(user_name, timeout=timeout, **kwargs)

    async def update_password(
        self,
        user_name: str,
        old_password: str,
        new_password: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.update_password(user_name, old_password, new_password, timeout=timeout, **kwargs)

    async def list_users(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_users(timeout=timeout, **kwargs)

    async def describe_user(
        self, user_name: str, timeout: Optional[float] = None, **kwargs
    ) -> dict:
        conn = self._get_connection()
        try:
            res = await conn.describe_user(user_name, True, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe user: %s", user_name)
            raise ex from ex
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
        await conn.create_privilege_group(group_name, timeout=timeout, **kwargs)

    async def drop_privilege_group(
        self,
        group_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.drop_privilege_group(group_name, timeout=timeout, **kwargs)

    async def list_privilege_groups(
        self,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        conn = self._get_connection()
        try:
            res = await conn.list_privilege_groups(timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to list privilege groups")
            raise ex from ex
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
        await conn.add_privileges_to_group(group_name, privileges, timeout=timeout, **kwargs)

    async def remove_privileges_from_group(
        self,
        group_name: str,
        privileges: List[str],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        await conn.remove_privileges_from_group(group_name, privileges, timeout=timeout, **kwargs)

    async def create_role(self, role_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.create_role(role_name, timeout=timeout, **kwargs)

    async def drop_role(
        self, role_name: str, force_drop: bool = False, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.drop_role(role_name, force_drop=force_drop, timeout=timeout, **kwargs)

    async def grant_role(
        self, user_name: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.grant_role(user_name, role_name, timeout=timeout, **kwargs)

    async def revoke_role(
        self, user_name: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.revoke_role(user_name, role_name, timeout=timeout, **kwargs)

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
            role_name, object_type, object_name, privilege, db_name, timeout=timeout, **kwargs
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
            role_name, object_type, object_name, privilege, db_name, timeout=timeout, **kwargs
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
            **kwargs,
        )

    async def describe_role(
        self, role_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        db_name = kwargs.pop("db_name", "")
        try:
            res = await conn.select_grant_for_one_role(
                role_name, db_name, timeout=timeout, **kwargs
            )
        except Exception as ex:
            logger.error("Failed to describe role: %s", role_name)
            raise ex from ex
        ret = {}
        ret["role"] = role_name
        ret["privileges"] = [dict(i) for i in res.groups]
        return ret

    async def list_roles(self, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        try:
            res = await conn.list_roles(False, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to list roles")
            raise ex from ex

        role_info = RoleInfo(res)
        return [g.role_name for g in role_info.groups]

    async def create_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.create_resource_group(name, timeout=timeout, **kwargs)

    async def drop_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        await conn.drop_resource_group(name, timeout=timeout, **kwargs)

    async def update_resource_groups(
        self, configs: Dict[str, ResourceGroupConfig], timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        await conn.update_resource_groups(configs, timeout=timeout, **kwargs)

    async def describe_resource_group(self, name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return await conn.describe_resource_group(name, timeout=timeout, **kwargs)

    async def list_resource_groups(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        conn = self._get_connection()
        return await conn.list_resource_groups(timeout=timeout, **kwargs)

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
            source, target, collection_name, num_replica, timeout=timeout, **kwargs
        )

    async def flush(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        try:
            await conn.flush([collection_name], timeout=timeout, **kwargs)
            logger.debug("Successfully flushed collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to flush collection: %s", collection_name)
            raise ex from ex

    async def compact(
        self,
        collection_name: str,
        is_clustering: Optional[bool] = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> int:
        conn = self._get_connection()
        try:
            compaction_id = await conn.compact(
                collection_name, is_clustering=is_clustering, timeout=timeout, **kwargs
            )
            logger.debug("Successfully started compaction for collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to compact collection: %s", collection_name)
            raise ex from ex
        else:
            return compaction_id

    async def get_compaction_state(
        self, job_id: int, timeout: Optional[float] = None, **kwargs
    ) -> str:
        conn = self._get_connection()
        try:
            result = await conn.get_compaction_state(job_id, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to get compaction state for job: %s", job_id)
            raise ex from ex
        else:
            return result.state_name

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
        try:
            return await conn.run_analyzer(
                texts,
                analyzer_params=analyzer_params,
                with_hash=with_hash,
                with_detail=with_detail,
                collection_name=collection_name,
                field_name=field_name,
                analyzer_names=analyzer_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to run analyzer")
            raise ex from ex
