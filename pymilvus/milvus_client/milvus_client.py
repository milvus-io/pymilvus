"""MilvusClient for dealing with simple workflows."""

import logging
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.types import ExceptionsMessage, LoadState
from pymilvus.exceptions import (
    DataTypeNotMatchException,
    MilvusException,
    ParamError,
    PrimaryKeyException,
)
from pymilvus.orm import utility
from pymilvus.orm.collection import CollectionSchema
from pymilvus.orm.connections import connections
from pymilvus.orm.types import DataType

from .index import IndexParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MilvusClient:
    """The Milvus Client"""

    # pylint: disable=logging-too-many-args, too-many-instance-attributes, import-outside-toplevel

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
        """A client for the common Milvus use case.

        This client attempts to hide away the complexity of using Pymilvus. In a lot ofcases what
        the user wants is a simple wrapper that supports adding data, deleting data, and searching.

        Args:
            uri (str, optional): The connection address to use to connect to the
                instance. Defaults to "http://localhost:19530". Another example:
                "https://username:password@in01-12a.aws-us-west-2.vectordb.zillizcloud.com:19538
            timeout (float, optional): What timeout to use for function calls. Defaults
                to None.
        """
        # Optionial TQDM import
        try:
            import tqdm

            self.tqdm = tqdm.tqdm
        except ImportError:
            logger.debug("tqdm not found")
            self.tqdm = lambda x, disable: x

        self._using = self._create_connection(
            uri, user, password, db_name, token, timeout=timeout, **kwargs
        )
        self.is_self_hosted = bool(utility.get_server_type(using=self._using) == "milvus")

    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        primary_field_name: str = "id",  # default is "id"
        id_type: str = "int",  # or "string",
        vector_field_name: str = "vector",  # default is  "vector"
        metric_type: str = "IP",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        schema: Optional[CollectionSchema] = None,
        index_params: Optional[IndexParams] = None,
        **kwargs,
    ):
        if schema is None:
            return self._fast_create_collection(
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

        return self._create_collection_with_schema(
            collection_name, schema, index_params, timeout=timeout, **kwargs
        )

    def _fast_create_collection(
        self,
        collection_name: str,
        dimension: int,
        primary_field_name: str = "id",  # default is "id"
        id_type: str = "int",  # or "string",
        vector_field_name: str = "vector",  # default is  "vector"
        metric_type: str = "IP",
        auto_id: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if "enable_dynamic_field" not in kwargs:
            kwargs["enable_dynamic_field"] = True

        schema = self.create_schema(auto_id=auto_id, **kwargs)

        if id_type == "int":
            pk_data_type = DataType.INT64
        elif id_type in ("string", "str"):
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
            conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

        index_params = self.prepare_index_params()
        index_type = ""
        index_name = ""
        index_params.add_index(
            vector_field_name, index_type, index_name, metric_type=metric_type, params={}
        )
        self.create_index(collection_name, index_params, timeout=timeout)
        self.load_collection(collection_name, timeout=timeout)

    def create_index(
        self,
        collection_name: str,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        for index_param in index_params:
            self._create_index(collection_name, index_param, timeout=timeout, **kwargs)

    def _create_index(
        self,
        collection_name: str,
        index_param: Dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        try:
            conn.create_index(
                collection_name,
                index_param["field_name"],
                index_param.get("params", {}),
                index_name=index_param.get("index_name", ""),
                timeout=timeout,
                **kwargs,
            )
            logger.debug("Successfully created an index on collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create an index on collection: %s", collection_name)
            raise ex from ex

    def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Insert data into the collection.

        If the Milvus Client was initiated without an existing Collection, the first dict passed
        in will be used to initiate the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (float, optional): The timeout to use, will override init timeout. Defaults
                to None.

        Raises:
            DataNotMatchException: If the data has missing fields an exception will be thrown.
            MilvusException: General Milvus error on insert.

        Returns:
            Dict: Number of rows that were inserted.
        """
        # If no data provided, we cannot input anything
        if isinstance(data, Dict):
            data = [data]

        if len(data) == 0:
            return {"insert_count": 0}

        conn = self._get_connection()
        # Insert into the collection.
        try:
            res = conn.insert_rows(
                collection_name, data, partition_name=partition_name, timeout=timeout
            )
        except Exception as ex:
            raise ex from ex
        return {"insert_count": res.insert_count}

    def upsert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        timeout: Optional[float] = None,
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Insert data into the collection.

        If the Milvus Client was initiated without an existing Collection, the first dict passed
        in will be used to initiate the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (float, optional): The timeout to use, will override init timeout. Defaults
                to None.

        Raises:
            DataNotMatchException: If the data has missing fields an exception will be thrown.
            MilvusException: General Milvus error on insert.

        Returns:
            List[Union[str, int]]: A list of primary keys that were inserted.
        """
        # If no data provided, we cannot input anything
        if isinstance(data, Dict):
            data = [data]

        if len(data) == 0:
            return {"upsert_count": 0}

        conn = self._get_connection()
        # Upsert into the collection.
        try:
            res = conn.upsert_rows(
                collection_name, data, partition_name=partition_name, timeout=timeout, **kwargs
            )
        except Exception as ex:
            raise ex from ex

        return {"upsert_count": res.upsert_count}

    def search(
        self,
        collection_name: str,
        data: Union[List[list], list],
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[List[dict]]:
        """Search for a query vector/vectors.

        In order for the search to process, a collection needs to have been either provided
        at init or data needs to have been inserted.

        Args:
            data (Union[List[list], list]): The vector/vectors to search.
            top_k (int, optional): How many results to return per search. Defaults to 10.
            filter(str, optional): A filter to use for the search. Defaults to None.
            output_fields (List[str], optional): List of which field values to return. If None
                specified, only primary fields including distances will be returned.
            search_params (dict, optional): The search params to use for the search.
            timeout (float, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.

        Raises:
            ValueError: The collection being searched doesnt exist. Need to insert data first.

        Returns:
            List[List[dict]]: A nested list of dicts containing the result data. Embeddings are
                not included in the result data.
        """

        conn = self._get_connection()
        try:
            res = conn.search(
                collection_name,
                data,
                "",
                search_params or {},
                expression=filter,
                limit=limit,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to search collection: %s", collection_name)
            raise ex from ex

        ret = []
        for hits in res:
            query_result = []
            for hit in hits:
                query_result.append(hit.to_dict())
            ret.append(query_result)

        return ret

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        ids: Optional[Union[List, str, int]] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        """Query for entries in the Collection.

        Args:
            filter (str): The filter to use for the query.
            return_fields (List[str], optional): List of which field values to return. If None
                specified, all fields excluding vector field will be returned.
            partitions (List[str], optional): Which partitions to perform query. Defaults to None.
            timeout (float, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.

        Raises:
            ValueError: Missing collection.

        Returns:
            List[dict]: A list of result dicts, vectors are not included.
        """
        if filter and not isinstance(filter, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))

        if filter and ids is not None:
            raise ParamError(message=ExceptionsMessage.AmbiguousDeleteFilterParam)

        if isinstance(ids, (int, str)):
            ids = [ids]

        conn = self._get_connection()
        try:
            schema_dict = conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

        if ids:
            filter = self._pack_pks_expr(schema_dict, ids)

        if not output_fields:
            output_fields = ["*"]
            vec_field_name = self._get_vector_field_name(schema_dict)
            if vec_field_name:
                output_fields.append(vec_field_name)

        try:
            res = conn.query(
                collection_name,
                expr=filter,
                output_fields=output_fields,
                partition_names=partition_names,
                timeout=timeout,
                **kwargs,
            )
        except Exception as ex:
            logger.error("Failed to query collection: %s", collection_name)
            raise ex from ex

        return res

    def get(
        self,
        collection_name: str,
        ids: Union[list, str, int],
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        partition_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[dict]:
        """Grab the inserted vectors using the primary key from the Collection.

        Due to current implementations, grabbing a large amount of vectors is slow.

        Args:
            ids (str): The pk's to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
            timeout (float, optional): Timeout to use, overides the client level assigned at
                init. Defaults to None.

        Raises:
            ValueError: Missing collection.

        Returns:
            List[dict]: A list of result dicts with keys {pk_field, vector_field}
        """
        if not isinstance(ids, list):
            ids = [ids]

        if len(ids) == 0:
            return []

        conn = self._get_connection()
        try:
            schema_dict = conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

        if not output_fields:
            output_fields = ["*"]
            vec_field_name = self._get_vector_field_name(schema_dict)
            if vec_field_name:
                output_fields.append(vec_field_name)

        expr = self._pack_pks_expr(schema_dict, ids)
        try:
            res = conn.query(
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

    def delete(
        self,
        collection_name: str,
        ids: Optional[Union[list, str, int]] = None,
        timeout: Optional[float] = None,
        filter: Optional[str] = "",
        partition_name: Optional[str] = "",
        **kwargs,
    ) -> Dict:
        """Delete entries in the collection by their pk.

        Delete all the entries based on the pk. If unsure of pk you can first query the collection
        to grab the corresponding data. Then you can delete using the pk_field.


        Starting from version 2.3.2, Milvus no longer includes the primary keys in the result
        when processing the delete operation on expressions.
        This change is due to the large amount of data involved.
        The delete interface no longer returns any results.
        If no exceptions are thrown, it indicates a successful deletion.
        However, for backward compatibility, If the primary_keys returned from old
        Milvus(previous 2.3.2) is not empty, the list of primary keys is still returned.

        Args:
            ids (list, str, int): The pk's to delete. Depending on pk_field type it can be int
                or str or alist of either. Default to None.
            filter(str, optional): A filter to use for the deletion. Defaults to empty.
            timeout (int, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.

        Returns:
            Dict: Number of rows that were deleted.
        """
        pks = kwargs.get("pks", [])
        if isinstance(pks, (int, str)):
            pks = [pks]

        if ids:
            if isinstance(ids, (int, str)):
                pks.append(ids)
            elif isinstance(ids, list):
                pks.extend(ids)

        expr = ""
        conn = self._get_connection()
        if pks:
            try:
                schema_dict = conn.describe_collection(collection_name, timeout=timeout, **kwargs)
            except Exception as ex:
                logger.error("Failed to describe collection: %s", collection_name)
                raise ex from ex

            expr = self._pack_pks_expr(schema_dict, pks)

        if filter:
            if expr:
                raise ParamError(message=ExceptionsMessage.AmbiguousDeleteFilterParam)

            if not isinstance(filter, str):
                raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))

            expr = filter

        ret_pks = []
        try:
            res = conn.delete(collection_name, expr, partition_name, timeout=timeout, **kwargs)
            if res.primary_keys:
                ret_pks.extend(res.primary_keys)
        except Exception as ex:
            logger.error("Failed to delete primary keys in collection: %s", collection_name)
            raise ex from ex

        if ret_pks:
            return ret_pks

        return {"delete_count": res.delete_count}

    def get_collection_stats(self, collection_name: str, timeout: Optional[float] = None) -> Dict:
        conn = self._get_connection()
        stats = conn.get_collection_stats(collection_name, timeout=timeout)
        result = {stat.key: stat.value for stat in stats}
        result["row_count"] = int(result["row_count"])
        return result

    def describe_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return conn.describe_collection(collection_name, timeout=timeout, **kwargs)

    def has_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return conn.has_collection(collection_name, timeout=timeout, **kwargs)

    def list_collections(self, **kwargs):
        conn = self._get_connection()
        return conn.list_collections(**kwargs)

    def drop_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Delete the collection stored in this object"""
        conn = self._get_connection()
        conn.drop_collection(collection_name, timeout=timeout, **kwargs)

    def rename_collection(
        self,
        old_name: str,
        new_name: str,
        target_db: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        conn.rename_collections(old_name, new_name, target_db, timeout=timeout, **kwargs)

    @classmethod
    def create_schema(cls, **kwargs):
        kwargs["check_fields"] = False  # do not check fields for now
        return CollectionSchema([], **kwargs)

    @classmethod
    def prepare_index_params(cls):
        return IndexParams()

    def _create_collection_with_schema(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_params: IndexParams,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        schema.verify()
        if kwargs.get("auto_id", False):
            schema.auto_id = True
        if kwargs.get("enable_dynamic_field", False):
            schema.enable_dynamic_field = True
        schema.verify()

        conn = self._get_connection()
        if "consistency_level" not in kwargs:
            kwargs["consistency_level"] = DEFAULT_CONSISTENCY_LEVEL
        try:
            conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

        if index_params:
            self.create_index(collection_name, index_params, timeout=timeout)
            self.load_collection(collection_name, timeout=timeout)

    def close(self):
        connections.disconnect(self._using)

    def _get_connection(self):
        return connections._fetch_handler(self._using)

    def _create_connection(
        self,
        uri: str,
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        **kwargs,
    ) -> str:
        """Create the connection to the Milvus server."""
        # TODO: Implement reuse with new uri style
        using = uuid4().hex
        try:
            connections.connect(using, user, password, db_name, token, uri=uri, **kwargs)
        except Exception as ex:
            logger.error("Failed to create new connection using: %s", using)
            raise ex from ex
        else:
            logger.debug("Created new connection using: %s", using)
            return using

    def _extract_primary_field(self, schema_dict: Dict) -> dict:
        fields = schema_dict.get("fields", [])
        if not fields:
            return {}

        for field_dict in fields:
            if field_dict.get("is_primary", None) is not None:
                return field_dict

        return {}

    def _get_vector_field_name(self, schema_dict: Dict):
        fields = schema_dict.get("fields", [])
        if not fields:
            return {}

        for field_dict in fields:
            if field_dict.get("type", None) == DataType.FLOAT_VECTOR:
                return field_dict.get("name", "")
        return ""

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

    def load_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Loads the collection."""
        conn = self._get_connection()
        try:
            conn.load_collection(collection_name, timeout=timeout, **kwargs)
        except MilvusException as ex:
            logger.error(
                "Failed to load collection: %s",
                collection_name,
            )
            raise ex from ex

    def release_collection(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        try:
            conn.release_collection(collection_name, timeout=timeout, **kwargs)
        except MilvusException as ex:
            logger.error(
                "Failed to load collection: %s",
                collection_name,
            )
            raise ex from ex

    def get_load_state(
        self,
        collection_name: str,
        partition_name: Optional[str] = "",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        conn = self._get_connection()
        partition_names = None
        if partition_name:
            partition_names = [partition_name]
        try:
            state = conn.get_load_state(collection_name, partition_names, tiemout=timeout, **kwargs)
        except Exception as ex:
            raise ex from ex

        ret = {"state": state}
        if state == LoadState.Loading:
            progress = conn.get_loading_progress(collection_name, partition_names, timeout=timeout)
            ret["progress"] = progress

        return ret

    def refresh_load(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        kwargs.pop("_refresh", None)
        conn = self._get_connection()
        conn.load_collection(collection_name, timeout=timeout, _refresh=True, **kwargs)

    def list_indexes(self, collection_name: str, field_name: Optional[str] = "", **kwargs):
        """List all indexes of collection. If `field_name` is not specified,
            return all the indexes of this collection, otherwise this interface will return
            all indexes on this field of the collection.

        :param collection_name: The name of collection.
        :type  collection_name: str

        :param field_name: The name of field.  If no field name is specified, all indexes
                of this collection will be returned.

        :return: The name list of all indexes.
        :rtype: str list
        """
        conn = self._get_connection()
        indexes = conn.list_indexes(collection_name, **kwargs)
        index_name_list = []
        for index in indexes:
            if not index:
                continue
            if not field_name or index.field_name == field_name:
                index_name_list.append(index.index_name)
        return index_name_list

    def drop_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        conn.drop_index(collection_name, "", index_name, timeout=timeout, **kwargs)

    def describe_index(
        self,
        collection_name: str,
        index_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        conn = self._get_connection()
        return conn.describe_index(collection_name, index_name, timeout=timeout, **kwargs)

    def create_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        conn.create_partition(collection_name, partition_name, timeout=timeout, **kwargs)

    def drop_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        conn.drop_partition(collection_name, partition_name, timeout=timeout, **kwargs)

    def has_partition(
        self,
        collection_name: str,
        partition_name: str,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> bool:
        conn = self._get_connection()
        return conn.has_partition(collection_name, partition_name, timeout=timeout, **kwargs)

    def list_partitions(
        self, collection_name: str, timeout: Optional[float] = None, **kwargs
    ) -> List[str]:
        conn = self._get_connection()
        return conn.list_partitions(collection_name, timeout=timeout, **kwargs)

    def load_partitions(
        self,
        collection_name: str,
        partition_names: Union[str, List[str]],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(partition_names, str):
            partition_names = [partition_names]

        conn = self._get_connection()
        conn.load_partitions(collection_name, partition_names, timeout=timeout, **kwargs)

    def release_partitions(
        self,
        collection_name: str,
        partition_names: Union[str, List[str]],
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if isinstance(partition_names, str):
            partition_names = [partition_names]
        conn = self._get_connection()
        conn.release_partitions(collection_name, partition_names, timeout=timeout, **kwargs)

    def get_partition_stats(
        self, collection_name: str, partition_name: str, timeout: Optional[float] = None, **kwargs
    ) -> Dict:
        conn = self._get_connection()
        return conn.get_partition_stats(collection_name, partition_name, timeout=timeout, **kwargs)

    def create_user(self, user_name: str, password: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return conn.create_user(user_name, password, timeout=timeout, **kwargs)

    def drop_user(self, user_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return conn.delete_user(user_name, timeout=timeout, **kwargs)

    def update_password(
        self,
        user_name: str,
        old_password: str,
        new_password: str,
        reset_connection: Optional[bool] = False,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        conn = self._get_connection()
        conn.update_password(user_name, old_password, new_password, timeout=timeout, **kwargs)
        if reset_connection:
            conn._setup_authorization_interceptor(user_name, new_password, None)
            conn._setup_grpc_channel()

    def list_users(self, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        return conn.list_usernames(timeout=timeout, **kwargs)

    def describe_user(self, user_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        try:
            res = conn.select_one_user(user_name, True, timeout=timeout, **kwargs)
        except Exception as ex:
            raise ex from ex
        if res.groups:
            item = res.groups[0]
            return {
                "user_name": user_name,
                "roles": item.roles,
            }
        return {}

    def grant_role(self, user_name: str, role_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        conn.add_user_to_role(user_name, role_name, timeout=timeout, **kwargs)

    def revoke_role(
        self, user_name: str, role_name: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        conn.remove_user_from_role(user_name, role_name, timeout=timeout, **kwargs)

    def create_role(self, role_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        conn.create_role(role_name, timeout=timeout, **kwargs)

    def drop_role(self, role_name: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        conn.drop_role(role_name, timeout=timeout, **kwargs)

    def describe_role(
        self, role_name: str, timeout: Optional[float] = None, **kwargs
    ) -> List[Dict]:
        conn = self._get_connection()
        db_name = kwargs.pop("db_name", "")
        try:
            res = conn.select_grant_for_one_role(role_name, db_name, timeout=timeout, **kwargs)
        except Exception as ex:
            raise ex from ex
        return [dict(i) for i in res.groups]

    def list_roles(self, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        try:
            res = conn.select_all_role(False, timeout=timeout, **kwargs)
        except Exception as ex:
            raise ex from ex

        groups = res.groups
        return [g.role_name for g in groups]

    def grant_privilege(
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
        conn.grant_privilege(
            role_name, object_type, object_name, privilege, db_name, timeout=timeout, **kwargs
        )

    def revoke_privilege(
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
        conn.revoke_privilege(
            role_name, object_type, object_name, privilege, db_name, timeout=timeout, **kwargs
        )

    def create_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        conn.create_alias(collection_name, alias, tiemout=timeout, **kwargs)

    def drop_alias(self, alias: str, timeout: Optional[float] = None, **kwargs):
        conn = self._get_connection()
        conn.drop_alias(alias, tiemout=timeout, **kwargs)

    def alter_alias(
        self, collection_name: str, alias: str, timeout: Optional[float] = None, **kwargs
    ):
        conn = self._get_connection()
        conn.alter_alias(collection_name, alias, tiemout=timeout, **kwargs)

    def describe_alias(self, alias: str, timeout: Optional[float] = None, **kwargs) -> Dict:
        pass

    def list_aliases(self, timeout: Optional[float] = None, **kwargs) -> List[str]:
        pass

    def using_database(self, db_name: str, **kwargs):
        conn = self._get_connection()
        conn.reset_db_name(db_name)
