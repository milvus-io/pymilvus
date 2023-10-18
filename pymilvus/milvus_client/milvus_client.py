"""MilvusClient for dealing with simple workflows."""
import logging
from typing import Dict, List, Optional, Union
from uuid import uuid4

from pymilvus.client.constants import DEFAULT_CONSISTENCY_LEVEL
from pymilvus.client.types import ExceptionsMessage
from pymilvus.exceptions import (
    DataTypeNotMatchException,
    MilvusException,
    PrimaryKeyException,
)
from pymilvus.orm import utility
from pymilvus.orm.collection import CollectionSchema
from pymilvus.orm.connections import connections
from pymilvus.orm.types import DataType

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

        self.uri = uri
        self.timeout = timeout
        self.conn = None

        self.default_search_params = None

        self._using = self._create_connection(uri, user, password, db_name, token, **kwargs)
        self.is_self_hosted = bool(
            utility.get_server_type(using=self._using) == "milvus",
        )

    def create_collection(
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
        index_params = {
            "metric_type": metric_type,
            "params": {},
        }
        self._create_index(collection_name, vector_field_name, index_params, timeout=timeout)
        self._load(collection_name, timeout=timeout)

    def _create_index(
        self,
        collection_name: str,
        vec_field_name: str,
        index_params: Dict,
        timeout: Optional[float] = None,
    ) -> None:
        """Create a index on the collection"""
        conn = self._get_connection()
        try:
            conn.create_index(
                collection_name,
                vec_field_name,
                index_params,
                timeout=timeout,
            )
            logger.debug(
                "Successfully created an index on collection: %s",
                collection_name,
            )
        except Exception as ex:
            logger.error(
                "Failed to create an index on collection: %s",
                collection_name,
            )
            raise ex from ex

    def insert(
        self,
        collection_name: str,
        data: Union[Dict, List[Dict]],
        batch_size: int = 0,
        progress_bar: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Union[str, int]]:
        """Insert data into the collection.

        If the Milvus Client was initiated without an existing Collection, the first dict passed
        in will be used to initiate the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (float, optional): The timeout to use, will override init timeout. Defaults
                to None.
            batch_size (int, optional): The batch size to perform inputs with. Defaults to 0,
                which means not batch the input.
            progress_bar (bool, optional): Whether to display a progress bar for the input.
                Defaults to False.

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
            return []

        if batch_size < 0:
            logger.error("Invalid batch size provided for insert.")
            msg = "Invalid batch size provided for insert."
            raise ValueError(msg)

        if batch_size == 0:
            batch_size = len(data)

        conn = self._get_connection()
        pks = []
        for i in self.tqdm(range(0, len(data), batch_size), disable=not progress_bar):
            # Convert dict to list of lists batch for insertion
            insert_batch = data[i : i + batch_size]
            # Insert into the collection.
            try:
                res = conn.insert_rows(collection_name, insert_batch, timeout=timeout)
                pks.extend(res.primary_keys)
            except Exception as ex:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s",
                    str(i),
                    str(len(data)),
                )
                raise ex from ex

        return pks

    def search(
        self,
        collection_name: str,
        data: Union[List[list], list],
        filter: str = "",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[dict]:
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
            List[dict]: A list of dicts containing the score and the result data. Embeddings are
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
        filter: str,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[dict]:
        """Query for entries in the Collection.

        Args:
            filter_expression (str): The filter to use for the query.
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
        if not isinstance(filter, str):
            raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(filter))

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

        try:
            res = conn.query(
                collection_name, expr=filter, output_fields=output_fields, timeout=timeout, **kwargs
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
                collection_name, expr=expr, output_fields=output_fields, timeout=timeout, **kwargs
            )
        except Exception as ex:
            logger.error("Failed to get collection: %s", collection_name)
            raise ex from ex

        return res

    def delete(
        self,
        collection_name: str,
        pks: Union[list, str, int],
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Union[str, int]]:
        """Delete entries in the collection by their pk.

        Delete all the entries based on the pk. If unsure of pk you can first query the collection
        to grab the corresponding data. Then you can delete using the pk_field.

        Args:
            pks (list, str, int): The pk's to delete. Depending on pk_field type it can be int
                or str or alist of either.
            timeout (int, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.
        """

        if isinstance(pks, (int, str)):
            pks = [pks]

        if len(pks) == 0:
            return []

        conn = self._get_connection()
        try:
            schema_dict = conn.describe_collection(collection_name, timeout=timeout, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex

        expr = self._pack_pks_expr(schema_dict, pks)
        ret_pks = []
        try:
            res = conn.delete(collection_name, expr, timeout=timeout, **kwargs)
            ret_pks.extend(res.primary_keys)
        except Exception as ex:
            logger.error("Failed to delete primary keys in collection: %s", collection_name)
            raise ex from ex
        return ret_pks

    def num_entities(self, collection_name: str, timeout: Optional[float] = None) -> int:
        """return the number of rows in the collection.

        Returns:
            int: Number for rows.
        """
        conn = self._get_connection()
        stats = conn.get_collection_stats(collection_name, timeout=timeout)
        result = {stat.key: stat.value for stat in stats}
        result["row_count"] = int(result["row_count"])
        return result["row_count"]

    def flush(self, collection_name: str, timeout: Optional[float] = None, **kwargs):
        """Seal all segments in the collection. Inserts after flushing will be written into
            new segments. Only sealed segments can be indexed.

        Args:
            timeout (float): an optional duration of time in seconds to allow for the RPCs.
                If timeout is not set, the client keeps waiting until the server responds
                or an error occurs.
        """
        conn = self._get_connection()
        conn.flush([collection_name], timeout=timeout, **kwargs)

    def describe_collection(self, collection_name: str, **kwargs):
        conn = self._get_connection()
        try:
            schema_dict = conn.describe_collection(collection_name, **kwargs)
        except Exception as ex:
            logger.error("Failed to describe collection: %s", collection_name)
            raise ex from ex
        return schema_dict

    def list_collections(self, **kwargs):
        conn = self._get_connection()
        try:
            collection_names = conn.list_collections(**kwargs)
        except Exception as ex:
            logger.error("Failed to list collections")
            raise ex from ex
        return collection_names

    def drop_collection(self, collection_name: str):
        """Delete the collection stored in this object"""
        conn = self._get_connection()
        conn.drop_collection(collection_name)

    @classmethod
    def create_schema(cls, **kwargs):
        kwargs["check_fields"] = False  # do not check fields for now
        return CollectionSchema([], **kwargs)

    @classmethod
    def prepare_index_params(
        cls,
        field_name: str,
        index_type: Optional[str] = None,
        metric_type: Optional[str] = None,
        index_name: str = "",
        params: Optional[Dict] = None,
        **kwargs,
    ):
        index_params = {"field_name": field_name}
        if index_type is not None:
            index_params["index_type"] = index_type
        if metric_type:
            index_params["metric_type"] = metric_type
        if index_name:
            index_params["index_name"] = index_name

        index_params["params"] = params or {}

        index_params.update(**kwargs)

        return index_params

    def create_collection_with_schema(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_params: Dict,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        schema.verify()
        if kwargs.get("auto_id", False):
            schema.auto_id = True
        if kwargs.get("enable_dynamic_field", False):
            schema.enable_dynamic_field = True
        schema.verify()

        index_params = index_params or {}
        vector_field_name = index_params.pop("field_name", "")
        if not vector_field_name:
            schema_dict = schema.to_dict()
            vector_field_name = self._get_vector_field_name(schema_dict)

        conn = self._get_connection()
        if "consistency_level" not in kwargs:
            kwargs["consistency_level"] = DEFAULT_CONSISTENCY_LEVEL
        try:
            conn.create_collection(collection_name, schema, timeout=timeout, **kwargs)
            logger.debug("Successfully created collection: %s", collection_name)
        except Exception as ex:
            logger.error("Failed to create collection: %s", collection_name)
            raise ex from ex

        self._create_index(collection_name, vector_field_name, index_params, timeout=timeout)
        self._load(collection_name, timeout=timeout)

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

    def _load(self, collection_name: str, timeout: Optional[float] = None):
        """Loads the collection."""
        conn = self._get_connection()
        try:
            conn.load_collection(collection_name, timeout=timeout)
        except MilvusException as ex:
            logger.error(
                "Failed to load collection: %s",
                collection_name,
            )
            raise ex from ex
