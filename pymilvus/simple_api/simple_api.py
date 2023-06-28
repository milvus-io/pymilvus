"""MilvusClient for dealing with simple workflows."""
import logging
from typing import Dict, List, Union
from uuid import uuid4

import pymilvus.simple_api.simple_api_exceptions as simple_exception
from pymilvus.client.constants import ConsistencyLevel
from pymilvus.exceptions import MilvusException
from pymilvus.orm.collection import CollectionSchema, FieldSchema
from pymilvus.orm.connections import connections
from pymilvus.orm.types import DataType

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

VALID_PARTITION_TYPES = {"str", "int"}
VALID_DISTANCE_METRICS = {"L2", "IP"}
# Add a method to parse from collection


class SimpleAPI:
    # pylint: disable=logging-too-many-args, too-many-instance-attributes, import-outside-toplevel
    def __init__(
        self,
        uri: str = "http://localhost:19530/default",
        api_key: str = None,
        user: str = None,
        password: str = None,
        timeout: int = None,
    ):
        """Simple API is an API that hides a majority of steps that it takes to use Milvus.

        This client offers the most common operations that are performed on vector databases in a simple
        to use format.

        Args:
            - uri (str, optional): The URI to connect to. Defaults to "http://localhost:19530/default" with
                default being the database being connected to.
            - api_key (str, optional): An API key for authenitcation to the server. Defaults to None.
            - user (str, optional): The username used to connect to the DB. Defaults to None.
            - password (str, optional): The password used to connect to the DB. Defaults to None.
        """
        # Optionial TQDM import
        try:
            import tqdm

            self.tqdm = tqdm.tqdm
        except ImportError:
            logger.debug("tqdm not found")
            self.tqdm = lambda x, disable: x

        self.uri = uri or "http://localhost:19530/default"
        if api_key is None:
            api_key = f"{user if user is not None else ''}:{password if password is not None else ''}"
            logger.debug("Using USER:PASS API Key")

        self.alias = self._create_connection(uri, api_key, timeout=timeout)
        self.conn = connections._fetch_handler(self.alias)

    def list_collections(self, timeout: int = None) -> List[str]:
        """List the collections within the connected database.

        Args:
            - timeout (int, optional): Defaults to None.

        Returns:
            - list: List of database names
        """
        return self.conn.list_collections(timeout=timeout)

    # def list_partitions(self, collection_name: str, timeout: int = None):
    #     if not self.conn.has_collection(
    #         collection_name=collection_name, using=self.alias, timeout=timeout
    #     ):
    #         raise simple_exception.CollectionDoesNotExist()
    #     partition_strs = self.conn.list_partitions(collection_name)
    #     return partition_strs

    def describe_collection(
        self, collection_name: str, extra: bool = True, timeout: int = None
    ) -> dict:
        """Describe the named collection.

        Args:
            collection_name (str): Name of the collection.
            extra (bool, optional): Whether to include extra easier to read data. Defaults to True.
            timeout (int, optional): _description_. Defaults to None.

        Raises:
            simple_exception.CollectionDoesNotExist: Specified collection does not exist.

        Returns:
            dict: Dict containing all the information about the collection.
        """
        if not self.conn.has_collection(
            collection_name=collection_name, using=self.alias, timeout=timeout
        ):
            raise simple_exception.CollectionDoesNotExist(
                f"Collection {collection_name} does not exist."
            )
        res = self.conn.describe_collection(
            collection_name=collection_name, timeout=timeout
        )
        if extra:
            res["consistency_level_text"] = ConsistencyLevel.Name(
                res["consistency_level"]
            )
            fields = res["fields"]
            res["simplified_fields"] = {}
            res["primary_field"] = None
            res["vector_field"] = None
            res["dynamic_field"] = None
            res["partition_key_field"] = None
            for field in fields:
                if field.get("is_primary"):
                    res["primary_field"] = field["name"]
                if field.get("type") in (DataType.BINARY_VECTOR, DataType.FLOAT_VECTOR):
                    res["vector_field"] = field["name"]
                if field.get("is_partition_key"):
                    res["partition_key_field"] = field["name"]
                if field.get("is_dynamic"):
                    res["dynamic_field"] = field["name"]
                res["simplified_fields"][field["name"]] = DataType(field["type"]).name

        return res

    def num_entities(self, collection_name: str, timeout: int = None) -> int:
        """Return the number of entries within a collection.

        Calls a flush on the collection to seal the data and accurately update count.
        Will not be accurate after deletion.

        Args:
            - collection_name (str): Name of the collection.
            - timeout (int, optional): Defaults to None.

        Raises:
            - simple_exception.CollectionDoesNotExist: Specified collection does not exist.

        Returns:
            - int: Number of entries in the collection.
        """
        if not self.conn.has_collection(
            collection_name=collection_name, timeout=timeout
        ):
            raise simple_exception.CollectionDoesNotExist(
                f"Collection {collection_name} does not exist."
            )
        self.conn.flush(collection_names=[collection_name])
        stats = self.conn.get_collection_stats(collection_name=collection_name)
        result = {stat.key: stat.value for stat in stats}
        result["row_count"] = int(result["row_count"])
        return result["row_count"]

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        vector_field: str = "vector",
        primary_field: str = "id",
        primary_type: str = "int",
        primary_auto_id: bool = True,
        metric_type: str = "IP",
        partition_field: dict = None,
        index_params: dict = None,
        overwrite: bool = False,
        consistency_level: str = "Bounded",
        replicas: int = 1,
        timeout: int = None,
        **kwargs,
    ) -> None:
        """Create a collection.

        Creates a collection using the provided parameters.

        Args:
            - collection_name (str): Name of the collection.
            - dimension (int): Dimension of the float vector.
            - vector_field (str, optional): The name of the field that stores the vector
                in the insertion dict. Defaults to "vector".
            - primary_field (str, optional): The name of the field that stores the primary key
                in the insertion dict. Defaults to "vector".
            - primary_type (str, optional): The datatype of the primary key, at the moment it is limited to
                "int" and "str".
            - primary_auto_id (bool, optional): Whether Milvus should assign the value for the primary key
                on insert.
            - metric_type (str, optional): Which distance metric to use. Options are ("IP", "L2").
                Overwritten by index_params. Defaults to "IP".
            - partition_field (dict, optional): Information about the partition field key. By supplying a
                partition key, filtering on the value of this key in a search will greatly improve
                performance of the partition This partition field key must be within the inserted data
                if no default is given. The structure follows
                {"name":xxx, "type": "int" or "str", default: Optional} Defaults to None.
            - index_params (dict, optional): Index params to use if you want to specify your own.
                More info: https://milvus.io/docs/index.md. Defaults to None.
            - overwrite (bool, optional): Whether to replace an existing index with this name. Defaults to False.
            - consistency_level (str, optional): Which consisteny level to use, more info here:
                https://milvus.io/docs/consistency.md#Consistency-levels. Defaults to "Session".
            - replicas (int, optional): Numver of replicas to load. Defaults to 1.
            - timeout (int, optional):Defaults to None.


        Raises:
            - simple_exception.CollectionAlreadyExists: If collection with same name exists and overwrite is
                set to false.
            - simple_exception.InvalidPartitionFieldFormat: If the partition field is improperly formatted.
            - simple_exception.InvalidDistanceMetric: If the distance metric is not a valid one.
        """
        if self.conn.has_collection(
            collection_name=collection_name, using=self.alias, timeout=timeout
        ):
            if overwrite is True:
                self.conn.drop_collection(collection_name, timeout=timeout)
                logger.debug(
                    "Dropping collection %s due to overwrite param.", collection_name
                )
            else:
                raise simple_exception.CollectionAlreadyExists()

        if primary_type.lower() not in VALID_PARTITION_TYPES:
            raise simple_exception.InvalidPKFormat(
                f"primary_type must be in {VALID_PARTITION_TYPES}"
            )

        fields = []
        if primary_type.lower() == "str":
            if primary_auto_id:
                raise simple_exception.InvalidPKFormat(
                    "Str based primary_field cannot be auto-id'ed"
                )
            fields.append(
                FieldSchema(
                    primary_field,
                    DataType.VARCHAR,
                    is_primary=True,
                    max_length=65_535,
                )
            )
        elif primary_type.lower() == "int":
            fields.append(
                FieldSchema(
                    primary_field,
                    DataType.INT64,
                    is_primary=True,
                    auto_id=primary_auto_id,
                )
            )

        fields.append(FieldSchema(vector_field, DataType.FLOAT_VECTOR, dim=dimension))

        if partition_field:
            name = partition_field.get("name", None)
            dtype = partition_field.get("type", "")
            default_value = partition_field.get("default", None)
            if not isinstance(name, str):
                raise simple_exception.InvalidPartitionFieldFormat(
                    """Valid name must be of type str"""
                )
            if not isinstance(dtype, str) or dtype.lower() not in VALID_PARTITION_TYPES:
                raise simple_exception.InvalidPartitionFieldFormat(
                    f"""Valid partition dtypes are {VALID_PARTITION_TYPES}"""
                )
            if dtype.lower() == "str":
                fields.append(
                    FieldSchema(
                        name,
                        DataType.VARCHAR,
                        max_length=65_535,
                        default_value=default_value,
                        is_partition_key=True,
                    )
                )
            elif dtype.lower() == "int":
                fields.append(
                    FieldSchema(
                        name,
                        DataType.INT64,
                        default_value=default_value,
                        is_partition_key=True,
                    )
                )

        schema = CollectionSchema(
            fields, "Generated from SimpleAPI.", enable_dynamic_field=True
        )
        self.conn.create_collection(
            collection_name,
            schema,
            using=self.alias,
            consistency_level=consistency_level,
            timeout=timeout,
        )

        if index_params is None:
            if metric_type not in VALID_DISTANCE_METRICS:
                raise simple_exception.InvalidDistanceMetric(
                    f"Distance metric {metric_type} not in {VALID_DISTANCE_METRICS}"
                )
            index_params = {"metric_type": metric_type, "params": {}}

        self.conn.create_index(
            collection_name=collection_name,
            field_name=vector_field,
            params=index_params,
            timeout=timeout,
        )

        self.conn.load_collection(
            collection_name=collection_name, replica_number=replicas
        )

    def create_collection_from_schema(
        self,
        collection_name: str,
        schema: CollectionSchema,
        metric_type: str = "IP",
        index_params: dict = None,
        overwrite: bool = False,
        consistency_level: str = "Session",
        replicas: int = 1,
        timeout: int = None,
    ):
        """Create a collection using a premade Collection Schema.

        Args:
            - collection_name (str): Name of the colleciton to create.
            - schema (CollectionSchema): The premade CollectionSchema.
            - metric_type (str, optional): Distance metric to use. Gets overwritten
                by custom index_params. Defaults to "IP".
            - index_params (dict, optional): Custom indexing params to use. Defaults to None.
            - overwrite (bool, optional): Whether to overwrite collection with same name.
                Defaults to False.
            - consistency_level (str, optional): Which consitency level to use.
                Options are (Strong, Bounded, Seassion, Eventual), Defaults to "Bounded".
            - replicas (int, optional): How many replicas to load in. Defaults to 1.
            - timeout (int, optional): Defaults to None.

        Raises:
            simple_exception.CollectionAlreadyExists: Collection already exists.
            simple_exception.InvalidDistanceMetric: Invalid distance metric.
        """
        if self.conn.has_collection(
            collection_name=collection_name, using=self.alias, timeout=timeout
        ):
            if overwrite is True:
                self.conn.drop_collection(collection_name, timeout=timeout)
                logger.debug(
                    "Dropping collection %s due to overwrite param.", collection_name
                )
            else:
                raise simple_exception.CollectionAlreadyExists()

        self.conn.create_collection(
            collection_name,
            schema,
            using=self.alias,
            consistency_level=consistency_level,
            timeout=timeout,
        )

        if index_params is None:
            if metric_type not in VALID_DISTANCE_METRICS:
                raise simple_exception.InvalidDistanceMetric(
                    f"Distance metric {metric_type} not in {VALID_DISTANCE_METRICS}"
                )
            index_params = {"metric_type": metric_type, "params": {}}

        vector_field = ""

        fields = schema.to_dict().get("fields", [])
        for field_dict in fields:
            if field_dict.get("type", None) == DataType.FLOAT_VECTOR:
                vector_field = field_dict.get("name", "")

        self.conn.create_index(
            collection_name=collection_name,
            field_name=vector_field,
            params=index_params,
            timeout=timeout,
        )

        self.conn.load_collection(
            collection_name=collection_name, replica_number=replicas
        )

    def insert(
        self,
        collection_name: str,
        data: List[Dict[str, any]],
        batch_size: int = 100,
        progress_bar: bool = False,
        timeout: int = None,
    ) -> List[Union[str, int]]:
        """_summary_

        Args:
            - collection_name (str): Collection to insert into.
            - data (List[Dict[str, any]]): Insert data in row format with each row a dict of `field_name: value`.
            - batch_size (int, optional): Batch size of insert. Defaults to 100.
            - progress_bar (bool, optional): Whether to show progress bar on the batch of input. Defaults to False.
            - timeout (int, optional):Defaults to None.

        Raises:
            - simple_exception.InvalidInsertBatchSize: Invalid insert batch size provided.
            - ex: Exception caused during insert call.

        Returns:
            - List[Union[str, int]]: Primary keys that were inserted.
        """
        # If no data provided, we cannot input anything
        if len(data) == 0:
            return []

        if batch_size < 0:
            raise simple_exception.InvalidInsertBatchSize(
                f"Invalid batch size of {batch_size}"
            )

        if batch_size == 0:
            batch_size = len(data)

        pks = []

        for i in self.tqdm(range(0, len(data), batch_size), disable=not progress_bar):
            # Convert dict to list of lists batch for insertion
            # Insert into the collection.
            try:
                res = self.conn.insert_rows(
                    collection_name=collection_name,
                    entities=data[i : i + batch_size],
                    timeout=timeout,
                )
                pks.extend(res.primary_keys)
            except MilvusException as ex:
                logger.error(
                    "Failed to insert batch starting at entity: %s/%s",
                    str(i),
                    str(len(data)),
                )
                raise ex
        return pks

    def search(
        self,
        collection_name: str,
        data: Union[List[list], list],
        search_params: dict = None,
        top_k: int = 10,
        filter_expression: str = None,
        output_fields: List[str] = None,
        include_vectors=False,
        partition_keys: List[Union[str, int]] = None,
        timeout: int = None,
        consistency_level: str = None,
    ) -> List[List[dict]]:
        """Search for the vector within the given collection.

        Args:
            - collection_name (str): The collection to search.
            - data (Union[List[list], list]): The vector/s to search for.
            - search_params (dict, optional): Optional search params to pass through for
                custom index. Defaults to None.
            - top_k (int, optional): How many results to return per search. Defaults to 10.
            - filter_expression (str, optional): Attribute to be done on search. Defaults to None.
            - output_fields (List[str], optional): Which fields to include in output, defaults
                to all except vector field. Defaults to None.
            - include_vectors (bool, optional): Whether to include the vector field. Defaults to False.
            - partition_keys (List[Union[str, int]], optional): Which partitions to search
                through. Defaults to None.
            - timeout (int, optional): Defaults to None.
            - consistency_level (str, optional): Which consistency level to search with.
                Options are (Strong, Bounded, Seassion, Eventual) Defaults to collection default.

        Raises:
            - simple_exception.CollectionDoesNotExist: Collection does not exist.
            - ex: Any errors that occur during sending request over network.

        Returns:
            - List[dict]: List of list of entries. Top level list is for each search vector,
                inner level is for top-k results.
        """
        if not self.conn.has_collection(
            collection_name=collection_name, using=self.alias, timeout=timeout
        ):
            raise simple_exception.CollectionDoesNotExist()

        # Convert to list if single vector
        if not isinstance(data[0], list):
            data = [data]

        schema = self.describe_collection(collection_name)
        primary_field = schema["primary_field"]
        vec_field = schema["vector_field"]

        if output_fields is None or len(output_fields) == 0:
            output_fields = ["*"]

        try:
            output_fields.remove(vec_field)
            include_vectors = True
        except ValueError:
            pass

        expr = []

        if partition_keys is not None and len(partition_keys) != 0:
            partition_field = schema["partition_key_field"]
            expr.append(self._fetch_formatter(partition_field, partition_keys))

        if filter_expression is not None:
            expr.append(filter_expression)

        expr = " and ".join(expr)

        try:
            res = self.conn.search(
                collection_name,
                data,
                anns_field=vec_field,
                expr=expr,
                param=search_params or {},
                limit=top_k,
                output_fields=output_fields,
                timeout=timeout,
                consistency_level=consistency_level,
            )
        except Exception as ex:
            raise ex
        ret = []
        if include_vectors:
            pks = []
        for hits in res:
            query_result = []
            for hit in hits:
                ret_dict = hit.entity._row_data
                ret_dict[primary_field] = hit.id
                query_result.append({"score": hit.score, "data": ret_dict})
                if include_vectors:
                    pks.append(hit.id)
            ret.append(query_result)

        if include_vectors:
            vecs = self.fetch(
                collection_name=collection_name,
                field_name=primary_field,
                values=pks,
                output_fields=[vec_field],
                timeout=timeout,
                consistency_level=consistency_level,
            )
            vecs = {x[primary_field]: x[vec_field] for x in vecs}
            vecs = [vecs[pk] for pk in pks]
            count = 0
            for hits in ret:
                for hit in hits:
                    hit["data"][vec_field] = vecs[count]
                    count += 1

        return ret

    def query(
        self,
        collection_name: str,
        filter_expression: str,
        output_fields: List[str] = None,
        include_vectors: bool = False,
        partition_keys: List[Union[str, int]] = None,
        timeout: int = None,
        consistency_level: str = None,
    ) -> List[dict]:
        """Query the collection for values based on an expression.

        Returning vectors without using a singular primary key filter is SLOW.


        Args:
            - collection_name (str): The collection name.
            - filter_expression (str): The expression to base the query on.
            - output_fields (List[str], optional): Which fields to include in output dict, by
                default all will be returned except the vector field.
            - include_vectors (bool, optional): Whether to also include vectors. Defaults to False.
            - partition_keys (List[Union[str, int]], optional): Which partitions to query in.
                Defaults to None.
            - timeout (int, optional): Defaults to None.
            - consistency_level (str, optional): Which consistency level to use.
                Options are (Strong, Bounded, Seassion, Eventual) Defaults to collection default.

        Raises:
            - simple_exception.CollectionDoesNotExist: If the collection being fetched from does not exist.

        Returns:
            - List[dict]: List of entry dicts.
        """

        if not self.conn.has_collection(
            collection_name=collection_name, using=self.alias, timeout=timeout
        ):
            raise simple_exception.CollectionDoesNotExist()

        # Grab necessary schema info
        schema = self.describe_collection(collection_name)
        primary_field = schema["primary_field"]
        vec_field = schema["vector_field"]

        expr = []

        # Check if we are performing a partition search
        if partition_keys is not None and len(partition_keys) != 0:
            partition_field = schema["partition_key_field"]
            expr.append(self._fetch_formatter(partition_field, partition_keys))

        # Combine the filter expression with filter for partition
        expr.append(filter_expression)
        expr = " and ".join(expr)

        # Check if returning all data
        if output_fields is None or len(output_fields) == 0:
            output_fields = ["*"]

        # Change logic to avoid two output_field searches
        if include_vectors and vec_field not in output_fields:
            output_fields.append(vec_field)

        # If we are also returning a vector we need to query by pk
        if vec_field in output_fields:
            pks = self.conn.query(
                collection_name=collection_name,
                expr=expr,
                output_fields=None,
                timeout=timeout,
                consistency_level=consistency_level,
            )
            pks = [x[primary_field] for x in pks]
            expr = self._fetch_formatter(primary_field, pks)

        res = self.conn.query(
            collection_name=collection_name,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            consistency_level=consistency_level,
        )

        return res

    def delete(
        self,
        collection_name: str,
        field_name: str,
        values: Union[list, str, int],
        partition_keys: List[Union[str, int]] = None,
        timeout: int = None,
        consistency_level: str = None,
    ) -> None:
        """Delete entries from the collection.

        Args:
            - collection_name (str): The collection to delete from.
            - field_name (str): The field to match values to delete.
            - values (Union[list, str, int]): Which values to delete
            - partition_keys (List[Union[str, int]], optional): Which partitions to delete form.
                Defaults to all.
            - timeout (int, optional): Defaults to None.
            - consistency_level (str, optional):  Which consitency level to use.
                Options are (Strong, Bounded, Seassion, Eventual). Defaults to collection default

        Raises:
            - simple_exception.CollectionDoesNotExist: If the collection does not exist.
        """
        if not self.conn.has_collection(
            collection_name=collection_name, using=self.alias, timeout=timeout
        ):
            raise simple_exception.CollectionDoesNotExist()

        if not isinstance(values, list):
            values = [values]

        if len(values) == 0:
            return

        schema = self.describe_collection(collection_name, extra=True)

        primary_field = schema["primary_field"]

        expr = []

        if partition_keys is not None and len(partition_keys) != 0:
            partition_field = schema["partition_key_field"]
            expr.append(self._fetch_formatter(partition_field, partition_keys))

        expr.append(self._fetch_formatter(field=field_name, values=values))
        expr = " and ".join(expr)

        if field_name != primary_field:
            pks = self.conn.query(
                collection_name=collection_name,
                expr=expr,
                timeout=timeout,
                consistency_level=consistency_level,
            )
            field_name = primary_field
            values = [x[primary_field] for x in pks]
            expr = self._fetch_formatter(field=primary_field, values=values)

        self.conn.delete(
            collection_name=collection_name,
            expr=expr,
            timeout=timeout,
            consistency_level=consistency_level,
        )
        return

    def fetch(
        self,
        collection_name: str,
        field_name: str,
        values: Union[list, str, int],
        output_fields: List[str] = None,
        include_vectors: bool = False,
        partition_keys: List[Union[str, int]] = None,
        timeout: int = None,
        consistency_level: str = None,
    ) -> List[List[dict]]:
        """Fetch a row from the collection based on matching a field.

        Fetching a vector without using the primary key is SLOW.

        Args:
            - collection_name (str): The collection to fetch from.
            - field_name (str): The name of the field to match.
            - values (Union[list, str, int]): The values to match from the field.
            - output_fields (List[str], optional): Which fields to output, by default all
                will be returned excluding the vector field.
            - include_vectors (bool, optional): Whether to include vectors. Defaults to False.
            - partition_keys (List[Union[str, int]], optional): Which partitions to look in, defaults to all.
            - timeout (int, optional): Defaults to None.
            - consistency_level (str, optional):  Which consitency level to use.
                Options are (Strong, Bounded, Seassion, Eventual) Defaults to collection default.

        Returns:
            List[List[dict]]: A list of entry dicts.
        """
        expr = self._fetch_formatter(field=field_name, values=values)
        return self.query(
            collection_name,
            expr,
            output_fields=output_fields,
            include_vectors=include_vectors,
            partition_keys=partition_keys,
            timeout=timeout,
            consistency_level=consistency_level,
        )

    def drop_collection(self, collection_name: str, timeout: int = None):
        """Drop the collection and delete all relevant data and indexes.

        Args:
            - collection_name (str): The collection to drop.
            - timeout (int, optional): Defaults to None.
        """
        if self.conn.has_collection(collection_name=collection_name, timeout=timeout):
            self.conn.drop_collection(collection_name, timeout=timeout)

    def close(self):
        """Close the conneciton to the database."""
        connections.disconnect(self.alias)

    def _create_connection(self, uri: str, api_key: str, timeout: int = None) -> str:
        """Create the connection to the Milvus server."""
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, uri=uri, token=api_key, timeout=timeout)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as ex:
            raise ex

    def _fetch_formatter(self, field, values):
        # Varchar pks need double quotes around the values
        if len(values) < 1:
            return ""

        if isinstance(values[0], str):
            ids = ['"' + str(entry) + '"' for entry in values]
            expr = f"""{field} in [{','.join(ids)}]"""
        else:
            ids = [str(entry) for entry in values]
            expr = f"{field} in [{','.join(ids)}]"
        return expr
