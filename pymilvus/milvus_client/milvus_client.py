"""MilvusClient for dealing with simple workflows."""
import logging
import threading
from typing import Dict, List, Union
from uuid import uuid4

from pymilvus.exceptions import MilvusException
from pymilvus.milvus_client.defaults import DEFAULT_SEARCH_PARAMS
from pymilvus.orm import utility
from pymilvus.orm.collection import Collection, CollectionSchema, FieldSchema
from pymilvus.orm.connections import connections
from pymilvus.orm.types import DataType, infer_dtype_bydata

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class MilvusClient:
    """The Milvus Client"""

    # pylint: disable=logging-too-many-args, too-many-instance-attributes, import-outside-toplevel

    def __init__(
        self,
        collection_name: str = "ClientCollection",
        pk_field: str = None,
        vector_field: str = None,
        uri: str = "http://localhost:19530",
        shard_num: int = None,
        partitions: List[str] = None,
        consistency_level: str = "Session",
        replica_number: int = 1,
        index_params: dict = None,
        distance_metric: str = "L2",
        timeout: int = None,
        overwrite: bool = False,
    ):
        """A client for the common Milvus use case.

        This client attempts to hide away the complexity of using Pymilvus. In a lot ofcases what
        the user wants is a simple wrapper that supports adding data, deleting data, and searching.
        This wrapper can autoinfer the schema from a previous collection or newly inserted data,
        can update the paritions, can query, and can delete by pk.

        Args:
            pk_field (str, optional): Which entry in data is considered the primary key. If None,
                an auto-id will be created. Will be overwritten if loading from a previous
                collection. Defaults to None.
            vector_field (str, optional): Which entry in the data is considered the vector field.
                Will get overwritten if loading from previous collection. Required if not using
                already made collection.
            uri (str, optional): The connection address to use to connect to the
                instance. Defaults to "http://localhost:19530". Another example:
                "https://username:password@in01-12a.aws-us-west-2.vectordb.zillizcloud.com:19538
            shard_num (int, optional): The amount of shards to use for the collection. Unless
                dealing with huge scale, recommended to keep at default. Defaults to None and allows
                server to set.
            partitions (List[str], optional): Which paritions to create for the collection.
                Defaults to None.
            consistency_level (str, optional): Which consistency level to use for the Client.
                The options are "Strong", "Bounded", "Eventually", "Session". Defaults to "Bounded".
            replica_number (int, optional): The amount of in memomory replicas to use.
                Defaults to 1.
            distance_metric (str, optional): Which distance metric to use if not supplying
                index params. Valid types are "IP" and "L2".
            index_params (dict, optional): What index parameteres to use for the Collection.
                If none, will use a default one. If collection already exists, will overwrite
                using this index.
            timeout (int, optional): What timeout to use for function calls. Defaults
                to None.
            overwrite (bool, optional): Whether to overwrite existing collection if exists.
                Defaults to False
        """
        # Optionial TQDM import
        try:
            import tqdm
            self.tqdm = tqdm.tqdm
        except ImportError:
            logger.debug("tqdm not found")
            self.tqdm = (lambda x, disable: x)

        self.uri = uri
        self.collection_name = collection_name
        self.shard_num = shard_num
        self.partitions = partitions
        self.consistency_level = consistency_level
        self.replica_number = replica_number
        self.distance_metric = distance_metric
        self.index_params = index_params
        self.timeout = timeout
        self.pk_field = pk_field
        self.vector_field = vector_field

        # TODO: Figure out thread safety
        # self.concurrent_counter = 0
        self.concurrent_lock = threading.RLock()
        self.default_search_params = None
        self.collection = None
        self.fields = None

        self.alias = self._create_connection()
        self.is_self_hosted = bool(
            utility.get_server_type(using=self.alias) == "milvus"
        )
        if overwrite and utility.has_collection(self.collection_name, using=self.alias):
            utility.drop_collection(self.collection_name, using=self.alias)

        self._init(None)

    def __len__(self):
        return self.num_entities()

    def num_entities(self):
        """return the number of rows in the collection.

        Returns:
            int: Number for rows.
        """
        if self.collection is None:
            return 0

        self.collection.flush()
        return self.collection.num_entities

    def insert_data(
        self,
        data: List[Dict[str, any]],
        timeout: int = None,
        batch_size: int = 100,
        partition: str = None,
        progress_bar: bool = False,
    ) -> List[Union[str, int]]:
        """Insert data into the collection.

        If the Milvus Client was initiated without an existing Collection, the first dict passed
        in will be used to initiate the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to pass in. If list not provided, will
                cast to list.
            timeout (int, optional): The timeout to use, will override init timeout. Defaults
                to None.
            batch_size (int, optional): The batch size to perform inputs with. Defaults to 100.
            partition (str, optional): Which partition to insert into. Defaults to None.
            progress_bar (bool, optional): Whether to display a progress bar for the input.
                Defaults to False.

        Raises:
            DataNotMatchException: If the data has misssing fields an exception will be thrown.
            MilvusException: General Milvus error on insert.

        Returns:
            List[Union[str, int]]: A list of primary keys that were inserted.
        """
        # If no data provided, we cannot input anything
        if len(data) == 0:
            return []

        if batch_size < 1:
            logger.error("Invalid batch size provided for insert.")

            raise ValueError("Invalid batch size provided for insert.")

        # If the collection hasnt been initialized, initialize it
        with self.concurrent_lock:
            if self.collection is None:
                self._init(data[0])

        # Dont include the primary key if auto_id is true and they included it in data
        ignore_pk = self.pk_field if self.collection.schema.auto_id else None
        insert_dict = {}
        pks = []

        for k in data:
            for key, value in k.items():
                if key in self.fields:
                    insert_dict.setdefault(key, []).append(value)

        for i in self.tqdm(range(0, len(data), batch_size), disable=not progress_bar):
            # Convert dict to list of lists batch for insertion
            try:
                insert_batch = [
                    insert_dict[key][i : i + batch_size]
                    for key in self.fields
                    if key != ignore_pk
                ]
            except KeyError as ex:
                logger.error(
                    "Malformed data, at least one of the inserts does not contain all"
                    " the required fields."
                )
                raise KeyError(
                    f"Malformed data, at least one of the inserts does not"
                    f" the required fields: {ex}",
                ) from ex
            # Insert into the collection.
            try:
                res = self.collection.insert(
                    insert_batch,
                    timeout=timeout or self.timeout,
                    partition_name=partition,
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

    def upsert_data(
        self,
        data: List[Dict[str, any]],
        timeout: int = None,
        batch_size: int = 100,
        partition: str = None,
        progress_bar: bool = False,
    ) -> List[Union[str, int]]:
        """WARNING: SLOW AND NOT ATOMIC. Will be updated for 2.3 release.

        Upsert the data into the collection.

        If the Milvus Client was initiated without an existing Collection, the first dict passed
        in will be used to initiate the collection.

        Args:
            data (List[Dict[str, any]]): A list of dicts to upsert.
            timeout (int, optional): The timeout to use, will override init timeout. Defaults
                to None.
            batch_size (int, optional): The batch size to perform inputs with. Defaults to 100.
            partition (str, optional): Which partition to insert into. Defaults to None.
            progress_bar (bool, optional): Whether to display a progress bar for the input.
                Defaults to False.
        Returns:
            List[Union[str, int]]: A list of primary keys that were inserted.
        """
        # If the collection exists we need to first delete the values
        if self.collection is not None:
            pks = [x[self.pk_field] for x in data]
            self.delete_by_pk(pks, timeout)

        ret = self.insert_data(
            data=data,
            timeout=timeout,
            batch_size=batch_size,
            partition=partition,
            progress_bar=progress_bar,
        )

        return ret

    def search_data(
        self,
        data: Union[List[list], list],
        top_k: int = 10,
        filter_expression: str = None,
        return_fields: List[str] = None,
        partitions: List[str] = None,
        search_params: dict = None,
        timeout: int = None,
    ) -> List[dict]:
        """Search for a query vector/vectors.

        In order for the search to process, a collection needs to have been either provided
        at init or data needs to have been inserted.

        Args:
            data (Union[List[list], list]): The vector/vectors to search.
            top_k (int, optional): How many results to return per search. Defaults to 10.
            filter_expression (str, optional): A filter to use for the search. Defaults to None.
            return_fields (List[str], optional): List of which field values to return. If None
                specified, all fields excluding vector field will be returned.
            search_params (dict, optional): The search params to use for the search. Will default
                to the default set for the client.


            partitions (List[str], optional): Which partitions to search within. Defaults to
                searching through all.
            timeout (int, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.

        Raises:
            ValueError: The collection being searched doesnt exist. Need to insert data first.

        Returns:
            List[dict]: A list of dicts containing the score and the result data. Embeddings are
                not included in the result data.
        """

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter += 1

        if self.collection is None:
            logger.error("Collection does not exist: %s", self.collection_name)
            raise ValueError(
                "Missing collection. Make sure data inserted or intialized on existing collection."
            )

        if not isinstance(data[0], list):
            data = [data]
        if return_fields is None or len(return_fields) == 0:
            return_fields = list(self.fields.keys())
            return_fields.remove(self.vector_field)

        try:
            res = self.collection.search(
                data,
                anns_field=self.vector_field,
                expr=filter_expression,
                param=search_params or self.default_search_params,
                limit=top_k,
                partition_names=partitions,
                output_fields=return_fields,
                timeout=timeout or self.timeout,
            )
        except Exception as ex:
            logger.error("Failed to search collection: %s", self.collection_name)
            raise ex

        ret = []
        for hits in res:
            query_result = []
            for hit in hits:
                ret_dict = {x: hit.entity.get(x) for x in return_fields}
                query_result.append({"score": hit.score, "data": ret_dict})
            ret.append(query_result)

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter -= 1
        return ret

    def query_data(
        self,
        filter_expression: str,
        return_fields: List[str] = None,
        partitions: List[str] = None,
        timeout: int = None,
    ) -> List[dict]:
        """Query for entries in the Collection.

        Args:
            filter_expression (str): The filter to use for the query.
            return_fields (List[str], optional): List of which field values to return. If None
                specified, all fields excluding vector field will be returned.
            partitions (List[str], optional): Which partitions to perform query. Defaults to None.
            timeout (int, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.

        Raises:
            ValueError: Missing collection.

        Returns:
            List[dict]: A list of result dicts, vectors are not included.
        """

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter += 1

        if self.collection is None:
            logger.error("Collection does not exist: %s", self.collection_name)
            raise ValueError(
                "Missing collection. Make sure data inserted or intialized on existing collection."
            )

        if return_fields is None or len(return_fields) == 0:
            return_fields = list(self.fields.keys())
            return_fields.remove(self.vector_field)

        res = self.collection.query(
            expr=filter_expression,
            partition_names=partitions,
            output_fields=return_fields,
            timeout=timeout or self.timeout,
        )

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter -= 1

        return res

    def get_vectors_by_pk(
        self,
        pks: Union[list, str, int],
        timeout: int = None,
    ) -> List[List[float]]:
        """Grab the inserted vectors using the primary key from the Collection.

        Due to current implementations, grabbing a large amount of vectors is slow.

        Args:
            pks (str): The pk's to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
            timeout (int, optional): Timeout to use, overides the client level assigned at
                init. Defaults to None.

        Raises:
            ValueError: Missing collection.

        Returns:
            List[dict]: A list of result dicts with keys {pk_field, vector_field}
        """

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter += 1

        if self.collection is None:
            logger.error("Collection does not exist: %s", self.collection_name)
            raise ValueError(
                "Missing collection. Make sure data inserted or intialized on existing collection."
            )

        if not isinstance(pks, list):
            pks = [pks]

        if len(pks) == 0:
            return []

        # Varchar pks need double quotes around the values
        if self.fields[self.pk_field] == DataType.VARCHAR:
            ids = ['"' + str(entry) + '"' for entry in pks]
            expr = f"""{self.pk_field} in [{','.join(ids)}]"""
        else:
            ids = [str(entry) for entry in pks]
            expr = f"{self.pk_field} in [{','.join(ids)}]"

        res = self.collection.query(
            expr=expr,
            output_fields=[self.vector_field],
            timeout=timeout or self.timeout,
        )

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter -= 1

        return res

    def delete_by_pk(
        self,
        pks: Union[list, str, int],
        timeout: int = None,
    ) -> None:
        """Delete entries in the collection by their pk.

        Delete all the entries based on the pk. If unsure of pk you can first query the collection
        to grab the corresponding data. Then you can delete using the pk_field.

        Args:
            pks (list, str, int): The pk's to delete. Depending on pk_field type it can be int
                or str or alist of either.
            timeout (int, optional): Timeout to use, overides the client level assigned at init.
                Defaults to None.
        """

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter += 1

        if self.collection is None:
            logger.error("Collection does not exist: %s", self.collection_name)
            return

        if not isinstance(pks, list):
            pks = [pks]

        if len(pks) == 0:
            return

        if self.fields[self.pk_field] == DataType.VARCHAR:
            ids = ['"' + str(entry) + '"' for entry in pks]
            expr = f"""{self.pk_field} in [{','.join(ids)}]"""
        else:
            ids = [str(entry) for entry in pks]
            expr = f"{self.pk_field} in [{','.join(ids)}]"

        self.collection.delete(expr=expr, timout=timeout or self.timeout)

        # TODO: Figure out thread safety
        # with self.concurrent_lock:
        #     self.concurrent_counter -= 1

    def add_partitions(self, input_partitions: List[str]):
        """Add partitions to the collection.

        Add a list of partition names to the collection. If the collection is loaded
        it will first be unloaded, then the partitions will be added, and then reloaded.

        Args:
            input_partitions (List[str]): The list of partition names to be added.

        Raises:
            MilvusException: Unable to add the partition.
        """
        if self.collection is not None and self.is_self_hosted:
            # Calculate which partitions need to be added
            input_partitions = set(input_partitions)
            current_partitions = {
                partition.name for partition in self.collection.partitions
            }
            new_partitions = input_partitions.difference(current_partitions)
            # If partitions need to be added, add them
            if len(new_partitions) != 0:
                # TODO: Remove with Milvus 2.3
                # Try to unload the collection
                self.collection.release()
                try:
                    for part in new_partitions:
                        self.collection.create_partition(part)
                    logger.debug(
                        "Successfully added partitions to collection: %s partitions: %s",
                        self.collection_name,
                        ",".join(part for part in list(new_partitions)),
                    )
                    # TODO: Remove with Milvus 2.3
                    self._load()
                except MilvusException as ex:
                    logger.debug(
                        "Failed to add partitions to: %s", self.collection_name
                    )
                    # TODO: Remove with Milvus 2.3
                    # Even if failed, attempt to reload collection
                    self._load()
                    raise ex
            else:
                logger.debug(
                    "No parititons to add for collection: %s", self.collection_name
                )
        else:
            logger.debug(
                "Collection either on Zilliz or non existant for collection: %s",
                self.collection_name,
            )

    def delete_partitions(self, remove_partitions: List[str]):
        """Remove partitions from the collection.

        Remove a list of partition names from the collection. If the collection is loaded
        it will first be unloaded, then the partitions will be removed, and then reloaded.

        Args:
            remove_partitions (List[str]): The list of partition names to be removed.

        Raises:
            MilvusException: Unable to remove the partition.
        """
        if self.collection is not None and self.is_self_hosted:
            # Calculate which partitions need to be removed
            remove_partitions = set(remove_partitions)
            current_partitions = {
                partition.name for partition in self.collection.partitions
            }
            removal_partitions = remove_partitions.intersection(current_partitions)
            # If partitions need to be added, add them
            if len(removal_partitions) != 0:
                # TODO: Remove with Milvus 2.3
                # Try to unload the collection
                self.collection.release()
                try:
                    for part in removal_partitions:
                        self.collection.drop_partition(part)
                    logger.debug(
                        "Successfully deleted partitions from collection: %s partitions: %s",
                        self.collection_name,
                        ",".join(part for part in list(removal_partitions)),
                    )
                    # TODO: Remove with Milvus 2.3
                    self._load()
                except MilvusException as ex:
                    logger.debug(
                        "Failed to delete partitions from: %s", self.collection_name
                    )
                    # TODO: Remove with Milvus 2.3
                    # Even if failed, attempt to reload collection
                    self._load()
                    raise ex
            else:
                logger.debug(
                    "No parititons to delete for collection: %s",
                    self.collection_name,
                )

    def delete_collection(self):
        """Delete the collection stored in this object"""
        with self.concurrent_lock:
            if self.collection is None:
                return
            self.collection.drop()
            self.collection = None

    def close(self, delete_collection=False):
        if delete_collection:
            self.delete_collection()
        connections.disconnect(self.alias)

    def _create_connection(self) -> str:
        """Create the connection to the Milvus server."""
        # TODO: Implement reuse with new uri style
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, uri=self.uri)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as ex:
            logger.error("Failed to create new connection using: %s", alias)
            raise ex

    def _init(self, input_data: dict):
        """Create/connect to the colletion"""
        # If no input data and collection exists, use that
        if input_data is None and utility.has_collection(
            self.collection_name, using=self.alias
        ):
            self.collection = Collection(self.collection_name, using=self.alias)
            # Grab the field information from the existing collection
            self._extract_fields()
        # If data is supplied we can create a new collection
        elif input_data is not None:
            self._create_collection(input_data)
        # Nothin to init from
        else:
            logger.debug(
                "No information to perform init from for collection %s",
                self.collection_name,
            )
            return

        # TODO: Make sure this drops the correct index
        if self.index_params is not None:
            self.collection.drop_index()

        self._create_index()
        # Partitions only allowed on Milvus at the moment
        if self.is_self_hosted and self.partitions is not None:
            self.add_partitions(self.partitions)
        self._create_default_search_params()
        self._load()

    def _create_collection(self, data: dict) -> None:
        """Create the collection by autoinferring the schema."""

        fields = self._infer_fields(data)

        if self.vector_field is None:
            logger.error(
                "vector_field not supplied at init(), cannot infer schema from data collection: %s",
                self.collection_name,
            )
            raise ValueError(
                "vector_field not supplied at init(), cannot infer schema."
            )

        if self.vector_field not in fields:
            logger.error(
                "Missing vector_field: %s in data for collection: %s",
                self.vector_field,
                self.collection_name,
            )
            raise ValueError(
                "vector_field missing in inserted data, cannot infer schema."
            )

        if fields[self.vector_field]["dtype"] not in (
            DataType.BINARY_VECTOR,
            DataType.FLOAT_VECTOR,
        ):
            logger.error(
                "vector_field: %s does not correspond with vector dtype in data for collection: %s",
                self.vector_field,
                self.collection_name,
            )
            raise ValueError("vector_field does not correspond to vector dtype.")

        if fields[self.vector_field]["dtype"] == DataType.BINARY_VECTOR:
            dim = 8 * len(data[self.vector_field])
        elif fields[self.vector_field]["dtype"] == DataType.FLOAT_VECTOR:
            dim = len(data[self.vector_field])
        # Attach dim kwarg to vector field
        fields[self.vector_field]["dim"] = dim

        # If pk not provided, created autoid pk
        if self.pk_field is None:
            # Generate a unique auto-id field
            self.pk_field = "internal_pk_" + uuid4().hex[:4]
            # Create a new field for pk
            fields[self.pk_field] = {}
            fields[self.pk_field]["name"] = self.pk_field
            fields[self.pk_field]["dtype"] = DataType.INT64
            fields[self.pk_field]["auto_id"] = True
            fields[self.pk_field]["is_primary"] = True
            logger.debug(
                "Missing pk_field, creating auto-id pk for collection: %s",
                self.collection_name,
            )
        # If pk_field given, we assume it will be provided for all inputs
        else:
            try:
                fields[self.pk_field]["auto_id"] = False
                fields[self.pk_field]["is_primary"] = True
            except KeyError as ex:
                logger.error(
                    "Missing pk_field: %s in data for collection: %s",
                    self.pk_field,
                    self.collection_name,
                )
                raise ex
        try:
            # Create the fieldschemas
            fieldschemas = []
            # TODO: Assuming ordered dicts for 3.7
            self.fields = {}
            for field_dict in fields.values():
                fieldschemas.append(FieldSchema(**field_dict))
                self.fields[field_dict["name"]] = field_dict["dtype"]
            # Create the schema for the collection
            schema = CollectionSchema(fieldschemas)
            # Create the collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                consistency_level=self.consistency_level,
                shards_num=self.shard_num,
                using=self.alias,
            )
            logger.debug("Successfully created collection: %s", self.collection_name)
        except MilvusException as ex:
            logger.error("Failed to create collection: %s", self.collection_name)
            raise ex

    def _infer_fields(self, data):
        """Infer all the fields based on the input data."""
        # TODO: Assuming ordered dict for 3.7
        fields = {}
        # Figure out each datatype of the input.
        for key, value in data.items():
            # Infer the corresponding datatype of the metadata
            dtype = infer_dtype_bydata(value)
            # Datatype isnt compatible
            if dtype in (DataType.UNKNOWN, DataType.NONE):
                logger.error(
                    "Failed to parse schema for collection %s, unrecognized dtype for key: %s",
                    self.collection_name,
                    key,
                )
                raise ValueError(f"Unrecognized datatype for {key}.")

            # Create an entry under the field name
            fields[key] = {}
            fields[key]["name"] = key
            fields[key]["dtype"] = dtype

            # Area for attaching kwargs for certain datatypes
            if dtype == DataType.VARCHAR:
                fields[key]["max_length"] = 65_535

        return fields

    def _extract_fields(self) -> None:
        """Grab the existing fields from the Collection"""
        self.fields = {}
        schema = self.collection.schema
        for field in schema.fields:
            field_dict = field.to_dict()
            if field_dict.get("is_primary", None) is not None:
                logger.debug("Updating pk_field with one from collection.")
                self.pk_field = field_dict["name"]
            if field_dict["type"] in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                logger.debug("Updating vector_field  with one from collection.")
                self.vector_field = field_dict["name"]
            self.fields[field_dict["name"]] = field_dict["type"]

        logger.debug(
            "Successfully extracted fields from for collection: %s, total fields: %s, "
            "pk_field: %s, vector_field: %s",
            self.collection_name,
            len(self.fields),
            self.pk_field,
            self.vector_field,
        )

    def _create_index(self) -> None:
        """Create a index on the collection"""
        if self._get_index() is None:
            # If no index params, use a default HNSW based one
            if self.index_params is None:
                # TODO: Once segment normalization we can default to IP
                metric_type = (
                    self.distance_metric
                    if self.fields[self.vector_field] == DataType.FLOAT_VECTOR
                    else "JACCARD"
                )
                # TODO: Once AUTOINDEX type is supported by Milvus we can default to HNSW always
                index_type = "HNSW" if self.is_self_hosted else "AUTOINDEX"
                params = {"M": 8, "efConstruction": 64} if self.is_self_hosted else {}
                self.index_params = {
                    "metric_type": metric_type,
                    "index_type": index_type,
                    "params": params,
                }
            try:
                self.collection.create_index(
                    self.vector_field,
                    index_params=self.index_params,
                    using=self.alias,
                    timeout=self.timeout,
                )
                logger.debug(
                    "Successfully created an index on collection: %s",
                    self.collection_name,
                )
            except MilvusException as ex:
                logger.error(
                    "Failed to create an index on collection: %s", self.collection_name
                )
                raise ex
        else:
            logger.debug(
                "Index exists already for collection: %s", self.collection_name
            )

    def _get_index(self):
        """Return the index dict if index exists."""
        for index in self.collection.indexes:
            if index.field_name == self.vector_field:
                return index
        return None

    def _create_default_search_params(self) -> None:
        """Generate search params based on the current index type"""
        index = self._get_index().to_dict()
        if index is not None:
            index_type = index["index_param"]["index_type"]
            metric_type = index["index_param"]["metric_type"]
            self.default_search_params = DEFAULT_SEARCH_PARAMS[index_type]
            self.default_search_params["metric_type"] = metric_type

    def _load(self):
        """Loads the collection."""
        if self._get_index() is not None:
            if self.is_self_hosted:
                try:
                    self.collection.load(replica_number=self.replica_number)
                    logger.debug(
                        "Collection loaded: %s",
                        self.collection_name,
                    )
                # If the replica count is incorrect, release the collection
                except MilvusException:
                    try:
                        self.collection.release(timeout=self.timeout)
                        self.collection.load(replica_number=self.replica_number)
                        logger.debug(
                            "Successfully reloaded collection: %s",
                            self.collection_name,
                        )
                    except MilvusException as ex:
                        logger.error(
                            "Failed to load collection: %s",
                            self.collection_name,
                        )
                        raise ex
            else:
                try:
                    self.collection.load(replica_number=1)
                    logger.debug(
                        "Collection loaded: %s",
                        self.collection_name,
                    )
                # If both loads fail, raise exception
                except MilvusException as ex:
                    logger.error("Failed to load collection: %s", self.collection_name)
                    raise ex
