from . import connections
from .schema import CollectionSchema


class Collection(object):
    """
    This is a class corresponding to collection in milvus.
    """

    def __init__(self, name, schema, **kwargs):
        """
        Construct a collection by the name, schema and other parameters.
        Connection information is contained in kwargs.

        :param name: the name of collection
        :type name: str

        :param schema: the schema of collection
        :type schema: class `schema.CollectionSchema`
        """
        self._name = name
        self._kwargs = kwargs
        self._schema = schema
        self._num_entities = 0
        self._description = kwargs.get("description", "")

    def _get_using(self):
        return self._kwargs.get("_using", "default")

    def _get_connection(self):
        return connections.get_connection(self._get_using())

    @property
    def schema(self):
        """
        Return the schema of collection.

        :return: Schema of collection
        :rtype: schema.CollectionSchema
        """
        return self._schema

    @schema.setter
    def schema(self, value):
        """
        Set the schema of collection.

        :param value: the schema of collection
        :type value: class `schema.CollectionSchema`
        """
        pass

    @property
    def description(self):
        """
        Return the description text about the collection.

        :return:
            Collection description text, return when operation is successful

        :rtype: str

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> field1 = FieldSchema(name="int64", type="int64", is_primary=False, description="int64")
        >>> schema = CollectionSchema(fields=[field1], auto_id=True, description="collection schema has a int64 field")
        >>> collection = Collection(name="test_collection", schema=schema, description="test get description")
        >>> collection.description
        'test get description'

        """
        return self._description

    @description.setter
    def description(self, value):
        """
        Set the description text of collection.

        :type value: str
        :param value: the description text of collection

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> field1 = FieldSchema(name="int64", type="int64", is_primary=False, description="int64")
        >>> schema = CollectionSchema(fields=[field1], auto_id=True, description="collection schema has a int64 field")
        >>> collection = Collection(name="test_collection", schema=schema, description="test get description")
        >>> collection.description
        'test get description'
        >>> collection.description = "test set description"
        >>> collection.description
        'test set description'
        """

        self._description = value

    @property
    def name(self):
        """
        Return the collection name.

        :return: Collection name, return when operation is successful
        :rtype: str

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> field1 = FieldSchema(name="int64", type="int64", is_primary=False, description="int64")
        >>> schema = CollectionSchema(fields=[field1], auto_id=True, description="test get collection name.")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> collection.name
        'test_collection'
        """
        return self._name

    @name.setter
    def name(self, value):
        """
        Set the name of collection.

        :param value: the name of collection
        :type value: str

        :example:
        >>> from pymilvus_orm.collection import Collection
        >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
        >>> field1 = FieldSchema(name="int64", type="int64", is_primary=False, description="This field type is int64.")
        >>> schema = CollectionSchema(fields=[field1], auto_id=True, description="test set collection name.")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> collection.name
        'test_collection'
        >>> collection.name = "set_collection"
        >>> collection.name
        'set_collection'
        """
        self._name = value

    # read-only
    @property
    def is_empty(self):
        """
        Return whether the collection is empty.

        :return: Whether the collection is empty
        :rtype: bool
        """
        if self._num_entities == 0:
            return True
        return False

    # read-only
    @property
    def num_entities(self):
        """
        Return the number of entities.

        :return: Number of entities in this collection.
        :rtype: int
        """
        conn = self._get_connection()
        status = conn.get_collection_states(db_name="", collection_name=self._name)
        self._num_entities = status["row_count"]
        return self._num_entities

    def drop(self, **kwargs):
        """
        Drop the collection, as well as its corresponding index files.
        """
        conn = self._get_connection()
        indexes = self.indexes()
        conn.drop_collection(self._name, timeout=kwargs.get("timeout", None))
        for index in indexes:
            conn.drop_index(self._name, index.field_name(), index.name(), timeout=kwargs.get("timeout", None),
                            kwargs=kwargs)
        return

    def load(self, field_names=None, index_names=None, partition_names=None, **kwargs):
        """
        Load the collection from disk to memory.

        :param field_names: The specified fields to load.
        :type  field_names: list[str]

        :param index_names: The specified indexes to load.
        :type  index_names: list[str]

        :param partition_names: The specified partitions to load.
        :type partition_names: list[str]
        """
        conn = self._get_connection()
        conn.load_collection("", self._name, kwargs)

    # TODO(yukun): release_collection in pymilvus need db_name, but not field_name
    def release(self, **kwargs):
        """
        Release the collection from memory.
        """
        conn = self._get_connection()
        # TODO(yukun): release_collection in pymilvus need db_name, but not field_name
        conn.release_collection("", self._name, kwargs)

    def insert(self, data, **kwargs):
        """
        Insert data into collection.

        :param data: The specified data to insert, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple, numpy.ndarray) object or pandas.DataFrame
        """
        pass

    def search(self, data, params, limit, expr="", partition_names=None, fields=None, **kwargs):
        """
        Vector similarity search with an optional boolean expression as filters.

        :param data: Data to search, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple, numpy.ndarray) object or pandas.DataFrame

        :param params: Search parameters
        :type  params: dict

        :param limit:
        :type  limit: int

        :param expr: Search expression
        :type  expr: str

        :param fields: The fields to return in the search result
        :type  fields: list[str]

        :return: A Search object, you can call its' `execute` method to get the search result
        :rtype: class `search.Search`
        """
        pass

    @property
    def partitions(self):
        """
        Return all partitions of the collection.

        :return: List of Partition object, return when operation is successful
        :rtype: list[Partition]
        """
        from .partition import Partition
        conn = self._get_connection()
        partition_strs = conn.list_partitions(self._name)
        partitions = []
        for partition in partition_strs:
            partitions.append(Partition(self, partition))
        return partitions

    def partition(self, partition_name):
        """
        Return the partition corresponding to name. Create a new one if not existed.

        :param partition_name: The name of the partition to create.
        :type  partition_name: str

        :return:Partition object corresponding to partition_name
        :rtype: Partition
        """
        from .partition import Partition
        conn = self._get_connection()
        return Partition(self, partition_name)

    def has_partition(self, partition_name):
        """
        Checks if a specified partition exists.

        :param partition_name: The name of the partition to check
        :type  partition_name: str

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur.
        :type  timeout: float

        :return: Whether a specified partition exists.
        :rtype: bool
        """
        conn = self._get_connection()
        return conn.has_partition(self._name, partition_name)

    def drop_partition(self, partition_name, **kwargs):
        """
        Drop the partition and its corresponding index files.

        :param partition_name: The name of the partition to drop.
        :type  partition_name: str
        """
        conn = self._get_connection()
        return conn.drop_partition(self._name, partition_name, timeout=kwargs.get("timeout", None))

    @property
    def indexes(self):
        """
        Return all indexes of the collection..

        :return: List of Index object, return when operation is successful
        :rtype: list[Index]
        """
        pass

    def index(self, index_name):
        """
        Return the index corresponding to name.

        :param index_name: The name of the index to create.
        :type  index_name: str

        :return:Index object corresponding to index_name
        :rtype: Index
        """
        # TODO(yukun): Need field name, but provide index name
        from .index import Index
        conn = self._get_connection()
        tmp_index = conn.describe_index(self._name, "")
        return Index(self, index_name, "", tmp_index.params)

    def create_index(self, field_name, index_name, index_params, **kwargs):
        """
        Create index on a specified column according to the index parameters. Return Index Object.

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_name: The name of the index to create.
        :type  index_name: str

        :param index_params: Indexing parameters.
        :type  index_params: dict
        """
        # TODO(yukun): Add index_name
        conn = self._get_connection()
        return conn.create_index(self._name, field_name, index_params, timeout=kwargs.get("timeout", None),
                                 kwargs=kwargs)

    def has_index(self, index_name):
        """
        Checks whether a specified index exists.

        :param index_name: The name of the index to check.
        :type  index_name: str

        :return: If specified index exists
        :rtype: bool
        """
        conn = self._get_connection()
        # TODO(yukun): Need field name, but provide index name
        if conn.describe_index(self._name, "") == None:
            return False
        return True

    def drop_index(self, index_name, **kwargs):
        """
        Drop index and its corresponding index files.

        :param index_name: The name of the partition to drop.
        :type  index_name: str
        """
        # TODO(yukun): Need field name
        conn = self._get_connection()
        conn.drop_index(self._name, "", index_name, timeout=kwargs.get("timeout", None), kwargs=kwargs)
