from . import connections

class Collection(object):
    """
    This is a class coresponding to collection in milvus.
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
        Return the description text.

        :return: Collection description text, return when operation is successful
        :rtype: str
        """
        pass

    @description.setter
    def description(self, value):
        """
        Set the description text of collection.

        :param value: the description text of collection
        :type value: str
        """
        pass

    @property
    def name(self):
        """
        Return the collection name.

        :return: Collection name, return when operation is successful
        :rtype: str
        """
        pass

    @name.setter
    def name(self, value):
        """
        Set the name of collection.

        :param value: the name of collection
        :type value: str
        """
        pass

    # read-only
    @property
    def is_empty(self):
        """
        Return whether the collection is empty.

        :return: Whether the collection is empty.
        :rtype: bool
        """
        pass

    # read-only
    @property
    def num_entities(self):
        """
        Return the number of entities.

        :return: Number of entities in this collection.
        :rtype: int
        """
        pass

    def drop(self, **kwargs):
        """
        Drop the collection, as well as its corresponding index files.

        :return: Number of entities in this collection.
        :rtype: int
        """
        conn = self._get_connection()
        return conn.drop_collection(self._name, timeout=kwargs.get("timeout", None))

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
        pass

    def release(self, **kwargs):
        """
        Release the collection from memory.
        """
        pass

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
        pass

    def partition(self, partition_name):
        """
        Return the partition corresponding to name. Create a new one if not existed.

        :param partition_name: The name of the partition to create.
        :type  partition_name: str

        :return:Partition object corresponding to partition_name
        :rtype: Partition
        """
        pass

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
        pass

    def drop_partition(self, partition_name, **kwargs):
        """
        Drop the partition and its corresponding index files.

        :param partition_name: The name of the partition to drop.
        :type  partition_name: str
        """
        pass

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
        pass

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
        pass

    def has_index(self, index_name):
        """
        Checks whether a specified index exists.

        :param index_name: The name of the index to check.
        :type  index_name: str

        :return: If specified index exists
        :rtype: bool
        """
        pass

    def drop_index(self, index_name, **kwargs):
        """
        Drop index and its corresponding index files.

        :param index_name: The name of the partition to drop.
        :type  index_name: str
        """
        pass
