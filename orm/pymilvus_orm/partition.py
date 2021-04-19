from . import connections
from .collection import Collection

class Partition(object):

    def __init__(self, collection, name, **kwargs):
        self._collection = collection
        self._name = name
        self._kwargs = kwargs

    @property
    def description(self):
        """
        Return the description text.

        :return: Partition description text, return when operation is successful
        :rtype: str
        """
        pass

    @description.setter
    def description(self, value):
        pass

    @property
    def name(self):
        """
        Return the partition name.

        :return: Partition name, return when operation is successful
        :rtype: str
        """
        pass

    @name.setter
    def name(self, value):
        pass

    # read-only
    @property
    def is_empty(self):
        """
        Return whether the partition is empty

        :return: Whether the partition is empty
        :rtype: bool
        """
        pass

    # read-only
    @property
    def num_entities(self):
        """
        Return the number of entities.

        :return: Number of entities in this partition.
        :rtype: int
        """
        pass

    def drop(self, **kwargs):
        """
        Drop the partition, as well as its corresponding index files.

        :return: Number of entities in this partition.
        :rtype: int
        """
        pass

    def load(self, field_names=None, index_names=None, **kwargs):
        """
        Load the partition from disk to memory.

        :param field_names: The specified fields to load.
        :type  field_names: list[str]

        :param index_names: The specified indexes to load.
        :type  index_names: list[str]
        """
        pass

    def release(self, **kwargs):
        """
        Release the partition from memory.
        """
        pass

    def insert(self, data, **kwargs):
        """
        Insert data into partition.

        :param data: The specified data to insert, the dimension of data needs to align with column number
        :type  data: list-like(list, tuple, numpy.ndarray) object or pandas.DataFrame
        """
        pass

    def search(self, data, params, limit, expr=None, fields=None, **kwargs):
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

        :return: Query result. QueryResult is iterable and is a 2d-array-like class, the first dimension is
                 the number of vectors to query (nq), the second dimension is the number of topk.
        :rtype: QueryResult
        """
        pass
