from . import connections
from .collection import Collection

class Index(object):

    def __init__(self, collection, name, field_name, index_params, **kwargs):
        """
        Create index on a specified column according to the index parameters.

        :param collection: The collection of index
        :type  collection: Collection

        :param name: The name of index
        :type  name: str

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_params: Indexing parameters.
        :type  index_params: dict
        """
        self._collection = collection
        self._name = name
        self._field_name = field_name
        self._index_params = index_params
        self._kwargs = kwargs

    @property
    def name(self):
        """
        Return the index name.

        :return: The name of index
        :rtype:  str
        """
        pass

    @name.setter
    def name(self, value):
        pass

    @property
    def params(self):
        """
        Return the index params.

        :return: Index parameters
        :rtype:  dict
        """
        pass

    @params.setter
    def params(self, value):
        pass

    # read-only
    @property
    def collection_name(self):
        """
        Return corresponding collection name.

        :return: Corresponding collection name.
        :rtype:  str
        """
        pass

    @property
    def field_name(self):
        """
        Return corresponding column name.

        :return: Corresponding column name.
        :rtype:  str
        """
        pass

    def drop(self, **kwargs):
        """
        Drop index and its corresponding index files.
        """
        pass
