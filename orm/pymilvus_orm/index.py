from . import connections


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
        from .collection import Collection
        self._collection = collection
        self._name = name
        self._field_name = field_name
        self._index_params = index_params
        self._kwargs = kwargs

        conn = self._get_connection()
        index = conn.describe_index(self._collection.name, self._field_name)
        if index is None:
            conn.create_index(self._collection.name, self._field_name, self._index_params)
        else:
            if self._index_params != index["params"]:
                raise Exception("The index already exists, but the index params is not the same as the passed in")

    def _get_using(self):
        return self._kwargs.get("_using", "default")

    def _get_connection(self):
        return connections.get_connection(self._get_using())

    @property
    def name(self):
        """
        Return the index name.

        :return: The name of index
        :rtype:  str
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def params(self):
        """
        Return the index params.

        :return: Index parameters
        :rtype:  dict
        """
        return self._index_params

    @params.setter
    def params(self, value):
        self._index_params = value

    # read-only
    @property
    def collection_name(self):
        """
        Return corresponding collection name.

        :return: Corresponding collection name.
        :rtype:  str
        """
        return self._collection.name

    @property
    def field_name(self):
        """
        Return corresponding column name.

        :return: Corresponding column name.
        :rtype:  str
        """
        return self._field_name

    def drop(self, **kwargs):
        """
        Drop index and its corresponding index files.
        """
        conn = self._get_connection()
        conn.drop_index(self._collection.name, self.field_name, self.name, **kwargs)
