from . import connections

class Collection(object):
    """This is a class coresponding to collection in milvus.
    """

    def __init__(self, name, schema, **kwargs):
        """Construct a collection by the name, schema and other parameters.
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
        return self._schema

    @schema.setter
    def schema(self, value):
        pass

    @property
    def description(self):
        pass

    @description.setter
    def description(self, value):
        pass

    @property
    def name(self):
        pass

    @name.setter
    def name(self, value):
        pass

    # read-only
    @property
    def is_empty(self):
        pass

    # read-only
    @property
    def num_entities(self):
        pass

    def drop(self, **kwargs):
        pass

    def load(self, field_names=None, index_names=None, partition_names=None, **kwargs):
        pass

    def release(self, **kwargs):
        pass

    def insert(self, data, **kwargs):
        pass

    def search(self, data, params, limit, expr="", partition_names=None, fields=None, **kwargs):
        pass

    @property
    def partitions(self):
        pass

    def partition(self, partition_name):
        pass

    def has_partition(self, partition_name):
        pass

    def drop_partition(self, partition_name, **kwargs):
        pass

    @property
    def indexes(self):
        pass

    def index(self, index_name):
        pass

    def create_index(self, field_name, index_name, index_params, **kwargs):
        pass

    def has_index(self, index_name):
        pass

    def drop_index(self, index_name, **kwargs):
        pass
