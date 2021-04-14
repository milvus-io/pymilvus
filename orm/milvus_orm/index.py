from . import connections
from .collection import Collection

class Index(object):

    def __init__(self, collection, name, field_name, index_params, **kwargs):
        self._collection = collection
        self._name = name
        self._field_name = field_name
        self._index_params = index_params
        self._kwargs = kwargs

    @property
    def name(self):
        pass

    @name.setter
    def name(self, value):
        pass

    @property
    def params(self):
        pass

    @params.setter
    def params(self, value):
        pass

    # read-only
    @property
    def collection_name(self):
        pass

    @property
    def field_name(self):
        pass

    def drop(self, **kwargs):
        pass
