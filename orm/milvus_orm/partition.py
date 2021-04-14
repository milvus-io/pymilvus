from . import connections
from .collection import Collection

class Partition(object):

    def __init__(self, collection, name, **kwargs):
        self._collection = collection
        self._name = name
        self._kwargs = kwargs

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

    def load(self, field_names=None, index_names=None, **kwargs):
        pass

    def release(self, **kwargs):
        pass

    def insert(self, data, **kwargs):
        pass

    def search(self, data, params, limit, expr=None, fields=None, **kwargs):
        pass
