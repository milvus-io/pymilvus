from milvus import *


class MockMilvus:
    def __init__(self, host=None, port=None, handler="GRPC", pool="SingletonThread", **kwargs):
        self._collections = dict()

    def create_collection(self, collection_name, fields, timeout=None):
        if collection_name in self._collections:
            raise BaseException(1, f"Create collection failed: collection {collection_name} exist")
        self._collections[collection_name] = fields

    def drop_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        self._collections.pop(collection_name)

    def has_collection(self, collection_name, timeout=None):
        return collection_name in self._collections

    def describe_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        return self._collections[collection_name]

    def load_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")

    def release_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")

    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        return {'row_count': 0}

    def list_collections(self, timeout=None):
        return self._collections.keys()

    def create_partition(self, collection_name, partition_tag, timeout=None):
        pass

    def drop_partition(self, collection_name, partition_tag, timeout=None):
        pass

    def has_partition(self, collection_name, partition_tag, timeout=None):
        pass

    def load_partitions(self, collection_name, partition_names, timeout=None):
        pass

    def release_partitions(self, collection_name, partition_names, timeout=None):
        pass

    def list_partitions(self, collection_name, timeout=None):
        pass

    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        pass

    def drop_index(self, collection_name, field_name, timeout=None):
        pass

    def describe_index(self, collection_name, field_name, timeout=None):
        pass

    def insert(self, collection_name, entities, ids=None, partition_tag=None, timeout=None, **kwargs):
        pass

    def flush(self, collection_names=None, timeout=None, **kwargs):
        pass

    def search(self, collection_name, dsl, partition_tags=None, fields=None, timeout=None, **kwargs):
        pass


Milvus = MockMilvus
