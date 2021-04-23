from milvus import *


class MockMilvus:
    def __init__(self, host=None, port=None, handler="GRPC", pool="SingletonThread", **kwargs):
        pass

    def create_collection(self, collection_name, fields, timeout=None):
        pass

    def drop_collection(self, collection_name, timeout=None):
        pass

    def has_collection(self, collection_name, timeout=None):
        pass

    def describe_collection(self, collection_name, timeout=None):
        pass

    def load_collection(self, collection_name, timeout=None):
        pass

    def release_collection(self, collection_name, timeout=None):
        pass

    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        pass

    def list_collections(self, timeout=None):
        pass

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
