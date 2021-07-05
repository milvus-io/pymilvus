import pdb

from pymilvus import *
import logging

from mock_result import MockMutationResult


class MockMilvus:
    def __init__(self, host=None, port=None, handler="GRPC", pool="SingletonThread", **kwargs):
        self._collections = dict()
        self._collection_partitions = dict()
        self._collection_indexes = dict()

    def create_collection(self, collection_name, fields, timeout=None, **kwargs):
        if collection_name in self._collections:
            raise BaseException(1, f"Create collection failed: collection {collection_name} exist")
        self._collections[collection_name] = fields
        self._collection_partitions[collection_name] = {'_default'}
        self._collection_indexes[collection_name] = []
        logging.debug(f"create_collection: {collection_name}")

    def drop_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        self._collections.pop(collection_name)
        self._collection_partitions.pop(collection_name)
        logging.debug(f"drop_collection: {collection_name}")

    def has_collection(self, collection_name, timeout=None):
        logging.debug(f"has_collection: {collection_name}")
        return collection_name in self._collections

    def describe_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"describe_collection: {collection_name}")
        return self._collections[collection_name]

    def load_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"load_collection: {collection_name}")

    def release_collection(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"release_collection: {collection_name}")

    def get_collection_stats(self, collection_name, timeout=None, **kwargs):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"get_collection_stats: {collection_name}")
        return {'row_count': 0}

    def list_collections(self, timeout=None):
        logging.debug(f"list_collections")
        return list(self._collections.keys())

    def create_partition(self, collection_name, partition_tag, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"create partition failed: can't find collection: {collection_name}")
        if partition_tag in self._collection_partitions[collection_name]:
            raise BaseException(1, f"create partition failed: partition name = {partition_tag} already exists")
        logging.debug(f"create_partition: {collection_name}, {partition_tag}")
        self._collection_partitions[collection_name].add(partition_tag)

    def drop_partition(self, collection_name, partition_tag, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"DropPartition failed: can't find collection: {collection_name}")
        if partition_tag not in self._collection_partitions[collection_name]:
            raise BaseException(1, f"DropPartition failed: partition {partition_tag} does not exist")
        if partition_tag == "_default":
            raise BaseException(1, f"DropPartition failed: default partition cannot be deleted")
        logging.debug(f"drop_partition: {collection_name}, {partition_tag}")
        self._collection_partitions[collection_name].remove(partition_tag)

    def has_partition(self, collection_name, partition_tag, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"HasPartition failed: can't find collection: {collection_name}")
        logging.debug(f"has_partition: {collection_name}, {partition_tag}")
        return partition_tag in self._collection_partitions[collection_name]

    def load_partitions(self, collection_name, partition_names, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        for partition_name in partition_names:
            if partition_name not in self._collection_partitions[collection_name]:
                raise BaseException(1, f"partitionID of partitionName:{partition_name} can not be find")
        logging.debug(f"load_partition: {collection_name}, {partition_names}")

    def release_partitions(self, collection_name, partition_names, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        for partition_name in partition_names:
            if partition_name not in self._collection_partitions[collection_name]:
                raise BaseException(1, f"partitionID of partitionName:{partition_name} can not be find")
        logging.debug(f"release_partition: {collection_name}, {partition_names}")

    def get_partition_stats(self, collection_name, partition_name, timeout=None, **kwargs):
        if collection_name not in self._collections:
            raise BaseException(1, f"describe collection failed: can't find collection: {collection_name}")
        if partition_name not in self._collection_partitions[collection_name]:
            raise BaseException(1, f"GetPartitionStatistics failed: partition {partition_name} does not exist")
        logging.debug(f"get_partition_stats: {partition_name}")
        return {'row_count': 0}

    def list_partitions(self, collection_name, timeout=None):
        if collection_name not in self._collections:
            raise BaseException(1, f"can't find collection: {collection_name}")
        logging.debug(f"list_partitions: {collection_name}")
        return [e for e in self._collection_partitions[collection_name]]

    def create_index(self, collection_name, field_name, params, timeout=None, **kwargs):
        logging.debug(f"create_index: {collection_name}, {field_name}, {params}")
        index = {"field_name": field_name, "params": params}
        self._collection_indexes[collection_name].append(index)

    def drop_index(self, collection_name, field_name, timeout=None):
        logging.debug(f"drop_index: {collection_name}, {field_name}")
        self._collection_indexes[collection_name] = []

    def describe_index(self, collection_name, index_name="", timeout=None):
        logging.debug(f"describe_index: {collection_name}, {index_name}")
        if self._collection_indexes.get(collection_name) is None:
            return
        indexes = self._collection_indexes[collection_name].copy()
        if len(indexes) != 0:
            return indexes[0]

    def insert(self, collection_name, entities, ids=None, partition_tag=None, timeout=None, **kwargs):
        return MockMutationResult()

    def flush(self, collection_names=None, timeout=None, **kwargs):
        pass

    def search(self, collection_name, dsl, partition_tags=None, fields=None, timeout=None, **kwargs):
        pass

    def load_collection_progress(self, collection_name, timeout=None, **kwargs):
        return {'num_loaded_entities':3000, 'num_total_entities': 5000}

    def load_partitions_progress(self, collection_name, partition_names, timeout=None, **kwargs):
        return {'num_loaded_entities':3000, 'num_total_entities': 5000}

    def wait_for_loading_collection_complete(self, collection_name, timeout=None, **kwargs):
        pass

    def wait_for_loading_partitions_complete(self, collection_name, partition_names, timeout=None, **kwargs):
        pass

    def get_index_build_progress(self, collection_name, index_name, timeout=None, **kwargs):
        return {'total_rows':5000,'indexed_rows':3000}

    def wait_for_creating_index(self, collection_name, index_name, timeout=None, **kwargs):
        pass

    def close(self):
        pass

Milvus = MockMilvus
