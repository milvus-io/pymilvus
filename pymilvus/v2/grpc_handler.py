from abc import ABCMeta, abstractmethod
import logging


class IServer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def create_collection(self, collection_name, fields):
        pass

    @abstractmethod
    def drop_collection(self, collection_name):
        pass

    @abstractmethod
    def has_collection(self, collection_name):
        pass

    @abstractmethod
    def describe_collection(self, collection_name):
        pass

    @abstractmethod
    def load_collection(self, collection_name):
        pass

    @abstractmethod
    def release_collection(self, collection_name):
        pass

    @abstractmethod
    def get_collection_stats(self, collection_name):
        pass

    @abstractmethod
    def list_collections(self):
        pass

    @abstractmethod
    def create_partition(self, collection_name, partition_tag):
        pass

    @abstractmethod
    def drop_partition(self, collection_name, partition_tag):
        pass

    @abstractmethod
    def has_partition(self, collection_name, partition_tag):
        pass

    @abstractmethod
    def load_partitions(self, collection_name, partition_names):
        pass

    @abstractmethod
    def release_partitions(self, collection_name, partition_names):
        pass

    @abstractmethod
    def get_partition_stats(self, collection_name, partition_name):
        pass

    @abstractmethod
    def list_partitions(self, collection_name):
        pass


class MockServer(IServer):
    def __init__(self):
        super().__init__()
        self._collections = dict()
        self._collection_partitions = dict()
        self._collection_indexes = dict()

    def create_collection(self, collection_name, fields):
        if collection_name in self._collections:
            raise Exception(f"Create collection failed: collection {collection_name} exist")
        self._collections[collection_name] = fields
        self._collection_partitions[collection_name] = {'_default'}
        self._collection_indexes[collection_name] = []
        logging.debug(f"create_collection: {collection_name}")

    def drop_collection(self, collection_name):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        self._collections.pop(collection_name)
        self._collection_partitions.pop(collection_name)
        logging.debug(f"drop_collection: {collection_name}")

    def has_collection(self, collection_name):
        logging.debug(f"has_collection: {collection_name}")
        return collection_name in self._collections

    def describe_collection(self, collection_name):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"describe_collection: {collection_name}")
        return self._collections[collection_name]

    def load_collection(self, collection_name):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"load_collection: {collection_name}")

    def release_collection(self, collection_name):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"release_collection: {collection_name}")

    def get_collection_stats(self, collection_name):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        logging.debug(f"get_collection_stats: {collection_name}")
        return {'row_count': 0}

    def list_collections(self):
        logging.debug(f"list_collections")
        return list(self._collections.keys())

    def create_partition(self, collection_name, partition_tag):
        if collection_name not in self._collections:
            raise Exception(f"create partition failed: can't find collection: {collection_name}")
        if partition_tag in self._collection_partitions[collection_name]:
            raise Exception(f"create partition failed: partition name = {partition_tag} already exists")
        logging.debug(f"create_partition: {collection_name}, {partition_tag}")
        self._collection_partitions[collection_name].add(partition_tag)

    def drop_partition(self, collection_name, partition_tag):
        if collection_name not in self._collections:
            raise Exception(f"DropPartition failed: can't find collection: {collection_name}")
        if partition_tag not in self._collection_partitions[collection_name]:
            raise Exception(f"DropPartition failed: partition {partition_tag} does not exist")
        if partition_tag == "_default":
            raise Exception(f"DropPartition failed: default partition cannot be deleted")
        logging.debug(f"drop_partition: {collection_name}, {partition_tag}")
        self._collection_partitions[collection_name].remove(partition_tag)

    def has_partition(self, collection_name, partition_tag):
        if collection_name not in self._collections:
            raise Exception(f"HasPartition failed: can't find collection: {collection_name}")
        logging.debug(f"has_partition: {collection_name}, {partition_tag}")
        return partition_tag in self._collection_partitions[collection_name]

    def load_partitions(self, collection_name, partition_names):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        for partition_name in partition_names:
            if partition_name not in self._collection_partitions[collection_name]:
                raise Exception(f"partitionID of partitionName:{partition_name} can not be find")
        logging.debug(f"load_partition: {collection_name}, {partition_names}")

    def release_partitions(self, collection_name, partition_names):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        for partition_name in partition_names:
            if partition_name not in self._collection_partitions[collection_name]:
                raise Exception(f"partitionID of partitionName:{partition_name} can not be find")
        logging.debug(f"release_partition: {collection_name}, {partition_names}")

    def get_partition_stats(self, collection_name, partition_name):
        if collection_name not in self._collections:
            raise Exception(f"describe collection failed: can't find collection: {collection_name}")
        if partition_name not in self._collection_partitions[collection_name]:
            raise Exception(f"GetPartitionStatistics failed: partition {partition_name} does not exist")
        logging.debug(f"get_partition_stats: {partition_name}")
        return {'row_count': 0}

    def list_partitions(self, collection_name):
        if collection_name not in self._collections:
            raise Exception(f"can't find collection: {collection_name}")
        logging.debug(f"list_partitions: {collection_name}")
        return [e for e in self._collection_partitions[collection_name]]


class GRPCHandler:
    def __init__(self, server):
        if not isinstance(server, IServer):
            raise TypeError("Except an IServer")
        self._server = server

    def create_collection(self, collection_name, fields):
        return self._server.create_collection(collection_name, fields)


mock_server = MockServer()
GRPCHandler(mock_server)
