import abc


class MilvusClient(abc.ABC):
    """Enable asynchronous RPC operations as milvus client"""

    @abc.abstractmethod
    async def client_version(self):
        pass

    @abc.abstractmethod
    async def server_status(self):
        pass

    @abc.abstractmethod
    async def server_version(self):
        pass

    @abc.abstractmethod
    async def _cmd(self):
        pass

    @abc.abstractmethod
    async def create_collection(self, collection_name, fields, timeout):
        pass

    @abc.abstractmethod
    async def has_collection(self, collection_name, timeout, **kwargs):
        pass

