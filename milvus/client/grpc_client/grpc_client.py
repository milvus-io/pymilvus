import sys
from grpc import aio

from grpc_gen import milvus_pb2_grpc
from grpc_gen import milvus_pb2 as grpc_types

from _client import MilvusClient


if sys.version_info[0] < 3 or sys.version_info[1] < 8:
    raise NotImplementedError("Unsupported python version")


class GRPCClient(MilvusClient):
    def __init__(self, uri, pre_ping, conn_id, client_tag):
        self._channel = None
        self._stub = None
        self._uri = uri
        self.status = None
        self._connected = False
        self._pre_ping = pre_ping

        self._client_tag = client_tag  # if self._pre_ping:
        # self._max_retry = max_retry
        # record
        self._id = conn_id
        # condition
        # self._condition = threading.Condition()
        self._request_id = 0

        # client hook
        # self._search_hook = SearchHook()
        # self._hybrid_search_hook = HybridSearchHook()
        # self._search_file_hook = SearchHook()

        # set server uri if object is initialized with parameter
        self._setup(uri, pre_ping)

    def _setup(self, uri, pre_ping=False):
        self._uri = uri
        print(self._uri)
        self._channel = aio.insecure_channel(self._uri)
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._channel)

    async def client_version(self):
        return

    async def server_status(self):
        pass

    async def server_version(self):
        cmd = grpc_types.Command(cmd="version")
        return await self._stub.Cmd(cmd)

    async def _cmd(self):
        pass

    async def create_collection(self, collection_name, fields, timeout):
        pass

    async def has_collection(self, collection_name, timeout, **kwargs):
        pass
