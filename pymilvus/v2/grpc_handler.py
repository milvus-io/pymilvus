from .milvus_server import IServer, GrpcServer


class GRPCHandler:
    def __init__(self, server):
        if not isinstance(server, IServer):
            raise TypeError("Except an IServer")
        self._server = server

    def create_collection(self, collection_name, fields, shards_num=2):
        return self._server.create_collection(collection_name, fields, shards_num)


server_instance = GrpcServer()
GRPCHandler(server_instance)
