import grpc


class UnavailableRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE
