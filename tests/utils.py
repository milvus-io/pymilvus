import grpc


class MockGrpcError(grpc.RpcError):
    def __init__(self, code=1, details="error"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details
