import grpc.aio

from ...client.grpc_handler import (
    AbstractGrpcHandler,
    Status,
    MilvusException,
)


class GrpcHandler(AbstractGrpcHandler[grpc.aio.Channel]):
    _insecure_channel = grpc.aio.insecure_channel
    _secure_channel = grpc.aio.secure_channel

    async def _channel_ready(self):
        if self._channel is None:
            raise MilvusException(
                Status.CONNECT_FAILED,
                'No channel in handler, please setup grpc channel first',
            )
        await self._channel.channel_ready()

    def _header_adder_interceptor(self, header, value):
        raise NotImplementedError  # TODO
