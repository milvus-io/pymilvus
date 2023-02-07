import copy
import typing

from ...orm.connections import AbstractConnections
from ..client.grpc_handler import GrpcHandler as AsyncGrpcHandler


# pylint: disable=W0236
class Connections(AbstractConnections[AsyncGrpcHandler, typing.Awaitable[None]]):
    async def _disconnect(self, alias: str, *, remove_connection: bool):
        if alias in self._connected_alias:
            await self._connected_alias.pop(alias).close()
        if remove_connection:
            self._alias.pop(alias, None)

    async def _connect(self, alias, **kwargs):
        gh = AsyncGrpcHandler(**kwargs)

        await gh._channel_ready()
        kwargs.pop('password')
        kwargs.pop('secure', None)

        self._connected_alias[alias] = gh
        self._alias[alias] = copy.deepcopy(kwargs)


connections = Connections()
