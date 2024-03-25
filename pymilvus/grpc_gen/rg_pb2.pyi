from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceGroupLimit(_message.Message):
    __slots__ = ("node_num",)
    NODE_NUM_FIELD_NUMBER: _ClassVar[int]
    node_num: int
    def __init__(self, node_num: _Optional[int] = ...) -> None: ...

class ResourceGroupTransfer(_message.Message):
    __slots__ = ("resource_group",)
    RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    resource_group: str
    def __init__(self, resource_group: _Optional[str] = ...) -> None: ...

class ResourceGroupConfig(_message.Message):
    __slots__ = ("requests", "limits", "transfer_from", "transfer_to")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_FROM_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_TO_FIELD_NUMBER: _ClassVar[int]
    requests: ResourceGroupLimit
    limits: ResourceGroupLimit
    transfer_from: _containers.RepeatedCompositeFieldContainer[ResourceGroupTransfer]
    transfer_to: _containers.RepeatedCompositeFieldContainer[ResourceGroupTransfer]
    def __init__(self, requests: _Optional[_Union[ResourceGroupLimit, _Mapping]] = ..., limits: _Optional[_Union[ResourceGroupLimit, _Mapping]] = ..., transfer_from: _Optional[_Iterable[_Union[ResourceGroupTransfer, _Mapping]]] = ..., transfer_to: _Optional[_Iterable[_Union[ResourceGroupTransfer, _Mapping]]] = ...) -> None: ...
