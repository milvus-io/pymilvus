from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ResourceGroupLimit(_message.Message):
    __slots__ = ("nodeNum",)
    NODENUM_FIELD_NUMBER: _ClassVar[int]
    nodeNum: int
    def __init__(self, nodeNum: _Optional[int] = ...) -> None: ...

class ResourceGroupTransfer(_message.Message):
    __slots__ = ("resource_group",)
    RESOURCE_GROUP_FIELD_NUMBER: _ClassVar[int]
    resource_group: str
    def __init__(self, resource_group: _Optional[str] = ...) -> None: ...

class ResourceGroupConfig(_message.Message):
    __slots__ = ("requests", "limits", "to")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    requests: ResourceGroupLimit
    limits: ResourceGroupLimit
    to: _containers.RepeatedCompositeFieldContainer[ResourceGroupTransfer]
    def __init__(self, requests: _Optional[_Union[ResourceGroupLimit, _Mapping]] = ..., limits: _Optional[_Union[ResourceGroupLimit, _Mapping]] = ..., to: _Optional[_Iterable[_Union[ResourceGroupTransfer, _Mapping]]] = ..., **kwargs) -> None: ...
