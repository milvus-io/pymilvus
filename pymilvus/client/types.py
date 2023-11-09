import time
from enum import IntEnum
from typing import Any, ClassVar, Dict, List, TypeVar, Union

from pymilvus.exceptions import (
    AutoIDException,
    ExceptionsMessage,
    InvalidConsistencyLevel,
)
from pymilvus.grpc_gen import common_pb2
from pymilvus.grpc_gen import milvus_pb2 as milvus_types

Status = TypeVar("Status")
ConsistencyLevel = common_pb2.ConsistencyLevel


class Status:
    """
    :attribute code: int (optional) default as ok

    :attribute message: str (optional) current status message
    """

    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    CONNECT_FAILED = 2
    PERMISSION_DENIED = 3
    COLLECTION_NOT_EXISTS = 4
    ILLEGAL_ARGUMENT = 5
    ILLEGAL_RANGE = 6
    ILLEGAL_DIMENSION = 7
    # TODO in new error code, 8 is for RATE_LIMIT, 9 for FORCE_DENY
    ILLEGAL_INDEX_TYPE = 8
    ILLEGAL_COLLECTION_NAME = 9
    ILLEGAL_TOPK = 10
    ILLEGAL_ROWRECORD = 11
    ILLEGAL_VECTOR_ID = 12
    ILLEGAL_SEARCH_RESULT = 13
    FILE_NOT_FOUND = 14
    META_FAILED = 15
    CACHE_FAILED = 16
    CANNOT_CREATE_FOLDER = 17
    CANNOT_CREATE_FILE = 18
    CANNOT_DELETE_FOLDER = 19
    CANNOT_DELETE_FILE = 20
    BUILD_INDEX_ERROR = 21
    ILLEGAL_NLIST = 22
    ILLEGAL_METRIC_TYPE = 23
    OUT_OF_MEMORY = 24
    INDEX_NOT_EXIST = 25
    EMPTY_COLLECTION = 26

    def __init__(self, code: int = SUCCESS, message: str = "Success") -> None:
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        attr_list = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"{self.__class__.__name__}({', '.join(attr_list)})"

    def __eq__(self, other: Union[int, Status]):
        """Make Status comparable with self by code"""
        if isinstance(other, int):
            return self.code == other

        return isinstance(other, self.__class__) and self.code == other.code

    def OK(self):
        return self.code == Status.SUCCESS


class DataType(IntEnum):
    NONE = 0
    BOOL = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5

    FLOAT = 10
    DOUBLE = 11

    STRING = 20
    VARCHAR = 21
    ARRAY = 22
    JSON = 23

    BINARY_VECTOR = 100
    FLOAT_VECTOR = 101

    UNKNOWN = 999


class RangeType(IntEnum):
    LT = 0  # less than
    LTE = 1  # less than or equal
    EQ = 2  # equal
    GT = 3  # greater than
    GTE = 4  # greater than or equal
    NE = 5  # not equal


class IndexType(IntEnum):
    INVALID = 0
    FLAT = 1
    IVFLAT = 2
    IVF_SQ8 = 3
    RNSG = 4
    IVF_SQ8H = 5
    IVF_PQ = 6
    HNSW = 11
    ANNOY = 12

    # alternative name
    IVF_FLAT = IVFLAT
    IVF_SQ8_H = IVF_SQ8H

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name_}>"

    def __str__(self) -> str:
        return self._name_


class MetricType(IntEnum):
    INVALID = 0
    L2 = 1
    IP = 2
    # Only supported for byte vectors
    HAMMING = 3
    JACCARD = 4
    TANIMOTO = 5
    #
    SUBSTRUCTURE = 6
    SUPERSTRUCTURE = 7

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name_}>"

    def __str__(self) -> str:
        return self._name_


class IndexState(IntEnum):
    IndexStateNone = 0
    Unissued = 1
    InProgress = 2
    Finished = 3
    Failed = 4
    Deleted = 5


class PlaceholderType(IntEnum):
    NoneType = 0
    BinaryVector = 100
    FloatVector = 101


class State(IntEnum):
    """
    UndefiedState:  Unknown
    Executing:      indicating this compaction has undone plans.
    Completed:      indicating all the plans of this compaction are done,
                    no matter successful or not.
    """

    UndefiedState = 0
    Executing = 1
    Completed = 2

    @staticmethod
    def new(s: int):
        if s == State.Executing:
            return State.Executing
        if s == State.Completed:
            return State.Completed
        return State.UndefiedState

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name_}>"

    def __str__(self) -> str:
        return self._name_


class LoadState(IntEnum):
    """
    NotExist: collection or partition isn't existed
    NotLoad:  collection or partition isn't loaded
    Loading:  collection or partition is loading
    Loaded:   collection or partition is loaded
    """

    NotExist = 0
    NotLoad = 1
    Loading = 2
    Loaded = 3

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name_}>"

    def __str__(self) -> str:
        return self._name_


class CompactionState:
    """
    in_executing: number of plans in executing
    in_timeout:   number of plans failed of timeout
    completed:    number of plans successfully completed
    """

    def __init__(
        self,
        compaction_id: int,
        state: State,
        in_executing: int,
        in_timeout: int,
        completed: int,
    ) -> None:
        self.compaction_id = compaction_id
        self.state = state
        self.in_executing = in_executing
        self.in_timeout = in_timeout
        self.completed = completed

    def __repr__(self) -> str:
        return f"""
CompactionState
 - compaction id: {self.compaction_id}
 - State: {self.state}
 - executing plan number: {self.in_executing}
 - timeout plan number: {self.in_timeout}
 - complete plan number: {self.completed}
"""


class Plan:
    def __init__(self, sources: list, target: int) -> None:
        self.sources = sources
        self.target = target

    def __repr__(self) -> str:
        return f"""
Plan:
 - sources: {self.sources}
 - target: {self.target}
"""


class CompactionPlans:
    def __init__(self, compaction_id: int, state: int) -> None:
        self.compaction_id = compaction_id
        self.state = State.new(state)
        self.plans = []

    def __repr__(self) -> str:
        return f"""
Compaction Plans:
 - compaction id: {self.compaction_id}
 - state: {self.state}
 - plans: {self.plans}
 """


def cmp_consistency_level(l1: Union[str, int], l2: Union[str, int]):
    if isinstance(l1, str):
        try:
            l1 = ConsistencyLevel.Value(l1)
        except ValueError:
            return False

    if isinstance(l2, str):
        try:
            l2 = ConsistencyLevel.Value(l2)
        except ValueError:
            return False

    if isinstance(l1, int) and l1 not in ConsistencyLevel.values():
        return False

    if isinstance(l2, int) and l2 not in ConsistencyLevel.values():
        return False

    return l1 == l2


def get_consistency_level(consistency_level: Union[str, int]):
    if isinstance(consistency_level, int):
        if consistency_level in ConsistencyLevel.values():
            return consistency_level
        raise InvalidConsistencyLevel(message=f"invalid consistency level: {consistency_level}")
    if isinstance(consistency_level, str):
        try:
            return ConsistencyLevel.Value(consistency_level)
        except ValueError as e:
            raise InvalidConsistencyLevel(
                message=f"invalid consistency level: {consistency_level}"
            ) from e
    raise InvalidConsistencyLevel(message="invalid consistency level")


class Shard:
    def __init__(self, channel_name: str, shard_nodes: list, shard_leader: int) -> None:
        self._channel_name = channel_name
        self._shard_nodes = set(shard_nodes)
        self._shard_leader = shard_leader

    def __repr__(self) -> str:
        return (
            f"Shard: <channel_name:{self.channel_name}>, "
            f"<shard_leader:{self.shard_leader}>, <shard_nodes:{self.shard_nodes}>"
        )

    @property
    def channel_name(self) -> str:
        return self._channel_name

    @property
    def shard_nodes(self):
        return self._shard_nodes

    @property
    def shard_leader(self) -> int:
        return self._shard_leader


class Group:
    def __init__(
        self,
        group_id: int,
        shards: List[str],
        group_nodes: List[tuple],
        resource_group: str,
        num_outbound_node: dict,
    ) -> None:
        self._id = group_id
        self._shards = shards
        self._group_nodes = tuple(group_nodes)
        self._resource_group = resource_group
        self._num_outbound_node = num_outbound_node

    def __repr__(self) -> str:
        return (
            f"Group: <group_id:{self.id}>, <group_nodes:{self.group_nodes}>, "
            f"<shards:{self.shards}>, <resource_group: {self.resource_group}>, "
            f"<num_outbound_node: {self.num_outbound_node}>"
        )

    @property
    def id(self):
        return self._id

    @property
    def group_nodes(self):
        return self._group_nodes

    @property
    def shards(self):
        return self._shards

    @property
    def resource_group(self):
        return self._resource_group

    @property
    def num_outbound_node(self):
        return self._num_outbound_node


class Replica:
    """
    Replica groups:
        - Group: <group_id:2>, <group_nodes:(1, 2, 3)>,
            <shards:[Shard: <shard_id:10>,
                <channel_name:channel-1>,
                <shard_leader:1>,
                <shard_nodes:(1, 2, 3)>]>
        - Group: <group_id:2>, <group_nodes:(1, 2, 3)>,
            <shards:[Shard:
                <shard_id:10>,
                <channel_name:channel-1>,
                <shard_leader:1>,
                <shard_nodes:(1, 2, 3)>]>
    """

    def __init__(self, groups: list) -> None:
        self._groups = groups

    def __repr__(self) -> str:
        s = "Replica groups:"
        for g in self.groups:
            s += f"\n- {g}"
        return s

    @property
    def groups(self):
        return self._groups


class BulkInsertState:
    """enum states of bulk insert task"""

    ImportPending = 0
    ImportFailed = 1
    ImportStarted = 2
    ImportPersisted = 5
    ImportCompleted = 6
    ImportFailedAndCleaned = 7
    ImportUnknownState = 100

    """pre-defined keys of bulk insert task info"""
    FAILED_REASON = "failed_reason"
    IMPORT_FILES = "files"
    IMPORT_COLLECTION = "collection"
    IMPORT_PARTITION = "partition"
    IMPORT_PROGRESS = "progress_percent"

    """
    Bulk insert state example:
        - taskID    : 44353845454358,
        - state     : "BulkLoadPersisted",
        - row_count : 1000,
        - infos     : {"files": "rows.json",
                       "collection": "c1",
                       "partition": "",
                       "failed_reason": ""},
        - id_list   : [44353845455401, 44353845456401]
        - create_ts : 1661398759,
    """

    state_2_state: ClassVar[Dict] = {
        common_pb2.ImportPending: ImportPending,
        common_pb2.ImportFailed: ImportFailed,
        common_pb2.ImportStarted: ImportStarted,
        common_pb2.ImportPersisted: ImportPersisted,
        common_pb2.ImportCompleted: ImportCompleted,
        common_pb2.ImportFailedAndCleaned: ImportFailedAndCleaned,
    }

    state_2_name: ClassVar[Dict] = {
        ImportPending: "Pending",
        ImportFailed: "Failed",
        ImportStarted: "Started",
        ImportPersisted: "Persisted",
        ImportCompleted: "Completed",
        ImportFailedAndCleaned: "Failed and cleaned",
        ImportUnknownState: "Unknown",
    }

    def __init__(
        self,
        task_id: int,
        state: State,
        row_count: int,
        id_ranges: list,
        infos: Dict,
        create_ts: int,
    ):
        self._task_id = task_id
        self._state = state
        self._row_count = row_count
        self._id_ranges = id_ranges
        self._create_ts = create_ts

        self._infos = {kv.key: kv.value for kv in infos}

    def __repr__(self) -> str:
        fmt = """<Bulk insert state:
    - taskID          : {},
    - state           : {},
    - row_count       : {},
    - infos           : {},
    - id_ranges       : {},
    - create_ts       : {}
>"""
        return fmt.format(
            self._task_id,
            self.state_name,
            self.row_count,
            self.infos,
            self.id_ranges,
            self.create_time_str,
        )

    @property
    def task_id(self):
        """
        Return unique id of this task.
        """
        return self._task_id

    @property
    def row_count(self):
        """
        If the task is finished, this value is the number of rows imported.
        If the task is not finished, this value is the number of rows parsed.
        """
        return self._row_count

    @property
    def state(self):
        return self.state_2_state.get(self._state, BulkInsertState.ImportUnknownState)

    @property
    def state_name(self) -> str:
        return self.state_2_name.get(self._state, "unknown state")

    @property
    def id_ranges(self):
        """
        auto generated id ranges if the primary key is auto generated

        the id list of response is id ranges
        for example, if the response return [1, 100, 200, 250]
        the full id list should be [1, 2, 3 ... , 99, 100, 200, 201, 202 ... , 249, 250]
        """
        return self._id_ranges

    @property
    def ids(self):
        """
        auto generated ids if the primary key is auto generated

        the id list of response is id ranges
        for example, if the response return [1, 100, 200, 250], the id ranges: [1,100),[200,250)
        the full id list should be [1, 2, 3 ... , 99, 200, 201, 202 ... , 249]
        """

        if len(self._id_ranges) % 2 != 0:
            raise AutoIDException(message=ExceptionsMessage.AutoIDIllegalRanges)

        ids = []
        for i in range(int(len(self._id_ranges) / 2)):
            begin = self._id_ranges[i * 2]
            end = self._id_ranges[i * 2 + 1]
            for j in range(begin, end):
                ids.append(j)

        return ids

    @property
    def infos(self):
        """more informations about the task, progress percentage, file path, failed reason, etc."""
        return self._infos

    @property
    def failed_reason(self):
        """failed reason of the bulk insert task."""
        return self._infos.get(BulkInsertState.FAILED_REASON, "")

    @property
    def files(self):
        """data files of the bulk insert task."""
        return self._infos.get(BulkInsertState.IMPORT_FILES, "")

    @property
    def collection_name(self):
        """target collection's name of the bulk insert task."""
        return self._infos.get(BulkInsertState.IMPORT_COLLECTION, "")

    @property
    def partition_name(self):
        """target partition's name of the bulk insert task."""
        return self._infos.get(BulkInsertState.IMPORT_PARTITION, "")

    @property
    def create_timestamp(self):
        """the integer timestamp when this task is created."""
        return self._create_ts

    @property
    def create_time_str(self):
        """A readable string converted from the timestamp when this task is created."""
        ts = time.localtime(self._create_ts)
        return time.strftime("%Y-%m-%d %H:%M:%S", ts)

    @property
    def progress(self):
        """working progress percent value."""
        percent = self._infos.get(BulkInsertState.IMPORT_PROGRESS, "0")
        return int(percent)


class GrantItem:
    def __init__(self, entity: Any) -> None:
        self._object = entity.object.name
        self._object_name = entity.object_name
        self._db_name = entity.db_name
        self._role_name = entity.role.name
        self._grantor_name = entity.grantor.user.name
        self._privilege = entity.grantor.privilege.name

    def __repr__(self) -> str:
        return (
            f"GrantItem: <object:{self.object}>, <object_name:{self.object_name}>, "
            f"<db_name:{self.db_name}>, "
            f"<role_name:{self.role_name}>, <grantor_name:{self.grantor_name}>, "
            f"<privilege:{self.privilege}>"
        )

    @property
    def object(self):
        return self._object

    @property
    def object_name(self):
        return self._object_name

    @property
    def db_name(self):
        return self._db_name

    @property
    def role_name(self):
        return self._role_name

    @property
    def grantor_name(self):
        return self._grantor_name

    @property
    def privilege(self):
        return self._privilege


class GrantInfo:
    """
    GrantInfo groups:
    - GrantItem: <object:Collection>, <object_name:foo>, <role_name:x>,
        <grantor_name:root>, <privilege:Load>
    - GrantItem: <object:Global>, <object_name:*>, <role_name:x>,
        <grantor_name:root>, <privilege:CreateCollection>
    """

    def __init__(self, entities: List[milvus_types.RoleEntity]) -> None:
        groups = []
        for entity in entities:
            if isinstance(entity, milvus_types.GrantEntity):
                groups.append(GrantItem(entity))

        self._groups = groups

    def __repr__(self) -> str:
        s = "GrantInfo groups:"
        for g in self.groups:
            s += f"\n- {g}"
        return s

    @property
    def groups(self):
        return self._groups


class UserItem:
    def __init__(self, username: str, entities: List[milvus_types.RoleEntity]) -> None:
        self._username = username
        roles = []
        for entity in entities:
            if isinstance(entity, milvus_types.RoleEntity):
                roles.append(entity.name)
        self._roles = tuple(roles)

    def __repr__(self) -> str:
        return f"UserItem: <username:{self.username}>, <roles:{self.roles}>"

    @property
    def username(self):
        return self._username

    @property
    def roles(self):
        return self._roles


class UserInfo:
    """
    UserInfo groups:
    - UserItem: <username:root>, <roles:('admin', 'public')>
    """

    def __init__(self, results: List[milvus_types.UserResult]):
        groups = []
        for result in results:
            if isinstance(result, milvus_types.UserResult):
                groups.append(UserItem(result.user.name, result.roles))

        self._groups = groups

    def __repr__(self) -> str:
        s = "UserInfo groups:"
        for g in self.groups:
            s += f"\n- {g}"
        return s

    @property
    def groups(self):
        return self._groups


class RoleItem:
    def __init__(self, role_name: str, entities: List[milvus_types.UserEntity]):
        self._role_name = role_name
        users = []
        for entity in entities:
            if isinstance(entity, milvus_types.UserEntity):
                users.append(entity.name)
        self._users = tuple(users)

    def __repr__(self) -> str:
        return f"RoleItem: <role_name:{self.role_name}>, <users:{self.users}>"

    @property
    def role_name(self):
        return self._role_name

    @property
    def users(self):
        return self._users


class RoleInfo:
    """
    RoleInfo groups:
    - UserItem: <role_name:admin>, <users:('root',)>
    """

    def __init__(self, results: List[milvus_types.RoleResult]) -> None:
        groups = []
        for result in results:
            if isinstance(result, milvus_types.RoleResult):
                groups.append(RoleItem(result.role.name, result.users))

        self._groups = groups

    def __repr__(self) -> str:
        s = "RoleInfo groups:"
        for g in self.groups:
            s += f"\n- {g}"
        return s

    @property
    def groups(self):
        return self._groups


class ResourceGroupInfo:
    def __init__(self, resource_group: Any) -> None:
        self._name = resource_group.name
        self._capacity = resource_group.capacity
        self._num_available_node = resource_group.num_available_node
        self._num_loaded_replica = resource_group.num_loaded_replica
        self._num_outgoing_node = resource_group.num_outgoing_node
        self._num_incoming_node = resource_group.num_incoming_node

    def __repr__(self) -> str:
        return f"""ResourceGroupInfo:
<name:{self.name}>,
<capacity:{self.capacity}>,
<num_available_node:{self.num_available_node}>,
<num_loaded_replica:{self.num_loaded_replica}>,
<num_outgoing_node:{self.num_outgoing_node}>,
<num_incoming_node:{self.num_incoming_node}>"""

    @property
    def name(self):
        return self._name

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_available_node(self):
        return self._num_available_node

    @property
    def num_loaded_replica(self):
        return self._num_loaded_replica

    @property
    def num_outgoing_node(self):
        return self._num_outgoing_node

    @property
    def num_incoming_node(self):
        return self._num_incoming_node
