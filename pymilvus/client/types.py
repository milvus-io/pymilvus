from enum import IntEnum
from ..grpc_gen.common_pb2 import ConsistencyLevel
from .exceptions import InvalidConsistencyLevel


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

    def __init__(self, code=SUCCESS, message="Success"):
        self.code = code
        self.message = message

    def __repr__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attr_list))

    def __eq__(self, other):
        """
        Make Status comparable with self by code
        """
        if isinstance(other, int):
            return self.code == other

        return isinstance(other, self.__class__) and self.code == other.code

    def __ne__(self, other):
        return self != other

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

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
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

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_


class IndexState(IntEnum):
    IndexStateNone = 0
    Unissued = 1
    InProgress = 2
    Finished = 3
    Failed = 4
    Deleted = 5


class ErrorCode(IntEnum):
    Success = 0
    UnexpectedError = 1
    ConnectFailed = 2
    PermissionDenied = 3
    CollectionNotExists = 4
    IllegalArgument = 5
    IllegalDimension = 7
    IllegalIndexType = 8
    IllegalCollectionName = 9
    IllegalTOPK = 10
    IllegalRowRecord = 11
    IllegalVectorID = 12
    IllegalSearchResult = 13
    FileNotFound = 14
    MetaFailed = 15
    CacheFailed = 16
    CannotCreateFolder = 17
    CannotCreateFile = 18
    CannotDeleteFolder = 19
    CannotDeleteFile = 20
    BuildIndexError = 21
    IllegalNLIST = 22
    IllegalMetricType = 23
    OutOfMemory = 24
    IndexNotExist = 25


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
        elif s == State.Completed:
            return State.Completed
        else:
            return State.UndefiedState

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name_)

    def __str__(self):
        return self._name_


class CompactionState:
    """
    in_executing:   number of plans in executing
    in_timeout:     number of plans failed of timeout
    completed:      number of plans successfully completed
    """

    def __init__(self, compaction_id: int, state: State, in_executing: int, in_timeout: int, completed: int):
        self.compaction_id = compaction_id
        self.state = state
        self.in_executing = in_executing
        self.in_timeout = in_timeout
        self.completed = completed

    def __repr__(self):
        return f"""
CompactionState
 - compaction id: {self.compaction_id}
 - State: {self.state}
 - executing plan number: {self.in_executing}
 - timeout plan number: {self.in_timeout}
 - complete plan number: {self.completed}
"""


class Plan:
    def __init__(self, sources: list, target: int):
        self.sources = sources
        self.target = target

    def __repr__(self):
        return f"""
Plan:
 - sources: {self.sources}
 - target: {self.target}
"""


class CompactionPlans:
    def __init__(self, compaction_id: int, state: int):
        self.compaction_id = compaction_id
        self.state = State.new(state)
        self.plans = []

    def __repr__(self):
        return f"""
Compaction Plans:
 - compaction id: {self.compaction_id}
 - state: {self.state}
 - plans: {self.plans}
 """


def get_consistency_level(consistency_level):
    if isinstance(consistency_level, int):
        if consistency_level in ConsistencyLevel.values():
            return consistency_level
        raise InvalidConsistencyLevel(0, f"invalid consistency level: {consistency_level}")
    if isinstance(consistency_level, str):
        try:
            return ConsistencyLevel.Value(consistency_level)
        except ValueError as e:
            raise InvalidConsistencyLevel(0, f"invalid consistency level: {consistency_level}") from e
    raise InvalidConsistencyLevel(0, "invalid consistency level")
