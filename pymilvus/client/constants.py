from ..grpc_gen import common_pb2

ConsistencyLevel = common_pb2.ConsistencyLevel

LOGICAL_BITS = 18
LOGICAL_BITS_MASK = (1 << LOGICAL_BITS) - 1
EVENTUALLY_TS = 1
BOUNDED_TS = 2
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.Bounded
