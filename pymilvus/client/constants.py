from ..grpc_gen.common_pb2 import ConsistencyLevel

LOGICAL_BITS = 18
LOGICAL_BITS_MASK = (1 << LOGICAL_BITS) - 1
EVENTUALLY_TS = 1
BOUNDED_TS = 2
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.Bounded
