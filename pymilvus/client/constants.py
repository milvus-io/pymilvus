from pymilvus.grpc_gen import common_pb2

ConsistencyLevel = common_pb2.ConsistencyLevel

LOGICAL_BITS = 18
LOGICAL_BITS_MASK = (1 << LOGICAL_BITS) - 1
EVENTUALLY_TS = 1
BOUNDED_TS = 2
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.Bounded
DEFAULT_RESOURCE_GROUP = "__default_resource_group"
REDUCE_STOP_FOR_BEST = "reduce_stop_for_best"
GROUP_BY_FIELD = "group_by_field"
GROUP_SIZE = "group_size"
RANK_GROUP_SCORER = "rank_group_scorer"
GROUP_STRICT_SIZE = "group_strict_size"
ITERATOR_FIELD = "iterator"
ITERATOR_SESSION_TS_FIELD = "iterator_session_ts"
PAGE_RETAIN_ORDER_FIELD = "page_retain_order"

RANKER_TYPE_RRF = "rrf"
RANKER_TYPE_WEIGHTED = "weighted"

GUARANTEE_TIMESTAMP = "guarantee_timestamp"
