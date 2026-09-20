from pymilvus.grpc_gen import common_pb2

ConsistencyLevel = common_pb2.ConsistencyLevel

LOGICAL_BITS = 18
LOGICAL_BITS_MASK = (1 << LOGICAL_BITS) - 1
EVENTUALLY_TS = 1
BOUNDED_TS = 2
DEFAULT_CONSISTENCY_LEVEL = ConsistencyLevel.Bounded
DEFAULT_RESOURCE_GROUP = "__default_resource_group"
DYNAMIC_FIELD_NAME = "$meta"
REDUCE_STOP_FOR_BEST = "reduce_stop_for_best"
OFFSET = "offset"
MILVUS_LIMIT = "limit"
BATCH_SIZE = "batch_size"
CALC_DIST_L2 = "L2"
CALC_DIST_IP = "IP"
CALC_DIST_BM25 = "BM25"
CALC_DIST_HAMMING = "HAMMING"
CALC_DIST_TANIMOTO = "TANIMOTO"
CALC_DIST_JACCARD = "JACCARD"
CALC_DIST_COSINE = "COSINE"
CLUSTER_ID = "cluster_id"
COLLECTION_ID = "collection_id"
DEFAULT_SEARCH_EXTENSION_RATE = 10
EF = "ef"
FIELDS = "fields"
IS_PRIMARY = "is_primary"
GROUP_BY_FIELD = "group_by_field"
GROUP_SIZE = "group_size"
SEARCH_AGGREGATION = "search_aggregation"
RANK_GROUP_SCORER = "rank_group_scorer"
STRICT_GROUP_SIZE = "strict_group_size"
ORDER_BY_FIELDS = "order_by_fields"
JSON_PATH = "json_path"
JSON_TYPE = "json_type"
STRICT_CAST = "strict_cast"
ITERATOR_FIELD = "iterator"
QUERY_ITER_LAST_PK = "query_iter_last_pk"
QUERY_ITER_LAST_ELEMENT_OFFSET = "query_iter_last_element_offset"
ITERATOR_SESSION_CP_FILE = "iterator_cp_file"
QUERY_GROUP_BY_FIELDS = "group_by_fields"
ITERATOR_SESSION_TS_FIELD = "iterator_session_ts"
ITER_SEARCH_V2_KEY = "search_iter_v2"
ITER_SEARCH_BATCH_SIZE_KEY = "search_iter_batch_size"
ITER_SEARCH_LAST_BOUND_KEY = "search_iter_last_bound"
ITER_SEARCH_ID_KEY = "search_iter_id"
PAGE_RETAIN_ORDER_FIELD = "page_retain_order"
HINTS = "hints"

INT64_MAX = 9223372036854775807
MAX_BATCH_SIZE = 16384
MAX_FILTERED_IDS_COUNT_ITERATION = 100000
MAX_TRY_TIME = 20
METRIC_TYPE = "metric_type"
PARAMS = "params"
RADIUS = "radius"
RANGE_FILTER = "range_filter"
UNLIMITED = -1

RANKER_TYPE_RRF = "rrf"
RANKER_TYPE_WEIGHTED = "weighted"

GUARANTEE_TIMESTAMP = "guarantee_timestamp"

IS_EMBEDDING_LIST = "is_embedding_list"
