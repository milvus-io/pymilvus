# Copyright (C) 2019-2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

from pymilvus.client import constants as client_constants

COMMON_TYPE_PARAMS = (
    "dim",
    "max_length",
    "max_capacity",
    "enable_match",
    "enable_analyzer",
    "analyzer_params",
    "multi_analyzer_params",
)

CALC_DIST_IDS = "ids"
CALC_DIST_FLOAT_VEC = "float_vectors"
CALC_DIST_BIN_VEC = "bin_vectors"
CALC_DIST_METRIC = "metric"
CALC_DIST_L2 = "L2"
CALC_DIST_IP = "IP"
CALC_DIST_BM25 = "BM25"
CALC_DIST_HAMMING = "HAMMING"
CALC_DIST_TANIMOTO = "TANIMOTO"
CALC_DIST_JACCARD = "JACCARD"
CALC_DIST_COSINE = "COSINE"
CALC_DIST_SQRT = "sqrt"
CALC_DIST_DIM = "dim"

OFFSET = client_constants.OFFSET
MILVUS_LIMIT = client_constants.MILVUS_LIMIT
BATCH_SIZE = client_constants.BATCH_SIZE
ID = "id"
TYPE = "type"
METRIC_TYPE = "metric_type"
PARAMS = "params"
DISTANCE = "distance"
RADIUS = "radius"
RANGE_FILTER = "range_filter"
FIELDS = client_constants.FIELDS
EF = "ef"
IS_PRIMARY = client_constants.IS_PRIMARY
REDUCE_STOP_FOR_BEST = client_constants.REDUCE_STOP_FOR_BEST
COLLECTION_ID = client_constants.COLLECTION_ID
ITERATOR_FIELD = client_constants.ITERATOR_FIELD
ITERATOR_SESSION_TS_FIELD = client_constants.ITERATOR_SESSION_TS_FIELD
QUERY_ITER_LAST_PK = client_constants.QUERY_ITER_LAST_PK
QUERY_ITER_LAST_ELEMENT_OFFSET = client_constants.QUERY_ITER_LAST_ELEMENT_OFFSET
DEFAULT_MAX_L2_DISTANCE = 99999999.0
DEFAULT_MIN_IP_DISTANCE = -99999999.0
DEFAULT_MAX_HAMMING_DISTANCE = 99999999.0
DEFAULT_MAX_TANIMOTO_DISTANCE = 99999999.0
DEFAULT_MAX_JACCARD_DISTANCE = 2.0
DEFAULT_MIN_COSINE_DISTANCE = -2.0
MAX_FILTERED_IDS_COUNT_ITERATION = 100000
INT64_MAX = client_constants.INT64_MAX
MAX_BATCH_SIZE: int = client_constants.MAX_BATCH_SIZE
DEFAULT_SEARCH_EXTENSION_RATE: int = 10
UNLIMITED: int = client_constants.UNLIMITED
MAX_TRY_TIME: int = 20
GUARANTEE_TIMESTAMP = client_constants.GUARANTEE_TIMESTAMP
ITERATOR_SESSION_CP_FILE = client_constants.ITERATOR_SESSION_CP_FILE
BM25_k1 = "bm25_k1"
BM25_b = "bm25_b"
