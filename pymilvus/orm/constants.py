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
CALC_DIST_L2 = client_constants.CALC_DIST_L2
CALC_DIST_IP = client_constants.CALC_DIST_IP
CALC_DIST_BM25 = client_constants.CALC_DIST_BM25
CALC_DIST_HAMMING = client_constants.CALC_DIST_HAMMING
CALC_DIST_TANIMOTO = client_constants.CALC_DIST_TANIMOTO
CALC_DIST_JACCARD = client_constants.CALC_DIST_JACCARD
CALC_DIST_COSINE = client_constants.CALC_DIST_COSINE
CALC_DIST_SQRT = "sqrt"
CALC_DIST_DIM = "dim"

OFFSET = client_constants.OFFSET
MILVUS_LIMIT = client_constants.MILVUS_LIMIT
BATCH_SIZE = client_constants.BATCH_SIZE
ID = "id"
TYPE = "type"
METRIC_TYPE = client_constants.METRIC_TYPE
PARAMS = client_constants.PARAMS
DISTANCE = "distance"
RADIUS = client_constants.RADIUS
RANGE_FILTER = client_constants.RANGE_FILTER
FIELDS = client_constants.FIELDS
EF = client_constants.EF
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
MAX_FILTERED_IDS_COUNT_ITERATION = client_constants.MAX_FILTERED_IDS_COUNT_ITERATION
INT64_MAX = client_constants.INT64_MAX
MAX_BATCH_SIZE: int = client_constants.MAX_BATCH_SIZE
DEFAULT_SEARCH_EXTENSION_RATE: int = client_constants.DEFAULT_SEARCH_EXTENSION_RATE
UNLIMITED: int = client_constants.UNLIMITED
MAX_TRY_TIME: int = client_constants.MAX_TRY_TIME
GUARANTEE_TIMESTAMP = client_constants.GUARANTEE_TIMESTAMP
ITERATOR_SESSION_CP_FILE = client_constants.ITERATOR_SESSION_CP_FILE
BM25_k1 = "bm25_k1"
BM25_b = "bm25_b"
