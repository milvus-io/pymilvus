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

COMMON_TYPE_PARAMS = ("dim", "max_length")

CALC_DIST_IDS = "ids"
CALC_DIST_FLOAT_VEC = "float_vectors"
CALC_DIST_BIN_VEC = "bin_vectors"
CALC_DIST_METRIC = "metric"
CALC_DIST_L2 = "L2"
CALC_DIST_IP = "IP"
CALC_DIST_HAMMING = "HAMMING"
CALC_DIST_TANIMOTO = "TANIMOTO"
CALC_DIST_JACCARD = "JACCARD"
CALC_DIST_COSINE = "COSINE"
CALC_DIST_SQRT = "sqrt"
CALC_DIST_DIM = "dim"

OFFSET = "offset"
LIMIT = "limit"
ID = "id"
METRIC_TYPE = "metric_type"
PARAMS = "params"
DISTANCE = "distance"
RADIUS = "radius"
RANGE_FILTER = "range_filter"
FIELDS = "fields"
ITERATION_EXTENSION_REDUCE_RATE = "iteration_extension_reduce_rate"
DEFAULT_MAX_L2_DISTANCE = 99999999.0
DEFAULT_MIN_IP_DISTANCE = -99999999.0
DEFAULT_MAX_HAMMING_DISTANCE = 99999999.0
DEFAULT_MAX_TANIMOTO_DISTANCE = 99999999.0
DEFAULT_MAX_JACCARD_DISTANCE = 2.0
DEFAULT_MIN_COSINE_DISTANCE = -2.0
MAX_FILTERED_IDS_COUNT_ITERATION = 100000
