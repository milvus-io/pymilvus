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

# Re-export all iterator classes from client/iterator.py for backward compatibility
from pymilvus.client.iterator import (
    NO_CACHE_ID,
    IteratorCache,
    QueryIterator,
    SearchIterator,
    SearchPage,
    assert_info,
    check_set_flag,
    extend_batch_size,
    fall_back_to_latest_session_ts,
    io_operation,
    iterator_cache,
    metrics_positive_related,
)

__all__ = [
    "NO_CACHE_ID",
    "IteratorCache",
    "QueryIterator",
    "SearchIterator",
    "SearchPage",
    "assert_info",
    "check_set_flag",
    "extend_batch_size",
    "fall_back_to_latest_session_ts",
    "io_operation",
    "iterator_cache",
    "metrics_positive_related",
]
