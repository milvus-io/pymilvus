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

# Re-export all schema classes from client/schema.py for backward compatibility
from pymilvus.client.schema import (
    CollectionSchema,
    FieldSchema,
    Function,
    FunctionScore,
    Highlighter,
    LexicalHighlighter,
    StructFieldSchema,
    _check_data_schema_cnt,
    _check_insert_data,
    check_insert_schema,
    check_is_row_based,
    check_schema,
    check_upsert_schema,
    construct_fields_from_dataframe,
    infer_default_value_bydata,
    is_row_based,
    is_valid_insert_data,
    isVectorDataType,
    prepare_fields_from_dataframe,
    validate_clustering_key,
    validate_partition_key,
    validate_primary_key,
)

# Also re-export FunctionType from types for backward compatibility
from pymilvus.client.types import FunctionType

__all__ = [
    "CollectionSchema",
    "FieldSchema",
    "Function",
    "FunctionScore",
    "FunctionType",
    "Highlighter",
    "LexicalHighlighter",
    "StructFieldSchema",
    "_check_data_schema_cnt",
    "_check_insert_data",
    "check_insert_schema",
    "check_is_row_based",
    "check_schema",
    "check_upsert_schema",
    "construct_fields_from_dataframe",
    "infer_default_value_bydata",
    "isVectorDataType",
    "is_row_based",
    "is_valid_insert_data",
    "prepare_fields_from_dataframe",
    "validate_clustering_key",
    "validate_partition_key",
    "validate_primary_key",
]
