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

from typing import List, Union, Any, Dict

# TODO
def load_collection(collection_name: str, num_replica: int=1, timeout: float=None, **kwargs):
    pass

# TODO
def load_partitions(collection_name: str, partition_names: List[str], timeout: float=None, **kwargs):
    pass

# TODO
def release_collection(collection_name: str, timeout: float=None, **kwargs):
    pass

# TODO
def release_partitions(collection_name: str, partition_names: List[str], timeout: float=None, **kwargs):
    pass

# TODO
def flush(collection_name: str, timeout: float=None, **kwargs):
    pass

# TODO
def create_index(collection_name: str, field_name: str, index_name: str, timeout: float=None):
    pass

# TODO
def insert(collection_name: str, column_based_entities: List[list], partition_name: str="_default", timeout: float=None, **kwargs):
    pass

# TODO
def insert_by_rows(collection_name: str, row_based_entities: List[list], partition_name: str="_default", timeout: float=None, **kwargs):
    pass

# TODO
def bulk_insert(collection_name: str, partition_name: str, files: List[str], timeout: float=None, **kwargs) -> int:
    pass

# TODO
def upsert(timeout: float=None):
    pass

# TODO
def delete(collection_name: str, pks: List[Union[int, str]], partition_name: str="_default", timeout: float=None, **kwargs):
    pass

# TODO
def delete_by_expression(collection_name: str, expression: str, partition_name: str="_default", timeout: float=None, **kwargs):
    pass

# TODO
def search(collection_name: str, data: List[Any], anns_field: str, param: Dict[str, Any], limit: int,
    expression: str=None, partition_names: List[str]=None, output_fields: List[str]=None,
    round_decimal: int=-1, timeout: float=None, schema: Any=None, **kwargs):
    pass

# TODO
def query(collection_name: str, expr: str, output_fields: List[str]=None, partition_names: List[str]=None, timeout: float=None, **kwargs):
    pass
