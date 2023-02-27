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

from typing import List, Any
from .exceptions import ParamError


# TODO
def generate_address(host, port) -> str:
    return f"{host}:{port}"


def str_checker(var: Any) -> bool:
    return isinstance(var, str)

def list_str_checker(var: Any) -> bool:
    if var and isinstance(var, list):
        return all(isinstance(v, str) for v in var)

    return False

def int_checker(var: Any) -> bool:
    return isinstance(var, int)


CHECKERS = {
    "collection_name":  str_checker,
    "partition_name":   str_checker,
    "index_name":       str_checker,
    "field_name":       str_checker,
    "alias":            str_checker,
    "name":             str_checker,
    "round_decimal":    int_checker,
    "num_replica":      int_checker,
    "dim":              int_checker,
    "partition_names":  list_str_checker,
    "output_fields":    list_str_checker,
    # TODO
    "anns_field":       str_checker,
    "limit":            int_checker,
    "topk":             int_checker,
}

class TypeChecker:
    @classmethod
    def check(cls, **kwargs):
        for k, var in kwargs.items():
            checker = CHECKERS.get(k)
            if checker is None:
                raise ParamError(message=f"No checker for parameter {k}")

            if not checker(var):
                raise ParamError(message=f"Invalid type {type(var)} for {k}")

    @property
    def list_checkers(self) -> List[str]:
        return CHECKERS.keys()
