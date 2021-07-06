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

class MutationResult:
    def __init__(self, mr):
        self._mr = mr
        self._primary_keys = list()
        self._insert_cnt = 0
        self._delete_cnt = 0
        self._upsert_cnt = 0
        self._timestamp = 0
        self._pack(mr)

    @property
    def primary_keys(self):
        return self._primary_keys

    @property
    def insert_count(self):
        return self._insert_cnt

    @property
    def delete_count(self):
        return self._delete_cnt

    @property
    def upsert_count(self):
        return self._upsert_cnt

    @property
    def timestamp(self):
        return self._timestamp

    def __str__(self):
        """
        Return the information of mutation result

        :return str:
            The information of mutation result.
        """
        return "(insert count: {}, delete count: {}, upsert count: {}, timestamp: {})".\
            format(self._insert_cnt, self._delete_cnt, self._upsert_cnt, self._timestamp)

    __repr__ = __str__

    # TODO
    # def error_code(self):
    #     pass
    #
    # def error_reason(self):
    #     pass

    def _pack(self, mr):
        if mr is None:
            return
        self._primary_keys = mr.primary_keys
        self._insert_cnt = mr.insert_count
        self._delete_cnt = mr.delete_count
        self._upsert_cnt = mr.upsert_count
        self._timestamp = mr.timestamp
