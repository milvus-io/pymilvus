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

    @property
    def primary_keys(self):
        return self._mr.primary_keys

    @property
    def insert_count(self):
        return self._mr.insert_count

    @property
    def delete_count(self):
        return self._mr.delete_count

    @property
    def upsert_count(self):
        return self._mr.upsert_count

    @property
    def timestamp(self):
        return self._mr.timestamp

    def __str__(self):
        """
        Return the information of mutation result

        :return str:
            The information of mutation result.
        """
        return self._mr.__str__()

    __repr__ = __str__

    # TODO
    # def error_code(self):
    #     pass
    #
    # def error_reason(self):
    #     pass
