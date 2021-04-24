# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

class SearchResult(object):

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def done(self):
        pass


class Hits(object):
    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @property
    def distances(self):
        pass

    @property
    def ids(self):
        pass


class Hit(object):

    @property
    def distance(self):
        pass

    @property
    def id(self):
        pass

    @property
    def score(self):
        pass