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


class Hit(object):
    def __init__(self):
        # TODO: Construct a Hit object from response. A hit represent a record corresponding to the query
        pass

    @property
    def id(self) -> int:
        # TODO: Return the id of the hit record
        return 0

    @property
    def distance(self) -> float:
        # TODO: Return the distance between the hit record and the query
        return 0.0

    @property
    def score(self) -> float:
        # TODO: Return the calculated score of the hit record, now the score is equal to distance
        return 0.0


class Hits(object):
    def __init__(self):
        # TODO: Construct a Hits object from response
        pass

    def __iter__(self):
        # TODO: Iterate the Hits object. Every iteration returns a Hit which
        #  represent a record corresponding to the query
        pass

    def __getitem__(self, item) -> Hit:
        # TODO: Return the kth Hit corresponding to the query
        pass

    def __len__(self) -> int:
        # TODO: Return the number of hit record
        pass

    @property
    def ids(self) -> list:
        # TODO: Return the ids of all hit record
        return []

    @property
    def distances(self) -> list:
        # TODO: Return the distances of all hit record
        return []


class SearchResult(object):
    def __init__(self):
        # TODO: construct a search result from response
        pass

    def __iter__(self):
        # TODO: Iterate the Search Result. Every iteration returns a Hits coresponding to a query
        pass

    def __getitem__(self, item) -> Hits:
        # TODO: Return the Hits corresponding to the nth query
        pass

    def __len__(self) -> len:
        # TODO: Return the number of query of Search Result
        pass

    def done(self):
        # TODO: Blocking util search done
        pass
