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


from milvus.client.abstract import LoopBase


class _IterableBase(LoopBase):
    # hide this in doc
    def get__item(self, item):
        pass


class Hit(object):
    def __init__(self, hit):
        """
        Construct a Hit object from response. A hit represent a record corresponding to the query.
        """
        self._hit = hit

    @property
    def id(self) -> int:
        """
        Return the id of the hit record.

        :return int:
            The id of the hit record.
        """
        return self._hit.id

    @property
    def distance(self) -> float:
        """
        Return the distance between the hit record and the query.

        :return float:
            The distance of the hit record.
        """
        return self._hit.distance

    @property
    def score(self) -> float:
        """
        Return the calculated score of the hit record, now the score is equal to distance.

        :return float:
            The score of the hit record.
        """
        return self._hit.score


class Hits(_IterableBase):
    def __init__(self, hits):
        """
        Construct a Hits object from response.
        """
        super(Hits, self).__init__()
        self._hits = hits

    def __iter__(self):
        """
        Iterate the Hits object. Every iteration returns a Hit which represent a record
        corresponding to the query.
        """
        return super(Hits, self).__iter__()

    def __next__(self):
        """
        Iterate the Hits object. Every iteration returns a Hit which represent a record
        corresponding to the query.
        """
        return super(Hits, self).__next__()

    def __getitem__(self, item) -> Hit:
        """
        Return the kth Hit corresponding to the query.

        :return Hit:
            The kth specified by item Hit corresponding to the query.
        """
        return Hit(self._hits[item])

    def __len__(self) -> int:
        """
        Return the number of hit record.

        :return int:
            The number of hit record.
        """
        return len(self._hits)

    @property
    def ids(self) -> list:
        """
        Return the ids of all hit record.

        :return list[int]:
            The ids of all hit record.
        """
        return self._hits.ids

    @property
    def distances(self) -> list:
        """
        Return the distances of all hit record.

        :return list[float]:
            The distances of all hit record.
        """
        return self._hits.distances


class SearchResult(_IterableBase):
    def __init__(self, query_result=None):
        """
        Construct a search result from response.
        """
        super(SearchResult, self).__init__()
        self._qs = query_result

    def __iter__(self):
        """
        Iterate the Search Result. Every iteration returns a Hits corresponding to a query.
        """
        return super(SearchResult, self).__iter__()

    def __next__(self):
        """
        Iterate the Search Result. Every iteration returns a Hits corresponding to a query.
        """
        return super(SearchResult, self).__next__()

    def __getitem__(self, item) -> Hits:
        """
        Return the Hits corresponding to the nth query.

        :return Hits:
            The hits corresponding to the nth(item) query.
        """
        return Hits(self._qs[item])

    def __len__(self) -> int:
        """
        Return the number of query of Search Result.

        :return int:
            The number of query of search result.
        """
        return len(self._qs)


class SearchResultFuture(object):
    def __init__(self, future):
        self._f = future

    def result(self, **kwargs):
        """
        Return the SearchResult from future object.

        It's a synchronous interface. It will wait executing until
        server respond or timeout occur(if specified).
        """
        return SearchResult(self._f.result())

    def cancel(self):
        """
        Cancel the search request.
        """
        return self._f.cancel()

    def done(self):
        """
        Wait for search request done.
        """
        return self._f.done()

