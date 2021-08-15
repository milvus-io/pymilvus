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

import abc
from pymilvus.client.abstract import Entity


class _IterableWrapper:
    def __init__(self, iterable_obj):
        self._iterable = iterable_obj

    def __iter__(self):
        return self

    def __next__(self):
        return self.on_result(self._iterable.__next__())

    def __getitem__(self, item):
        s = self._iterable.__getitem__(item)
        if isinstance(item, slice):
            _start = item.start or 0
            i_len = self._iterable.__len__()
            _end = min(item.stop, i_len) if item.stop else i_len

            elements = []
            for i in range(_start, _end):
                elements.append(self.on_result(s[i]))
            return elements
        return s

    def __len__(self):
        return self._iterable.__len__()

    @abc.abstractmethod
    def on_result(self, res):
        raise NotImplementedError


# TODO: how to add docstring to method of subclass and don't change the implementation?
#       for example like below:
# class Hits(_IterableWrapper):
#     __init__.__doc__ = """doc of __init__"""
#     __iter__.__doc__ = """doc of __iter__"""
#     __next__.__doc__ = """doc of __next__"""
#     __getitem__.__doc__ = """doc of __getitem__"""
#     __len__.__doc__ = """doc of __len__"""
#
#     def on_result(self, res):
#         return Hit(res)


class DocstringMeta(type):
    def __new__(cls, name, bases, attrs):
        doc_meta = attrs.pop("docstring", None)
        new_cls = super(DocstringMeta, cls).__new__(cls, name, bases, attrs)
        if doc_meta:
            for member_name, member in attrs.items():
                if member_name in doc_meta:
                    member.__doc__ = doc_meta[member_name]
        return new_cls


# for example:
# class Hits(_IterableWrapper, metaclass=DocstringMeta):
#     docstring = {
#         "__init__": """doc of __init__""",
#         "__iter__": """doc of __iter__""",
#         "__next__": """doc of __next__""",
#         "__getitem__": """doc of __getitem__""",
#         "__len__": """doc of __len__""",
#     }
#
#     def on_result(self, res):
#         return Hit(res)


class Hit:
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
    def entity(self) -> Entity:
        """
        Return the Entity of the hit record.

        :return pymilvus Entity object:
            The entity content of the hit record.
        """
        return self._hit.entity

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

    def __str__(self):
        """
        Return the information of hit record.

        :return str:
            The information of hit record.
        """
        return "(distance: {}, id: {})".format(self._hit.distance, self._hit.id)

    __repr__ = __str__


class Hits:
    def __init__(self, hits):
        """
        Construct a Hits object from response.
        """
        self._hits = hits

    def __iter__(self):
        """
        Iterate the Hits object. Every iteration returns a Hit which represent a record
        corresponding to the query.
        """
        return self

    def __next__(self):
        """
        Iterate the Hits object. Every iteration returns a Hit which represent a record
        corresponding to the query.
        """
        return Hit(self._hits.__next__())

    def __getitem__(self, item):
        """
        Return the kth Hit corresponding to the query.

        :return Hit:
            The kth specified by item Hit corresponding to the query.
        """
        s = self._hits.__getitem__(item)
        if isinstance(item, slice):
            _start = item.start or 0
            i_len = self._hits.__len__()
            _end = min(item.stop, i_len) if item.stop else i_len

            elements = []
            for i in range(_start, _end):
                elements.append(self.on_result(s[i]))
            return elements
        return s

    def __len__(self) -> int:
        """
        Return the number of hit record.

        :return int:
            The number of hit record.
        """
        return self._hits.__len__()

    def on_result(self, res):
        return Hit(res)

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


class SearchResult:
    def __init__(self, query_result=None):
        """
        Construct a search result from response.
        """
        self._qs = query_result

    def __iter__(self):
        """
        Iterate the Search Result. Every iteration returns a Hits corresponding to a query.
        """
        return self

    def __next__(self):
        """
        Iterate the Search Result. Every iteration returns a Hits corresponding to a query.
        """
        return self.on_result(self._qs.__next__())

    def __getitem__(self, item):
        """
        Return the Hits corresponding to the nth query.

        :return Hits:
            The hits corresponding to the nth(item) query.
        """
        s = self._qs.__getitem__(item)
        if isinstance(item, slice):
            _start = item.start or 0
            i_len = self._qs.__len__()
            _end = min(item.stop, i_len) if item.stop else i_len

            elements = []
            for i in range(_start, _end):
                elements.append(self.on_result(s[i]))
            return elements
        return s

    def __len__(self) -> int:
        """
        Return the number of query of Search Result.

        :return int:
            The number of query of search result.
        """
        return self._qs.__len__()

    def on_result(self, res):
        return Hits(res)
