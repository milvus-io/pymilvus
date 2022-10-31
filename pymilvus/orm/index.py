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

import copy

from ..exceptions import CollectionNotExistException, ExceptionsMessage
from ..client.configs import DefaultConfigs


class Index:
    def __init__(self, collection, field_name, index_params, **kwargs):
        """
        Creates index on a specified field according to the index parameters.

        :param collection: The collection in which the index is created
        :type  collection: Collection

        :param field_name: The name of the field to create an index for.
        :type  field_name: str

        :param index_params: Indexing parameters.
        :type  index_params: dict

        :param kwargs:
            * *index_name* (``str``) --
              The name of index which will be created. Then you can use the index name to check the state of index.
              If no index name is specified, default index name is used.

        :raises ParamError: If parameters are invalid.

        :example:
        >>> from pymilvus import *
        >>> from pymilvus.schema import *
        >>> from pymilvus.types import DataType
        >>> connections.connect()
        <pymilvus.client.stub.Milvus object at 0x7fac15e53470>
        >>> field1 = FieldSchema("int64", DataType.INT64, is_primary=True)
        >>> field2 = FieldSchema("fvec", DataType.FLOAT_VECTOR, is_primary=False, dim=128)
        >>> schema = CollectionSchema(fields=[field1, field2], description="collection description")
        >>> collection = Collection(name='test_collection', schema=schema)
        >>> # insert some data
        >>> index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        >>> index = Index(collection, "fvec", index_params)
        >>> print(index.params)
        {'index_type': 'IVF_FLAT', 'metric_type': 'L2', 'params': {'nlist': 128}}
        >>> print(index.collection_name)
        test_collection
        >>> print(index.field_name)
        fvec
        >>> index.drop()
        """
        from .collection import Collection
        if not isinstance(collection, Collection):
            raise CollectionNotExistException(message=ExceptionsMessage.CollectionType)
        self._collection = collection
        self._field_name = field_name
        self._index_params = index_params
        index_name = kwargs.get("index_name", DefaultConfigs.IndexName)
        self._index_name = index_name
        self._kwargs = kwargs
        if self._kwargs.pop("construct_only", False):
            return

        conn = self._get_connection()
        conn.create_index(self._collection.name, self._field_name, self._index_params, **kwargs)
        indexes = conn.list_indexes(self._collection.name)
        for index in indexes:
            if index.field_name == self._field_name:
                self._index_name = index.index_name
                break

    def _get_connection(self):
        return self._collection._get_connection()

    # read-only
    @property
    def params(self) -> dict:
        """
        Returns the index parameters.

        :return dict:
            The index parameters
        """
        return copy.deepcopy(self._index_params)

    # read-only
    @property
    def collection_name(self) -> str:
        """
        Returns the corresponding collection name.

        :return str:
            The corresponding collection name
        """
        return self._collection.name

    @property
    def field_name(self) -> str:
        """
        Returns the corresponding field name.

        :return str:
            The corresponding field name.
        """
        return self._field_name

    @property
    def index_name(self) -> str:
        """
        Returns the corresponding index name.

        :return str:
            The corresponding index name.
        """
        return self._index_name

    def __eq__(self, other) -> bool:
        """
        The order of the fields of index must be consistent.
        """
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """
        Put collection name, field name and index params into dict.
        """
        _dict = {
            "collection": self._collection._name,
            "field": self._field_name,
            "index_name": self._index_name,
            "index_param": self.params
        }
        return _dict

    def drop(self, timeout=None, **kwargs):
        """
        Drop an index and its corresponding index files.

        :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                        is set to None, client waits until server response or error occur
        :type  timeout: float

        :param kwargs:
            * *index_name* (``str``) --
              The name of index. If no index is specified, the default index name is used.

        """
        copy_kwargs = copy.deepcopy(kwargs)
        index_name = copy_kwargs.pop("index_name", DefaultConfigs.IndexName)
        conn = self._get_connection()
        conn.drop_index(self._collection.name, self.field_name, index_name, timeout=timeout, **copy_kwargs)
