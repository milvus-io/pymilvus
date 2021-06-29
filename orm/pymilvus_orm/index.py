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

from .exceptions import CollectionNotExistException, ExceptionsMessage, IndexNotExistException


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

        :raises ParamError: If parameters are invalid.
        :raises IndexConflictException:
        If an index of the same name but of different param already exists.

        :example:
        >>> from pymilvus_orm import *
        >>> from pymilvus_orm.schema import *
        >>> from pymilvus_orm.types import DataType
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
            raise CollectionNotExistException(0, ExceptionsMessage.CollectionType)
        self._collection = collection
        self._field_name = field_name
        self._index_params = index_params
        self._kwargs = kwargs
        if self._kwargs.pop("construct_only", False):
            return

        conn = self._get_connection()
        index = conn.describe_index(self._collection.name)
        if index is not None:
            tmp_field_name = index.pop("field_name", None)
        if index is None or index != index_params or tmp_field_name != field_name:
            conn.create_index(self._collection.name, self._field_name, self._index_params)

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
            "index_param": self.params
        }
        return _dict

    def drop(self, **kwargs):
        """
        Drop an index and its corresponding index files.

        :raises IndexNotExistException: If the specified index does not exist.
        """
        conn = self._get_connection()
        if conn.describe_index(self._collection.name) is None:
            raise IndexNotExistException(0, ExceptionsMessage.IndexNotExist)
        conn.drop_index(self._collection.name, self.field_name, **kwargs)
