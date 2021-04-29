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

from .connections import get_connection


def loading_progress(collection_name, partition_names=[], using="default"):
    """
    Show #loaded entities vs #total entities.

    :param collection_name: The name of collection to show
    :type  collection_name: str

    :param partition_names: The name of partition to show
    :type  partition_names: str list

    :return Loading progress:
        Loading progress contains num of loaded entities and num of total entities
    :rtype: dict {'num_loaded_entities':loaded_segments_nums, 'num_total_entities': total_segments_nums}

    :raises CollectionNotExistException: If collection doesn't exist.
    :raises PartitionNotExistException: If partition doesn't exist.

    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_parimary=False)
    >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> import pandas as pd
    >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))
    >>> data = pd.DataFrame(data={"int64" : int64_series})
    >>> collection.insert(data)
    >>> collection.load() # load collection to memory
    >>> utility.loading_progress("test_collection")
    """
    if len(partition_names) == 0:
        return get_connection(using).load_collection_progress(collection_name)
    else:
        return get_connection(using).load_partitions_progress(collection_name, partition_names)


def wait_for_loading_complete(collection_name, partition_names=[], timeout=None, using="default"):
    """
    Block until loading is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param partition_names: The name of partition to wait
    :type  partition_names: str list

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int

    :raises CollectionNotExistException: If collection doesn't exist.
    :raises PartitionNotExistException: If partition doesn't exist.

    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_parimary=False)
    >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> import pandas as pd
    >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))
    >>> data = pd.DataFrame(data={"int64" : int64_series})
    >>> collection.insert(data)
    >>> collection.load() # load collection to memory
    >>> utility.wait_for_loading_complete("test_collection")
    """
    if len(partition_names) == 0:
        return get_connection(using).wait_for_loading_collection_complete(collection_name, timeout)
    else:
        return get_connection(using).wait_for_loading_partitions_complete(collection_name, partition_names, timeout)


def index_building_progress(collection_name, index_name="", using="default"):
    """
    Show # indexed entities vs. # total entities.

    :param collection_name: The name of collection to show
    :type  collection_name: str

    :param index_name: The name of index to show.
                       Default index_name is to be used if index_name is not specific.
    :type  index_name: str

    :rtype:  dict {'total_rows':total_rows,'indexed_rows':indexed_rows}

    :raises CollectionNotExistException: If collection doesn't exist.
    :raises IndexNotExistException: If index doesn't exist.
    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, descrition="float64", is_parimary=False, dim=_DIM)
    >>> schema = CollectionSchema(fields=[field], description="test")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> import random
    >>> import numpy as np
    >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range(5000)]
    >>> collection.insert([vectors])
    >>> collection.load() # load collection to memory
    >>> index_param = {
    >>>     "metric_type": "L2",
    >>>     "index_type": "IVF_FLAT",
    >>>     "params": {"nlist": 1024}
    >>> }
    >>> collection.create_index("float_vector", index_param)
    >>> utility.wait_for_index_building_complete("test_collection", "")
    >>> utility.loading_progress("test_collection")
    """
    return get_connection(using).get_index_build_progress(collection_name, index_name)


def wait_for_index_building_complete(collection_name, index_name="", timeout=None, using="default"):
    """
    Block until building is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait
    :type  collection_name: str

    :param index_name: The name of index to wait
    :type  index_name: str

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int

    :raises CollectionNotExistException: If collection doesn't exist.
    :raises IndexNotExistException: If index doesn't exist.

    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="float_vector", dtype=DataType.FLOAT_VECTOR, descrition="float64", is_parimary=False, dim=_DIM)
    >>> schema = CollectionSchema(fields=[field], description="test")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> import random
    >>> import numpy as np
    >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range(5000)]
    >>> collection.insert([vectors])
    >>> collection.load() # load collection to memory
    >>> index_param = {
    >>>     "metric_type": "L2",
    >>>     "index_type": "IVF_FLAT",
    >>>     "params": {"nlist": 1024}
    >>> }
    >>> collection.create_index("float_vector", index_param)
    >>> utility.wait_for_index_building_complete("test_collection", "")
    >>> utility.loading_progress("test_collection")
    """
    return get_connection(using).wait_for_creating_index(collection_name, index_name, timeout)


def has_collection(collection_name, using="default"):
    """
    Checks whether a specified collection exists.

    :param collection_name: The name of collection to check.
    :type  collection_name: str

    :return: Whether the collection exists.
    :rtype:  bool

    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_parimary=False)
    >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> utility.has_collection("test_collection")
    """
    return get_connection(using).has_collection(collection_name)


def has_partition(collection_name, partition_name, using="default"):
    """
    Checks if a specified partition exists in a collection.

    :param collection_name: The collection name of partition to check
    :type  collection_name: str

    :param partition_name: The name of partition to check.
    :type  partition_name: str

    :return: Whether the partition exist.
    :rtype:  bool

    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_parimary=False)
    >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> utility.has_partition("_default")
    """
    return get_connection(using).has_partition(collection_name, partition_name)


def list_collections(timeout=None, using="default") -> list:
    """
    Returns a list of all collection names.

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur.
    :type  timeout: float

    :return: List of collection names, return when operation is successful
    :rtype: list[str]

    :example:
    >>> from pymilvus_orm.collection import Collection
    >>> from pymilvus_orm.schema import FieldSchema, CollectionSchema
    >>> from pymilvus_orm.types import DataType
    >>> from pymilvus_orm import connections
    >>> from pymilvus_orm import utility
    >>> connections.create_connection(alias="default")
    >>> field = FieldSchema(name="int64", dtype=DataType.INT64, descrition="int64", is_parimary=False)
    >>> schema = CollectionSchema(fields=[field], description="get collection entities num")
    >>> collection = Collection(name="test_collection", schema=schema)
    >>> utility.list_collections()
    """
    return get_connection(using).list_collections()
