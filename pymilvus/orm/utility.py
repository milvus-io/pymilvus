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

from . import constants
from .connections import get_connection
from .exceptions import (
    ConnectionNotExistException,
    ExceptionsMessage,
    ResultError,
)


def _get_connection(alias):
    conn = get_connection(alias)
    if conn is None:
        raise ConnectionNotExistException(0, ExceptionsMessage.ConnectFirst)
    return conn


def loading_progress(collection_name, partition_names=None, using="default"):
    """
    Show #loaded entities vs #total entities.

    :param collection_name: The name of collection is loading
    :type  collection_name: str

    :param partition_names: The names of partitions is loading
    :type  partition_names: str list

    :return dict:
        Loading progress is a dict contains num of loaded entities and num of total entities.
        {'num_loaded_entities':loaded_segments_nums, 'num_total_entities': total_segments_nums}
    :raises PartitionNotExistException: If partition doesn't exist.
    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="get collection entities num")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))i
        >>> float_vector_series = [[random.random() for _ in range _DIM] for _ in range (10)]
        >>> data = pd.DataFrame({"int64" : int64_series, "float_vector": float_vector_series})
        >>> collection.insert(data)
        >>> collection.load() # load collection to memory
        >>> utility.loading_progress("test_collection")
    """
    if not partition_names or len(partition_names) == 0:
        return _get_connection(using).load_collection_progress(collection_name)
    return _get_connection(using).load_partitions_progress(collection_name, partition_names)


def wait_for_loading_complete(collection_name, partition_names=None, timeout=None, using="default"):
    """
    Block until loading is done or Raise Exception after timeout.

    :param collection_name: The name of collection to wait for loading complete
    :type  collection_name: str

    :param partition_names: The names of partitions to wait for loading complete
    :type  partition_names: str list

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int

    :raises CollectionNotExistException: If collection doesn't exist.
    :raises PartitionNotExistException: If partition doesn't exist.

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_float_vector], description="get collection entities num")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))i
        >>> float_vector_series = [[random.random() for _ in range _DIM] for _ in range (10)]
        >>> data = pd.DataFrame({"int64" : int64_series, "float_vector": float_vector_series})
        >>> collection.insert(data)
        >>> collection.load() # load collection to memory
        >>> utility.wait_for_loading_complete("test_collection")
    """
    if not partition_names or len(partition_names) == 0:
        return _get_connection(using).wait_for_loading_collection_complete(collection_name, timeout)
    return _get_connection(using).wait_for_loading_partitions_complete(collection_name,
                                                                       partition_names,
                                                                       timeout)


def index_building_progress(collection_name, index_name="", using="default"):
    """
    Show # indexed entities vs. # total entities.

    :param collection_name: The name of collection is building index
    :type  collection_name: str

    :param index_name: The name of index is building.
                        Default index_name is to be used if index_name is not specific.
    :type index_name: str

    :return dict:

        Index building progress is a dict contains num of indexed entities and num of total
        entities.
        {'total_rows':total_rows,'indexed_rows':indexed_rows}

    :raises CollectionNotExistException: If collection doesn't exist.
    :raises IndexNotExistException: If index doesn't exist.
    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_float_vector], description="test")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import random
        >>> import numpy as np
        >>> import pandas as pd
        >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range(5000)]
        >>> int64_series = pd.Series(data=list(range(5000, 10000)), index=list(range(5000)))
        >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range (5000)]
        >>> data = pd.DataFrame({"int64" : int64_series, "float_vector": vectors})
        >>> collection.insert(data)
        >>> collection.load() # load collection to memory
        >>> index_param = {
        >>>    "metric_type": "L2",
        >>>    "index_type": "IVF_FLAT",
        >>>    "params": {"nlist": 1024}
        >>> }
        >>> collection.create_index("float_vector", index_param)
        >>> utility.index_building_progress("test_collection", "")
        >>> utility.loading_progress("test_collection")
    """
    return _get_connection(using).get_index_build_progress(collection_name, index_name)


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
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_float_vector], description="test")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import random
        >>> import numpy as np
        >>> import pandas as pd
        >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range(5000)]
        >>> int64_series = pd.Series(data=list(range(5000, 10000)), index=list(range(5000)))
        >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range (5000)]
        >>> data = pd.DataFrame({"int64" : int64_series, "float_vector": vectors})
        >>> collection.insert(data)
        >>> collection.load() # load collection to memory
        >>> index_param = {
        >>>    "metric_type": "L2",
        >>>    "index_type": "IVF_FLAT",
        >>>    "params": {"nlist": 1024}
        >>> }
        >>> collection.create_index("float_vector", index_param)
        >>> utility.index_building_progress("test_collection", "")
        >>> utility.loading_progress("test_collection")

    """
    return _get_connection(using).wait_for_creating_index(collection_name, index_name, timeout)[0]


def has_collection(collection_name, using="default"):
    """
    Checks whether a specified collection exists.

    :param collection_name: The name of collection to check.
    :type  collection_name: str

    :return bool:
        Whether the collection exists.

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_float_vector], description="test")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> utility.has_collection("test_collection")
    """
    return _get_connection(using).has_collection(collection_name)


def has_partition(collection_name, partition_name, using="default"):
    """
    Checks if a specified partition exists in a collection.

    :param collection_name: The collection name of partition to check
    :type  collection_name: str

    :param partition_name: The name of partition to check.
    :type  partition_name: str

    :return bool:
        Whether the partition exist.

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_float_vector], description="test")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> utility.has_partition("_default")
    """
    return _get_connection(using).has_partition(collection_name, partition_name)


def drop_collection(collection_name, timeout=None, using="default"):
    """
    Drop a collection by name

    :param collection_name: A string representing the collection to be deleted
    :type  collection_name: str
    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur.
    :type  timeout: float

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> schema = CollectionSchema(fields=[
        ...     FieldSchema("int64", DataType.INT64, description="int64", is_primary=True),
        ...     FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=128),
        ... ])
        >>> collection = Collection(name="drop_collection_test", schema=schema)
        >>> utility.has_collection("drop_collection_test")
        >>> True
        >>> utility.drop_collection("drop_collection_test")
        >>> utility.has_collection("drop_collection_test")
        >>> False
    """
    return _get_connection(using).drop_collection(collection_name, timeout)


def list_collections(timeout=None, using="default") -> list:
    """
    Returns a list of all collection names.

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur.
    :type  timeout: float

    :return list[str]:
        List of collection names, return when operation is successful

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_float_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, description="float_vector", is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_float_vector], description="test")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> utility.list_collections()
    """
    return _get_connection(using).list_collections()


def calc_distance(vectors_left, vectors_right, params=None, timeout=None, using="default"):
    """
    Calculate distance between two vector arrays.

    :param vectors_left: The vectors on the left of operator.
    :type  vectors_left: dict
    `{"ids": [1, 2, 3, .... n], "collection": "c_1", "partition": "p_1", "field": "v_1"}`
    or
    `{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]}`
    or
    `{"bin_vectors": [b'\x94', b'N', ... b'\xca']}`

    :param vectors_right: The vectors on the right of operator.
    :type  vectors_right: dict
    `{"ids": [1, 2, 3, .... n], "collection": "col_1", "partition": "p_1", "field": "v_1"}`
    or
    `{"float_vectors": [[1.0, 2.0], [3.0, 4.0], ... [9.0, 10.0]]}`
    or
    `{"bin_vectors": [b'\x94', b'N', ... b'\xca']}`

    :param params: key-value pair parameters
                       Key: "metric_type"/"metric"    Value: "L2"/"IP"/"HAMMING"/"TANIMOTO", default is "L2",
                       Key: "sqrt"    Value: true or false, default is false    Only for "L2" distance
                       Key: "dim"     Value: set this value if dimension is not a multiple of 8,
                                             otherwise the dimension will be calculted by list length,
                                             only for "HAMMING" and "TANIMOTO"
        :type  params: dict
            Examples of supported metric_type:
                `{"metric_type": "L2", "sqrt": true}`
                `{"metric_type": "IP"}`
                `{"metric_type": "HAMMING", "dim": 17}`
                `{"metric_type": "TANIMOTO"}`
            Note: metric type are case insensitive

    :return: 2-d array distances
    :rtype: list[list[int]] for "HAMMING" or list[list[float]] for others
        Assume the vectors_left: L_1, L_2, L_3
        Assume the vectors_right: R_a, R_b
        Distance between L_n and R_m we called "D_n_m"
        The returned distances are arranged like this:
          [[D_1_a, D_1_b],
           [D_2_a, D_2_b],
           [D_3_a, D_3_b]]

        Note: if some vectors doesn't exist in collection, the returned distance is "-1.0"

    :example:
        >>> vectors_l = [[random.random() for _ in range(64)] for _ in range(5)]
        >>> vectors_r = [[random.random() for _ in range(64)] for _ in range(10)]
        >>> op_l = {"float_vectors": vectors_l}
        >>> op_r = {"float_vectors": vectors_r}
        >>> params = {"metric": "L2", "sqrt": True}
        >>> results = utility.calc_distance(vectors_left=op_l, vectors_right=op_r, params=params)
    """
    res = _get_connection(using).calc_distance(vectors_left, vectors_right, params, timeout)

    def vector_count(op):
        x = 0
        if constants.CALC_DIST_IDS in op:
            x = len(op[constants.CALC_DIST_IDS])
        elif constants.CALC_DIST_FLOAT_VEC in op:
            x = len(op[constants.CALC_DIST_FLOAT_VEC])
        elif constants.CALC_DIST_BIN_VEC in op:
            x = len(op[constants.CALC_DIST_BIN_VEC])
        return x

    n = vector_count(vectors_left)
    m = vector_count(vectors_right)
    if len(res) != n * m:
        raise ResultError("Returned distance array is illegal")

    # transform 1-d distances to 2-d distances
    res_2_d = []
    for i in range(n):
        res_2_d.append(res[i * m:i * m + m])
    return res_2_d
