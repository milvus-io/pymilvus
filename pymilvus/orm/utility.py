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

from .connections import connections

from ..client.utils import mkts_from_hybridts as _mkts_from_hybridts
from ..client.utils import mkts_from_unixtime as _mkts_from_unixtime
from ..client.utils import mkts_from_datetime as _mkts_from_datetime
from ..client.utils import hybridts_to_unixtime as _hybridts_to_unixtime
from ..client.types import BulkInsertState


def mkts_from_hybridts(hybridts, milliseconds=0., delta=None):
    """
    Generate a hybrid timestamp based on an existing hybrid timestamp, timedelta and incremental time internval.

    :param hybridts: The original hybrid timestamp used to generate a new hybrid timestamp.
                     Non-negative interger range from 0 to 18446744073709551615.
    :type  hybridts: int

    :param milliseconds: Incremental time interval. The unit of time is milliseconds.
    :type  milliseconds: float

    :param delta: A duration expressing the difference between two date, time, or datetime instances
                  to microsecond resolution.
    :type  delta: datetime.timedelta

    :return int:
        Hybrid timetamp is a non-negative interger range from 0 to 18446744073709551615.

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="get collection entities num")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))i
        >>> float_vector_series = [[random.random() for _ in range _DIM] for _ in range (10)]
        >>> data = pd.DataFrame({"int64" : int64_series, "float_vector": float_vector_series})
        >>> m = collection.insert(data)
        >>> ts_new = utility.mkts_from_hybridts(m.timestamp, milliseconds=1000.0)
    """
    return _mkts_from_hybridts(hybridts, milliseconds=milliseconds, delta=delta)


def mkts_from_unixtime(epoch, milliseconds=0., delta=None):
    """
    Generate a hybrid timestamp based on Unix Epoch time, timedelta and incremental time internval.

    :param epoch: The known Unix Epoch time used to generate a hybrid timestamp.
                  The Unix Epoch time is the number of seconds that have elapsed
                  since January 1, 1970 (midnight UTC/GMT).
    :type  epoch: float

    :param milliseconds: Incremental time interval. The unit of time is milliseconds.
    :type  milliseconds: float

    :param delta: A duration expressing the difference between two date, time, or datetime instances
                  to microsecond resolution.
    :type  delta: datetime.timedelta

    :return int:
        Hybrid timetamp is a non-negative interger range from 0 to 18446744073709551615.

    :example:
        >>> import time
        >>> from pymilvus import utility
        >>> epoch_t = time.time()
        >>> ts = utility.mkts_from_unixtime(epoch_t, milliseconds=1000.0)
    """
    return _mkts_from_unixtime(epoch, milliseconds=milliseconds, delta=delta)


def mkts_from_datetime(d_time, milliseconds=0., delta=None):
    """
    Generate a hybrid timestamp based on datetime, timedelta and incremental time internval.

    :param d_time: The known datetime used to generate a hybrid timestamp.
    :type  d_time: datetime.datetime.

    :param milliseconds: Incremental time interval. The unit of time is milliseconds.
    :type  milliseconds: float

    :param delta: A duration expressing the difference between two date, time, or datetime instances
                  to microsecond resolution.
    :type  delta:  datetime.timedelta

    :return int:
        Hybrid timetamp is a non-negative interger range from 0 to 18446744073709551615.

    :example:
        >>> import datetime
        >>> from pymilvus import utility
        >>> d = datetime.datetime.now()
        >>> ts = utility.mkts_from_datetime(d, milliseconds=1000.0)
    """
    return _mkts_from_datetime(d_time, milliseconds=milliseconds, delta=delta)


def hybridts_to_datetime(hybridts, tz=None):
    """
    Convert a hybrid timestamp to the datetime according to timezone.

    :param hybridts: The known hybrid timestamp to convert to datetime.
                     Non-negative interger range from 0 to 18446744073709551615.
    :type  hybridts: int
    :param tz: Timezone defined by a fixed offset from UTC. If argument tz is None or not specified,
               the hybridts is converted to the platform’s local date and time.
    :type  tz: datetime.timezone

    :return datetime:
        The datetime object.

    :raises Exception: If parameter tz is not of type datetime.timezone.

    :example:
        >>> import time
        >>> from pymilvus import utility
        >>> epoch_t = time.time()
        >>> ts = utility.mkts_from_unixtime(epoch_t)
        >>> d = utility.hybridts_to_datetime(ts)
    """
    import datetime
    if tz is not None and not isinstance(tz, datetime.timezone):
        raise Exception("parameter tz should be type of datetime.timezone")
    epoch = _hybridts_to_unixtime(hybridts)
    return datetime.datetime.fromtimestamp(epoch, tz=tz)


def hybridts_to_unixtime(hybridts):
    """
    Convert a hybrid timestamp to UNIX Epoch time ignoring the logic part.

    :param hybridts: The known hybrid timestamp to convert to UNIX Epoch time.
                     Non-negative interger range from 0 to 18446744073709551615.
    :type  hybridts: int

    :return float:
        The Unix Epoch time is the number of seconds that have elapsed since January 1, 1970 (midnight UTC/GMT).

    :example:
        >>> import time
        >>> from pymilvus import utility
        >>> epoch1 = time.time()
        >>> ts = utility.mkts_from_unixtime(epoch1)
        >>> epoch2 = utility.hybridts_to_unixtime(ts)
        >>> assert epoch1 == epoch2
    """
    return _hybridts_to_unixtime(hybridts)


def _get_connection(alias):
    return connections._fetch_handler(alias)


def loading_progress(collection_name, partition_names=None, using="default"):
    """ Show loading progress of sealed segments in percentage.

    :param collection_name: The name of collection is loading
    :type  collection_name: str

    :param partition_names: The names of partitions is loading
    :type  partition_names: str list

    :return dict:
        {'loading_progress': '100%'}
    :raises PartitionNotExistException: If partition doesn't exist.
    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> import pandas as pd
        >>> import random
        >>> connections.connect()
        >>> fields = [
        ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
        ...     FieldSchema("films", DataType.FLOAT_VECTOR, dim=8),
        ... ]
        >>> schema = CollectionSchema(fields)
        >>> collection = Collection("test_loading_progress", schema)
        >>> data = pd.DataFrame({
        ...     "film_id" : pd.Series(data=list(range(10, 20)), index=list(range(10))),
        ...     "films": [[random.random() for _ in range(8)] for _ in range (10)],
        ... })
        >>> collection.insert(data)
        >>> collection.create_index("films", {"index_type": "IVF_FLAT", "params": {"nlist": 8}, "metric_type": "L2"})
        >>> collection.load(_async=True)
        >>> utility.loading_progress("test_loading_progress")
        {'loading_progress': '100%'}
    """
    progress = _get_connection(using).get_loading_progress(collection_name, partition_names)
    return {
        "loading_progress": f"{progress:.0f}%",
    }


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
        >>> field_fvec = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_fvec], description="get collection entities num")
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
        return _get_connection(using).wait_for_loading_collection(collection_name, timeout=timeout)
    return _get_connection(using).wait_for_loading_partitions(collection_name, partition_names, timeout=timeout)


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
        >>> connections.connect()
        >>>
        >>> fields = [
        ...     FieldSchema("int64", DataType.INT64, is_primary=True, auto_id=True),
        ...     FieldSchema("float_vector", DataType.FLOAT_VECTOR, dim=128),
        ... ]
        >>> schema = CollectionSchema(fields, description="test index_building_progress")
        >>> c = Collection(name="test_index_building_progress", schema=schema)

        >>> import random
        >>> vectors = [[random.random() for _ in range(_DIM)] for _ in range(5000)]
        >>> c.insert([vectors])
        >>> c.load()
        >>> index_params = {
        ...    "metric_type": "L2",
        ...    "index_type": "IVF_FLAT",
        ...    "params": {"nlist": 1024}
        ... }
        >>> index = c.create_index(field_name="float_vector", index_params=index_params, index_name="ivf_flat")
        >>> utility.index_building_progress("test_collection", c.name)
    """
    return _get_connection(using).get_index_build_progress(collection_name=collection_name, index_name=index_name)


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
        >>> field_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="test")
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
    return _get_connection(using).wait_for_creating_index(collection_name, index_name, timeout=timeout)[0]


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
        >>> field_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="test")
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
        >>> field_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="test")
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
    return _get_connection(using).drop_collection(collection_name, timeout=timeout)


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
        >>> field_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="test")
        >>> collection = Collection(name="test_collection", schema=schema)
        >>> utility.list_collections()
    """
    return _get_connection(using).list_collections(timeout=timeout)


def load_balance(collection_name: str, src_node_id, dst_node_ids=None, sealed_segment_ids=None, timeout=None, using="default"):
    """ Do load balancing operation from source query node to destination query node.

    :param collection_name: The collection to balance.
    :type  collection_name: str

    :param src_node_id: The source query node id to balance.
    :type  src_node_id: int

    :param dst_node_ids: The destination query node ids to balance.
    :type  dst_node_ids: list[int]

    :param sealed_segment_ids: Sealed segment ids to balance.
    :type  sealed_segment_ids: list[int]

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int

    :raises BaseException: If query nodes not exist.
    :raises BaseException: If sealed segments not exist.

    :example:
        >>> from pymilvus import connections, utility
        >>>
        >>> connections.connect()
        >>>
        >>> src_node_id = 0
        >>> dst_node_ids = [1]
        >>> sealed_segment_ids = []
        >>> res = utility.load_balance("test_collection", src_node_id, dst_node_ids, sealed_segment_ids)
    """
    if dst_node_ids is None:
        dst_node_ids = []
    if sealed_segment_ids is None:
        sealed_segment_ids = []
    return _get_connection(using).\
            load_balance(collection_name, src_node_id, dst_node_ids, sealed_segment_ids, timeout=timeout)


def get_query_segment_info(collection_name, timeout=None, using="default"):
    """
    Notifies Proxy to return segments information from query nodes.

    :param collection_name: A string representing the collection to get segments info.
    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur.
    :type  timeout: float

    :return: QuerySegmentInfo:
        QuerySegmentInfo is the growing segments's information in query cluster.
    :rtype: QuerySegmentInfo

    :example:
        >>> from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
        >>> connections.connect(alias="default")
        >>> _DIM = 128
        >>> field_int64 = FieldSchema("int64", DataType.INT64, description="int64", is_primary=True)
        >>> field_vector = FieldSchema("float_vector", DataType.FLOAT_VECTOR, is_primary=False, dim=_DIM)
        >>> schema = CollectionSchema(fields=[field_int64, field_vector], description="get collection entities num")
        >>> collection = Collection(name="test_get_segment_info", schema=schema)
        >>> import pandas as pd
        >>> int64_series = pd.Series(data=list(range(10, 20)), index=list(range(10)))i
        >>> float_vector_series = [[random.random() for _ in range _DIM] for _ in range (10)]
        >>> data = pd.DataFrame({"int64" : int64_series, "float_vector": float_vector_series})
        >>> collection.insert(data)
        >>> collection.load() # load collection to memory
        >>> res = utility.get_query_segment_info("test_get_segment_info")
    """
    return _get_connection(using).get_query_segment_info(collection_name, timeout=timeout)


def create_alias(collection_name: str, alias: str, timeout=None, using="default"):
    """ Specify alias for a collection.
    Alias cannot be duplicated, you can't assign the same alias to different collections.
    But you can specify multiple aliases for a collection, for example:
        before create_alias("collection_1", "bob"):
            aliases of collection_1 are ["tom"]
        after create_alias("collection_1", "bob"):
            aliases of collection_1 are ["tom", "bob"]

    :param alias: The alias of the collection.
    :type  alias: str.

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur
    :type  timeout: float

    :raises CollectionNotExistException: If the collection does not exist.
    :raises BaseException: If the alias failed to create.

    :example:
        >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        >>> connections.connect()
        >>> schema = CollectionSchema([
        ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
        ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
        ... ])
        >>> collection = Collection("test_collection_create_alias", schema)
        >>> utility.create_alias(collection.name, "alias")
        Status(code=0, message='')
    """
    return _get_connection(using).create_alias(collection_name, alias, timeout=timeout)


def drop_alias(alias: str, timeout=None, using="default"):
    """ Delete the alias.
    No need to provide collection name because an alias can only be assigned to one collection
    and the server knows which collection it belongs.
    For example:
        before drop_alias("bob"):
            aliases of collection_1 are ["tom", "bob"]
        after drop_alias("bob"):
            aliases of collection_1 are = ["tom"]

    :param alias: The alias to drop.
    :type  alias: str

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur
    :type  timeout: float

    :raises CollectionNotExistException: If the collection does not exist.
    :raises BaseException: If the alias doesn't exist.

    :example:
        >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        >>> connections.connect()
        >>> schema = CollectionSchema([
        ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
        ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
        ... ])
        >>> collection = Collection("test_collection_drop_alias", schema)
        >>> utility.create_alias(collection.name, "alias")
        >>> utility.drop_alias("alias")
        Status(code=0, message='')
    """
    return _get_connection(using).drop_alias(alias, timeout=timeout)


def alter_alias(collection_name: str, alias: str, timeout=None, using="default"):
    """ Change the alias of a collection to another collection.
    Raise error if the alias doesn't exist.
    Alias cannot be duplicated, you can't assign same alias to different collections.
    This api can change alias owner collection, for example:
        before alter_alias("collection_2", "bob"):
            collection_1's aliases = ["bob"]
            collection_2's aliases = []
        after alter_alias("collection_2", "bob"):
            collection_1's aliases = []
            collection_2's aliases = ["bob"]

    :param collection_name: The collection name to witch this alias is goting to alter.
    :type  collection_name: str.

    :param alias: The alias of the collection.
    :type  alias: str

    :param timeout: An optional duration of time in seconds to allow for the RPC. When timeout
                    is set to None, client waits until server response or error occur
    :type  timeout: float

    :raises CollectionNotExistException: If the collection does not exist.
    :raises BaseException: If the alias failed to alter.

    :example:
        >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        >>> connections.connect()
        >>> schema = CollectionSchema([
        ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
        ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2)
        ... ])
        >>> collection = Collection("test_collection_alter_alias", schema)
        >>> utility.alter_alias(collection.name, "alias")
        if the alias exists, return Status(code=0, message='')
        otherwise return Status(code=1, message='alias does not exist')
    """
    return _get_connection(using).alter_alias(collection_name, alias, timeout=timeout)


def list_aliases(collection_name: str, timeout=None, using="default"):
    """ Returns alias list of the collection.

    :return list of str:
        The collection aliases, returned when the operation succeeds.

    :example:
        >>> from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        >>> connections.connect()
        >>> fields = [
        ...     FieldSchema("film_id", DataType.INT64, is_primary=True),
        ...     FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=128)
        ... ]
        >>> schema = CollectionSchema(fields)
        >>> collection = Collection("test_collection_list_aliases", schema)
        >>> utility.create_alias(collection.name, "tom")
        >>> utility.list_aliases(collection.name)
        ['tom']
    """
    conn = _get_connection(using)
    resp = conn.describe_collection(collection_name, timeout=timeout)
    aliases = resp["aliases"]
    return aliases


def do_bulk_insert(collection_name: str, files: list, partition_name=None, timeout=None, using="default", **kwargs) -> int:
    """ do_bulk_insert inserts entities through files, currently supports row-based json file.
    User need to create the json file with a specified json format which is described in the official user guide.
    Let's say a collection has two fields: "id" and "vec"(dimension=8), the row-based json format is:
      {"rows": [
          {"id": "0", "vec": [0.190, 0.046, 0.143, 0.972, 0.592, 0.238, 0.266, 0.995]},
          {"id": "1", "vec": [0.149, 0.586, 0.012, 0.673, 0.588, 0.917, 0.949, 0.944]},
          ......
        ]
      }
    The json file must be uploaded to root path of MinIO/S3 storage which is accessed by milvus server.
    For example:
        the milvus.yml specify the MinIO/S3 storage bucketName as "a-bucket", user can upload his json file
         to a-bucket/xxx.json, then call do_bulk_insert(files=["a-bucket/xxx.json"])

    :param collection_name: the name of the collection
    :type  collection_name: str

    :param partition_name: the name of the partition
    :type  partition_name: str

    :param files: related path of the file to be imported, for row-based json file, only allow one file each invocation.
    :type  files: list[str]

    :param timeout: The timeout for this method, unit: second
    :type  timeout: int

    :param kwargs: other infos

    :return: id of the task
    :rtype:  int

    :raises BaseException: If collection_name doesn't exist.
    :raises BaseException: If the files input is illegal.
    """
    return _get_connection(using).do_bulk_insert(collection_name, partition_name, files, timeout=timeout, **kwargs)


def get_bulk_insert_state(task_id, timeout=None, using="default", **kwargs) -> BulkInsertState:
    """get_bulk_insert_state returns state of a certain task_id

    :param task_id: the task id returned by bulk_insert
    :type  task_id: int

    :return: BulkInsertState
    :rtype:  BulkInsertState
    """
    return _get_connection(using).get_bulk_insert_state(task_id, timeout=timeout, **kwargs)


def list_bulk_insert_tasks(limit=0, collection_name=None, timeout=None, using="default", **kwargs) -> list:
    """list_bulk_insert_tasks lists all bulk load tasks

    :param limit: maximum number of tasks returned, list all tasks if the value is 0, else return the latest tasks
    :type  limit: int

    :param collection_name: target collection name, list all tasks if the name is empty
    :type  collection_name: str

    :return: list[BulkInsertState]
    :rtype:  list[BulkInsertState]

    """
    return _get_connection(using).list_bulk_insert_tasks(limit, collection_name, timeout=timeout, **kwargs)


def reset_password(user: str, old_password: str, new_password: str, using="default"):
    """
        Reset the user & password of the connection.
        You must provide the original password to check if the operation is valid.
        Note: after this operation, the connection is also ready to use.

    :param user: the user of the Milvus connection.
    :type  user: str
    :param old_password: the original password of the Milvus connection.
    :type  old_password: str
    :param new_password: the newly password of this user.
    :type  new_password: str
    """
    return _get_connection(using).reset_password(user, old_password, new_password)


def create_user(user: str, password: str, using="default"):
    """ Create User using the given user and password.
    :param user: the user name.
    :type  user: str
    :param password: the password.
    :type  password: str
    """
    return _get_connection(using).create_user(user, password)


def update_password(user: str, old_password, new_password: str, using="default"):
    """
        Update user password using the given user and password.
        You must provide the original password to check if the operation is valid.
        Note: after this operation, PyMilvus won't change the related header of this connection.
        So if you update credential for this connection, the connection may be invalid.

    :param user: the user name.
    :type  user: str
    :param old_password: the original password.
    :type  old_password: str
    :param new_password: the newly password of this user.
    :type  new_password: str
    """
    return _get_connection(using).update_password(user, old_password, new_password)


def delete_user(user: str, using="default"):
    """ Delete User corresponding to the username.
    :param user: the user name.
    :type  user: str
    """
    return _get_connection(using).delete_user(user)


def list_usernames(using="default"):
    """ List all usernames.
    :return list of str:
        The usernames in Milvus instances.
    """
    return _get_connection(using).list_usernames()


def list_roles(include_user_info: bool, using="default"):
    """ List All Role Info
    :param include_user_info: whether to obtain the user information associated with roles
    :type  include_user_info: bool
    :return RoleInfo
    """
    return _get_connection(using).select_all_role(include_user_info)


def list_user(username: str, include_role_info: bool, using="default"):
    """ List One User Info
    :param username: user name.
    :type  username: str
    :param include_role_info: whether to obtain the role information associated with the user
    :type  include_role_info: bool
    :return UserInfo
    """
    return _get_connection(using).select_one_user(username, include_role_info)


def list_users(include_role_info: bool, using="default"):
    """ List All User Info
    :param include_role_info: whether to obtain the role information associated with users
    :type  include_role_info: bool
    :return UserInfo
    """
    return _get_connection(using).select_all_user(include_role_info)
