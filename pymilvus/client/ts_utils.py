import datetime
from typing import Any, Dict, Optional

from pymilvus.grpc_gen import common_pb2

from .cache import GlobalCache
from .constants import BOUNDED_TS, EVENTUALLY_TS, GUARANTEE_TIMESTAMP, ITERATOR_FIELD
from .types import get_consistency_level
from .utils import hybridts_to_unixtime

ConsistencyLevel = common_pb2.ConsistencyLevel


# Update the last write ts of collection.
def update_collection_ts(collection_name: str, ts: int, endpoint: str = "", db_name: str = ""):
    GlobalCache.collection_ts.set(endpoint, db_name, collection_name, ts)


# Return a callback corresponding to the collection.
def update_ts_on_mutation(collection_name: str, endpoint: str = "", db_name: str = ""):
    def _update(mutation_result: Any):
        update_collection_ts(collection_name, mutation_result.timestamp, endpoint, db_name)

    return _update


# Get the last write ts of collection.
def get_collection_ts(collection_name: str, endpoint: str = "", db_name: str = ""):
    return GlobalCache.collection_ts.get(endpoint, db_name, collection_name)


# Get the last write timestamp of collection.
def get_collection_timestamp(collection_name: str, endpoint: str = "", db_name: str = ""):
    ts = get_collection_ts(collection_name, endpoint, db_name)
    return hybridts_to_unixtime(ts)


# Get the last write datetime of collection.
def get_collection_datetime(
    collection_name: str,
    endpoint: str = "",
    db_name: str = "",
    tz: Optional[datetime.timezone] = None,
):
    timestamp = get_collection_timestamp(collection_name, endpoint, db_name)
    return datetime.datetime.fromtimestamp(timestamp, tz=tz)


def get_eventually_ts():
    return EVENTUALLY_TS


def get_bounded_ts():
    return BOUNDED_TS


def construct_guarantee_ts(
    collection_name: str, kwargs: Dict, endpoint: str = "", db_name: str = ""
):
    if kwargs.get(ITERATOR_FIELD) is not None:
        return True

    consistency_level = kwargs.get("consistency_level")
    use_default = consistency_level is None
    if use_default:
        # in case of the default consistency is Customized or Session,
        # we set guarantee_timestamp to the cached mutation ts or 1
        kwargs[GUARANTEE_TIMESTAMP] = (
            get_collection_ts(collection_name, endpoint, db_name) or get_eventually_ts()
        )
        return True
    consistency_level = get_consistency_level(consistency_level)
    kwargs["consistency_level"] = consistency_level
    if consistency_level == ConsistencyLevel.Strong:
        # Milvus will assign a newest ts.
        kwargs[GUARANTEE_TIMESTAMP] = 0
    elif consistency_level == ConsistencyLevel.Session:
        # Using the last write ts of the collection.
        # TODO: get a timestamp from server?
        kwargs[GUARANTEE_TIMESTAMP] = (
            get_collection_ts(collection_name, endpoint, db_name) or get_eventually_ts()
        )
    elif consistency_level == ConsistencyLevel.Bounded:
        # Milvus will assign ts according to the server timestamp and a configured time interval
        kwargs[GUARANTEE_TIMESTAMP] = get_bounded_ts()
    else:
        # Users customize the consistency level, no modification on `guarantee_timestamp`.
        kwargs.setdefault(GUARANTEE_TIMESTAMP, get_eventually_ts())
    return use_default
