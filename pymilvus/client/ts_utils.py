import threading
import datetime

from .singleton_utils import Singleton
from .utils import hybridts_to_unixtime, mkts_from_unixtime
from .constants import DEFAULT_GRACEFUL_TIME, EVENTUALLY_TS

from ..grpc_gen.common_pb2 import ConsistencyLevel


class GTsDict(metaclass=Singleton):
    def __init__(self):
        # collection id -> last write ts
        self._last_write_ts_dict = dict()
        self._last_write_ts_dict_lock = threading.Lock()

    def __repr__(self):
        return self._last_write_ts_dict.__repr__()

    def update(self, collection, ts):
        # use lru later if necessary
        with self._last_write_ts_dict_lock:
            if ts > self._last_write_ts_dict.get(collection, 0):
                self._last_write_ts_dict[collection] = ts

    def get(self, collection):
        return self._last_write_ts_dict.get(collection, 0)


# Return a GTsDict instance.
def _get_gts_dict():
    return GTsDict()


# Update the last write ts of collection.
def update_collection_ts(collection, ts):
    _get_gts_dict().update(collection, ts)


# Return a callback coresponding to the collection.
def update_ts_on_mutation(collection):
    def _update(mutation_result):
        update_collection_ts(collection, mutation_result.timestamp)

    return _update


# Get the last write ts of collection.
def get_collection_ts(collection):
    return _get_gts_dict().get(collection)


# Get the last write timestamp of collection.
def get_collection_timestamp(collection):
    ts = _get_gts_dict().get(collection)
    return hybridts_to_unixtime(ts)


# Get the last write datetime of collection.
def get_collection_datetime(collection, tz=None):
    timestamp = get_collection_timestamp(collection)
    return datetime.datetime.fromtimestamp(timestamp, tz=tz)


# Get the bounded timestamp according to the current timestamp.
# The default graceful time is 3000ms (greater than the time tick period, bigger enough).
def get_current_bounded_ts(graceful_time_in_ms=DEFAULT_GRACEFUL_TIME):
    # Fortunately, we're in 21th century and we don't need to worry about that bounded_ts < 0.
    current = datetime.datetime.now().timestamp()
    return mkts_from_unixtime(current, -graceful_time_in_ms)


def get_eventually_ts():
    return EVENTUALLY_TS


def construct_guarantee_ts(consistency_level, collection_id, kwargs):
    if consistency_level == ConsistencyLevel.Strong:
        # Milvus will assign a newest ts.
        kwargs["guarantee_timestamp"] = 0
    elif consistency_level == ConsistencyLevel.Session:
        # Using the last write ts of the collection.
        kwargs["guarantee_timestamp"] = get_collection_ts(collection_id) or get_eventually_ts()
    elif consistency_level == ConsistencyLevel.Bounded:
        # Using a timestamp which is close to but less than the current epoch.
        graceful_time = kwargs.get("graceful_time", DEFAULT_GRACEFUL_TIME)
        kwargs["guarantee_timestamp"] = get_current_bounded_ts(graceful_time) or get_eventually_ts()
    elif consistency_level == ConsistencyLevel.Eventually:
        # Using a very small timestamp.
        kwargs["guarantee_timestamp"] = get_eventually_ts()
    else:
        # Users customize the consistency level, no modification on `guarantee_timestamp`.
        pass
