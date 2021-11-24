import threading

from .singleton_utils import Singleton


class GTsDict(metaclass=Singleton):
	def __init__(self):
		# collection id -> last write ts
		self._last_write_ts_dict = dict()
		self._last_write_ts_dict_lock = threading.Lock()

	def __repr__(self):
		with self._last_write_ts_dict_lock:
			return self._last_write_ts_dict.__repr__()

	def Update(self, collection, ts):
		# use lru later if necessary
		with self._last_write_ts_dict_lock:
			if ts > self._last_write_ts_dict.get(collection, 0):
				self._last_write_ts_dict[collection] = ts

	def Get(self, collection):
		with self._last_write_ts_dict_lock:
			return self._last_write_ts_dict.get(collection, 0)


# Return a GTsDict instance.
def _get_gts_dict():
	return GTsDict()


# Update the last write ts of collection.
def Update(collection, ts):
	_get_gts_dict().Update(collection, ts)


# Return a callback coresponding to the collection.
def UpdateOnMutation(collection):
	def _update(mutation_result):
		Update(collection, mutation_result.timestamp)
	return _update


# Get the last write ts of collection.
def Get(collection):
	return _get_gts_dict().Get(collection)
