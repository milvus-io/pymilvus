import datetime
from .hooks import BaseaSearchHook
from .abstract import TopKQueryResult


class SearchHook(BaseaSearchHook):

    def __init__(self):
        self._time_stamp0 = None

    def pre_search(self, *args, **kwargs):
        super().pre_search(*args, **kwargs)

    def aft_search(self, *args, **kwargs):
        super().aft_search(*args, **kwargs)

    def on_response(self, *args, **kwargs):
        return super().on_response(*args, **kwargs)

    def handle_response(self, _response):
        return TopKQueryResult(_response)
