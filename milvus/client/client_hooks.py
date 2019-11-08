import datetime
from .hooks import BaseaSearchHook
from .abstract import TopKQueryBinResult, TopKQueryResult


class SearchHook(BaseaSearchHook):

    def __init__(self):
        self._time_stamp0 = None

    def pre_search(self, *args, **kwargs):
        super().pre_search(*args, **kwargs)
        self._time_stamp0 = datetime.datetime.now()
        print("[{}] <Intraval> Start search ....".format(self._time_stamp0))

    def aft_search(self, *args, **kwargs):
        super().aft_search(*args, **kwargs)
        time_stamp0 = datetime.datetime.now()
        print("[{}] <Interval> Search done ....".format(time_stamp0))
        time_r = time_stamp0 - self._time_stamp0
        print("Search interface cost {} ms".format(time_r.seconds * 1000 + time_r.microseconds // 1000))

    def on_response(self, *args, **kwargs):
        return super().on_response(*args, **kwargs)

    def handle_response(self, _response):
        return TopKQueryResult(_response)
        # return TopKQueryBinResult(_response)


class SearchBinHook(BaseaSearchHook):
    def pre_search(self, *args, **kwargs):
        super().pre_search(*args, **kwargs)

    def aft_search(self, *args, **kwargs):
        super().aft_search(*args, **kwargs)

    def on_response(self, *args, **kwargs):
        return super().on_response(*args, **kwargs)

    def handle_response(self, _response):
        return TopKQueryBinResult(_response)
