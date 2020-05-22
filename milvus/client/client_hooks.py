from .hooks import BaseSearchHook
from .abstract import TopKQueryResult, HybridResult


class SearchHook(BaseSearchHook):

    def handle_response(self, _response):
        """
        use class `TopKQueryResult` to deal with response from server
        """
        return TopKQueryResult(_response)


class HybridSearchHook(BaseSearchHook):

    def handle_response(self, _response):
        return HybridResult(_response)
