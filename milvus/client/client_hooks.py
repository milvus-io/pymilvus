from .hooks import BaseSearchHook
from .abstract import TopKQueryResult


class SearchHook(BaseSearchHook):

    def handle_response(self, _response):
        """
        use class `TopKQueryResult` to deal with response from server
        """
        return TopKQueryResult(_response)
