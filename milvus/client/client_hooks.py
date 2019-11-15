from .hooks import BaseaSearchHook
from .abstract import TopKQueryResult


class SearchHook(BaseaSearchHook):

    def handle_response(self, _response):
        """
        use class `TopKQueryResult` to deal with response from server
        """
        return TopKQueryResult(_response)
