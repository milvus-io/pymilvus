from .hooks import BaseaSearchHook
from .abstract import TopKQueryResult


class SearchHook(BaseaSearchHook):

    def handle_response(self, _response):
        return TopKQueryResult(_response)
